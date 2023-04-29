/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala, Oskar Lappi.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "astaroth_utils.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "errchk.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define fabs(x) ((_Generic((x), float : fabsf, double : fabs, long double : fabsl))(x))

// Defines for colored output
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

static inline bool
is_valid(const AcReal a)
{
    return !isnan(a) && !isinf(a);
}

Error
acGetError(const AcReal model, const AcReal candidate)
{
    Error error;
    error.abs_error = 0;

    error.model     = model;
    error.candidate = candidate;

    if (error.model == error.candidate ||
        fabsl((long double)model - (long double)candidate) == 0) { // If exact
        error.abs_error = 0;
        error.rel_error = 0;
        error.ulp_error = 0;
    }
    else if (!is_valid(error.model) || !is_valid(error.candidate)) {
        error.abs_error = (long double)INFINITY;
        error.rel_error = (long double)INFINITY;
        error.ulp_error = (long double)INFINITY;
    }
    else {
        const int base = 2;
        const int p    = sizeof(AcReal) == 4 ? 24 : 53; // Bits in the significant

        const long double e = floorl(logl(fabsl((long double)error.model)) / logl(2));

        const long double ulp             = powl(base, e - (p - 1));
        const long double machine_epsilon = 0.5l * powl(base, -(p - 1));
        error.abs_error                   = fabsl((long double)model - (long double)candidate);
        error.ulp_error                   = error.abs_error / ulp;
        error.rel_error = fabsl(1.0l - (long double)candidate / (long double)model) /
                          machine_epsilon;
    }

    error.maximum_magnitude = error.minimum_magnitude = 0;

    return error;
}

static inline void
print_error_to_file(const char* label, const Error error, const char* path)
{
    FILE* file = fopen(path, "a");
    ERRCHK_ALWAYS(file);
    fprintf(file, "%s, %Lg, %Lg, %Lg, %g, %g\n", label, error.ulp_error, error.abs_error,
            error.rel_error, (double)error.maximum_magnitude, (double)error.minimum_magnitude);
    fclose(file);
}

/** Returns true if the error is acceptable, false otherwise. */
bool
acEvalError(const char* label, const Error error)
{
    // Accept the error if the relative error is < max_ulp_error ulps.
    // Also consider the error zero if it is less than the minimum value in the mesh scaled to
    // machine epsilon
    const long double max_ulp_error = 5;

    bool acceptable;
    if (error.ulp_error < max_ulp_error)
        acceptable = true;
    else if (error.abs_error < (long double)error.minimum_magnitude * (long double)AC_REAL_EPSILON)
        acceptable = true;
    else
        acceptable = false;

    printf("%-15s... %s ", label, acceptable ? GRN "OK! " RESET : RED "FAIL! " RESET);

    printf("| %.3Lg (abs), %.3Lg (ulps), %.3Lg (rel). Range: [%.3g, %.3g]\n", //
           error.abs_error, error.ulp_error, error.rel_error,                 //
           (double)error.minimum_magnitude, (double)error.maximum_magnitude);
    print_error_to_file(label, error, "verification.out");

    return acceptable;
}

static AcReal
get_maximum_magnitude(const AcReal* field, const AcMeshInfo info)
{
    AcReal maximum = (AcReal)-INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        maximum = max(maximum, fabs(field[i]));

    return maximum;
}

static AcReal
get_minimum_magnitude(const AcReal* field, const AcMeshInfo info)
{
    AcReal minimum = (AcReal)INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        minimum = min(minimum, fabs(field[i]));

    return minimum;
}

// Get the maximum absolute error. Works well if all the values in the mesh are approximately
// in the same range.
// Finding the maximum ulp error is not useful, as it picks up on the noise beyond the
// floating-point precision range and gives huge errors with values that should be considered
// zero (f.ex. 1e-19 and 1e-22 give error of around 1e4 ulps)
static Error
get_max_abs_error(const AcReal* model, const AcReal* candidate, const AcMeshInfo info)
{
    Error error = {.abs_error = -1};

    for (size_t i = 0; i < acVertexBufferSize(info); ++i) {
        Error curr_error = acGetError(model[i], candidate[i]);
        if (curr_error.abs_error > error.abs_error)
            error = curr_error;
    }
    error.maximum_magnitude = get_maximum_magnitude(model, info);
    error.minimum_magnitude = get_minimum_magnitude(model, info);

    return error;
}

/** Returns true when successful, false if errors were found. */
AcResult
acVerifyMesh(const char* label, const AcMesh model, const AcMesh candidate)
{
    printf("---Test: %s---\n", label);
    fflush(stdout);
    printf("Errors at the point of the maximum absolute error:\n");

    int errors_found = 0;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        const Error error = get_max_abs_error(model.vertex_buffer[i], candidate.vertex_buffer[i],
                                              model.info);
        const bool acceptable = acEvalError(vtxbuf_names[i], error);
        if (!acceptable)
            ++errors_found;
    }

    if (errors_found > 0)
        printf("Failure. Found %d errors\n", errors_found);

    return errors_found ? AC_FAILURE : AC_SUCCESS;
}

/** Writes an error slice in the z direction */
AcResult
acMeshDiffWriteSliceZ(const char* path, const AcMesh model, const AcMesh candidate, const size_t z)
{
    ERRCHK_ALWAYS(NUM_VTXBUF_HANDLES > 0);

    const AcMeshInfo info = model.info;
    ERRCHK_ALWAYS((int)z < info.int_params[AC_mz]);

    FILE* fp = fopen(path, "w");
    ERRCHK_ALWAYS(fp);

    const size_t mx = info.int_params[AC_mx];
    const size_t my = info.int_params[AC_my];
    for (size_t y = 0; y < my; ++y) {
        for (size_t x = 0; x < mx; ++x) {
            const size_t idx                = acVertexBufferIdx(x, y, z, info);
            const VertexBufferHandle vtxbuf = 0;
            const AcReal m                  = model.vertex_buffer[vtxbuf][idx];
            const AcReal c                  = candidate.vertex_buffer[vtxbuf][idx];
            const Error error               = acGetError(m, c);
            fprintf(fp, "%Lg ", error.ulp_error); // error.abs_error);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return AC_SUCCESS;
}

/** Writes out the entire diff of two meshes */
AcResult
acMeshDiffWrite(const char* path, const AcMesh model, const AcMesh candidate)
{
    ERRCHK_ALWAYS(NUM_VTXBUF_HANDLES > 0);

    const AcMeshInfo info = model.info;

    FILE* fp = fopen(path, "w");
    ERRCHK_ALWAYS(fp);

    const size_t mx = info.int_params[AC_mx];
    const size_t my = info.int_params[AC_my];
    const size_t mz = info.int_params[AC_mz];
    for (size_t z = 0; z < mz; ++z) {
        for (size_t y = 0; y < my; ++y) {
            for (size_t x = 0; x < mx; ++x) {
                const size_t idx                = acVertexBufferIdx(x, y, z, info);
                const VertexBufferHandle vtxbuf = 0;
                const AcReal m                  = model.vertex_buffer[vtxbuf][idx];
                const AcReal c                  = candidate.vertex_buffer[vtxbuf][idx];
                const Error error               = acGetError(m, c);
                fprintf(fp, "%Lg ", error.ulp_error); // error.abs_error);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n\n");
    }
    fprintf(fp, "\nSTEP_BOUNDARY\n");

    fclose(fp);
    return AC_SUCCESS;
}
