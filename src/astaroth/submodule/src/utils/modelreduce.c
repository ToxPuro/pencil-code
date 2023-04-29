/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "astaroth.h"

#include <math.h>

#include "errchk.h"

#if AC_DOUBLE_PRECISION == 0 // HACK TODO fix, make cleaner (purkkaratkaisu)
#define fabs fabsf
#define exp expf
#define sqrt sqrtf
#endif

// Function pointer definitions
typedef long double (*ReduceFunc)(const long double, const long double);
typedef long double (*ReduceInitialScalFunc)(const long double);
typedef long double (*ReduceInitialVecFunc)(const long double, const long double,
                                            const long double);
typedef long double (*ReduceInitialVecScalFunc)(const long double, const long double,
                                                const long double, const long double);

// clang-format off
/* Comparison funcs */
static inline long double
max(const long double a, const long double b) { return a > b ? a : b; }

static inline long double
min(const long double a, const long double b) { return a < b ? a : b; }

static inline long double
sum(const long double a, const long double b) { return a + b; }

/* Function used to determine the values used during reduction */
static inline long double
length_scal(const long double a) { return (long double)(a); }

static inline long double
length_vec(const long double a, const long double b, const long double c) { return sqrtl(a*a + b*b + c*c); }

static inline long double
squared_scal(const long double a) { return (long double)(a*a); }

static inline long double
squared_vec(const long double a, const long double b, const long double c) { return squared_scal(a) + squared_scal(b) + squared_scal(c); }

static inline long double
exp_squared_scal(const long double a) { return expl(a)*expl(a); }

static inline long double
exp_squared_vec(const long double a, const long double b, const long double c) { return exp_squared_scal(a) + exp_squared_scal(b) + exp_squared_scal(c); }

static inline long double
length_alf(const long double a, const long double b, const long double c, const long double d) { return sqrtl(a*a + b*b + c*c)/sqrtl(expl(d)); }

static inline long double
squared_alf(const long double a, const long double b, const long double c, const long double d) { return (squared_scal(a) + squared_scal(b) + squared_scal(c))/(expl(d)); }
// clang-format on

AcReal
acHostReduceScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a)
{
    ReduceInitialScalFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length_scal;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length_scal;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared_scal;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared_scal;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length_scal;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    long double res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; ++k) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; ++j) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 ++i) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial(
                    (long double)mesh.vertex_buffer[a][idx]);
                res = reduce(res, curr_val);
            }
        }
    }
    // fprintf(stderr, "%s host result %g\n", rtype_names[rtype], res);
    if (solve_mean) {
        const long double inv_n = 1.0l / mesh.info.int_params[AC_nxyz];
        return sqrtl(inv_n * res);
    }
    else {
        return res;
    }
}

AcReal
acHostReduceVec(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                const VertexBufferHandle b, const VertexBufferHandle c)
{
    // AcReal (*reduce_initial)(AcReal, AcReal, AcReal);
    ReduceInitialVecFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_MAX:
        reduce_initial = length_vec;
        reduce         = max;
        break;
    case RTYPE_MIN:
        reduce_initial = length_vec;
        reduce         = min;
        break;
    case RTYPE_RMS:
        reduce_initial = squared_vec;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_RMS_EXP:
        reduce_initial = exp_squared_vec;
        reduce         = sum;
        solve_mean     = true;
        break;
    case RTYPE_SUM:
        reduce_initial = length_vec;
        reduce         = sum;
        break;
    default:
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    long double res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx],
                             (long double)mesh.vertex_buffer[b][initial_idx],
                             (long double)mesh.vertex_buffer[c][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; k++) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; j++) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 i++) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial((long double)mesh.vertex_buffer[a][idx],
                                                            (long double)mesh.vertex_buffer[b][idx],
                                                            (long double)
                                                                mesh.vertex_buffer[c][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const long double inv_n = (long double)1.0 / mesh.info.int_params[AC_nxyz];
        return sqrt(inv_n * res);
    }
    else {
        return res;
    }
}

AcReal
acHostReduceVecScal(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a,
                    const VertexBufferHandle b, const VertexBufferHandle c,
                    const VertexBufferHandle d)
{
    // AcReal (*reduce_initial)(AcReal, AcReal, AcReal);
    ReduceInitialVecScalFunc reduce_initial;
    ReduceFunc reduce;

    bool solve_mean = false;

    switch (rtype) {
    case RTYPE_ALFVEN_MAX:
        reduce_initial = length_alf;
        reduce         = max;
        break;
    case RTYPE_ALFVEN_MIN:
        reduce_initial = length_alf;
        reduce         = min;
        break;
    case RTYPE_ALFVEN_RMS:
        reduce_initial = squared_alf;
        reduce         = sum;
        solve_mean     = true;
        break;
    default:
        fprintf(stderr, "rtype %s %d\n", rtype_names[rtype], rtype);
        ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = acVertexBufferIdx(mesh.info.int_params[AC_nx_min],
                                              mesh.info.int_params[AC_ny_min],
                                              mesh.info.int_params[AC_nz_min], mesh.info);

    long double res;
    if (rtype == RTYPE_MAX || rtype == RTYPE_MIN || rtype == RTYPE_ALFVEN_MAX ||
        rtype == RTYPE_ALFVEN_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx],
                             (long double)mesh.vertex_buffer[b][initial_idx],
                             (long double)mesh.vertex_buffer[c][initial_idx],
                             (long double)mesh.vertex_buffer[d][initial_idx]);
    else
        res = 0;

    for (int k = mesh.info.int_params[AC_nz_min]; k < mesh.info.int_params[AC_nz_max]; k++) {
        for (int j = mesh.info.int_params[AC_ny_min]; j < mesh.info.int_params[AC_ny_max]; j++) {
            for (int i = mesh.info.int_params[AC_nx_min]; i < mesh.info.int_params[AC_nx_max];
                 i++) {
                const int idx              = acVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial((long double)mesh.vertex_buffer[a][idx],
                                                            (long double)mesh.vertex_buffer[b][idx],
                                                            (long double)mesh.vertex_buffer[c][idx],
                                                            (long double)
                                                                mesh.vertex_buffer[d][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (solve_mean) {
        const long double inv_n = (long double)1.0 / mesh.info.int_params[AC_nxyz];
        return sqrtl(inv_n * res);
    }
    else {
        return res;
    }
}
