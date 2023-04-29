/*
   Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

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
#include "run.h"

#include <stdio.h>

#include "config_loader.h"
#include "math_utils.h"
#include "model/host_forcing.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_boundconds.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"

#include "errchk.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

// Defines for colored output
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

#define GEN_TEST_RESULT (1) // Generate a test file always during testing

typedef struct {
    int x, y, z;
} vec3i;

typedef struct {
    AcReal x, y, z;
} vec3r;

typedef struct {
    ModelScalar model;
    AcReal candidate;
    ModelScalar error;
} ErrorInfo;

#define QUICK_TEST (0)
#define THOROUGH_TEST (1)
#define TEST_TYPE QUICK_TEST

static const InitType test_cases[] = {INIT_TYPE_RANDOM, INIT_TYPE_XWAVE,
                                      INIT_TYPE_GAUSSIAN_RADIAL_EXPL, INIT_TYPE_ABC_FLOW};
// #define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#ifndef AC_DOUBLE_PRECISION
#define AC_DOUBLE_PRECISION (0)
#endif

static inline bool
is_valid(const ModelScalar a)
{
    return !isnan(a) && !isinf(a);
}

#if TEST_TYPE ==                                                                                   \
    QUICK_TEST // REGULAR TEST START HERE
               // --------------------------------------------------------------------------------------------------------------
static inline ModelScalar
get_absolute_error(const ModelScalar& model, const AcReal& candidate)
{
    return fabsl(candidate - model);
}

static inline ModelScalar
get_acceptable_absolute_error(const ModelScalar& range)
{
    // This is the upper limit, which assumes that both the min and max values
    // are used in a calculation (which inherently leads to cancellation).
    //
    // AFAIK if this breaks, there is definitely something wrong with the code.
    // Otherwise the error is so small it's indistiguishable from inherent
    // inaccuracies in floating-point arithmetic.
    return range * AC_REAL_EPSILON;
}

static inline ModelScalar
get_acceptable_relative_error(void)
{
    return 30; // machine epsilons
}

static inline ModelScalar
get_relative_error(const ModelScalar& model, const AcReal& candidate)
{
    ModelScalar error = NAN;

#if 0
	const ModelScalar abs_epsilon = get_acceptable_absolute_error(range);
	if (fabsl(model) < abs_epsilon) { // Model is close to zero
		/*
		   if (fabsl(candidate - model) <= AC_REAL_EPSILON * fabsl(candidate))
		   error = 0;
		// Knuth section 4.2.2 pages 217-218 TODO
		 */
		if (fabsl(candidate) < abs_epsilon) // If candidate is close to zero
			error = fabsl(candidate);       // return candidate itself
		else
			error = INFINITY;
	}
	else {
		error = fabsl(1.0l - candidate / model);
	}
#endif
    error = fabsl(1.0l - candidate / model);

    // Return the relative error as multiples of the machine epsilon
    // See Sect. Relative Error and Ulps in
    // What Every Computer Scientist Should Know About Floating-Point Arithmetic
    // By David Goldberg (1991)
    return error / AC_REAL_EPSILON;
}

static bool
verify(const ModelScalar& model, const AcReal& cand, const ModelScalar& range)
{
    if (!is_valid(model) || !is_valid(cand))
        return false;

    const ModelScalar relative_error = get_relative_error(model, cand);
    if (relative_error < get_acceptable_relative_error())
        return true;

    const ModelScalar absolute_error = get_absolute_error(model, cand);
    if (absolute_error < get_acceptable_absolute_error(range))
        return true;

    return false;
}

static ModelScalar
get_reduction_range(const ModelMesh& mesh)
{
    ERRCHK(NUM_VTXBUF_HANDLES >= 3);

    const ModelScalar max0     = model_reduce_scal(mesh, RTYPE_MAX, VertexBufferHandle(0));
    const ModelScalar max1     = model_reduce_scal(mesh, RTYPE_MAX, VertexBufferHandle(1));
    const ModelScalar max2     = model_reduce_scal(mesh, RTYPE_MAX, VertexBufferHandle(2));
    const ModelScalar max_scal = max(max0, max(max1, max2));

    const ModelScalar min0     = model_reduce_scal(mesh, RTYPE_MIN, VertexBufferHandle(0));
    const ModelScalar min1     = model_reduce_scal(mesh, RTYPE_MIN, VertexBufferHandle(1));
    const ModelScalar min2     = model_reduce_scal(mesh, RTYPE_MIN, VertexBufferHandle(2));
    const ModelScalar min_scal = min(min0, min(min1, min2));

    return max_scal - min_scal;
}

static void
print_debug_info(const ModelScalar& model, const AcReal& candidate, const ModelScalar& range)
{
    printf("MeshPointInfo\n");
    printf("\tModel: %e\n", double(model));
    printf("\tCandidate: %e\n", double(candidate));
    printf("\tRange: %e\n", double(range));

    printf("\tAbsolute error: %Le (max acceptable: %Le)\n", get_absolute_error(model, candidate),
           get_acceptable_absolute_error(range));
    printf("\tRelative error: %Le (max acceptable: %Le)\n", get_relative_error(model, candidate),
           get_acceptable_relative_error());
    printf("\tIs acceptable: %d\n", verify(model, candidate, range));
}

static void
print_result(const ModelScalar& model, const AcReal& candidate, const ModelScalar& range,
             const char* name = "???")
{
    const ModelScalar rel_err = get_relative_error(model, candidate);
    const ModelScalar abs_err = get_absolute_error(model, candidate);
    if (!verify(model, candidate, range)) {
        printf("\t%-12s... ", name);
        printf(RED "FAIL! " RESET);
    }
    else {
        printf("\t%-12s... ", name);
        printf(GRN "OK! " RESET);
    }

    printf("(relative error: %.3Lg \u03B5, absolute error: %Lg)\n", rel_err, abs_err);
    /*
    // DEPRECATED: TODO remove
    if (rel_err < get_acceptable_relative_error())
    printf("(relative error: %Lg \u03B5, max accepted %Lg)\n", rel_err,
    get_acceptable_relative_error());
    else
    printf("(absolute error: %Lg, max accepted %Lg)\n", abs_err,
    get_acceptable_absolute_error(range));
     */
}

static int
check_reductions(const AcMeshInfo& config)
{
    printf("Testing reductions\n");
    int num_failures = 0;

    // Init CPU meshes
    AcMesh* mesh         = acmesh_create(config);
    ModelMesh* modelmesh = modelmesh_create(config);

    // Init GPU meshes
    acInit(config);

    for (unsigned int i = 0; i < ARRAY_SIZE(test_cases); ++i) {
        const InitType itype = test_cases[i];
        printf("Checking %s...\n", init_type_names[InitType(itype)]);

        // Init the mesh and figure out the acceptable range for error
        acmesh_init_to(InitType(itype), mesh);

        acmesh_to_modelmesh(*mesh, modelmesh);
        const ModelScalar range = get_reduction_range(*modelmesh);

        acLoad(*mesh);

        for (int rtype = 0; rtype < NUM_RTYPES; ++rtype) {

            if (rtype == RTYPE_SUM) {
                // Skip SUM test for now. The failure is either caused by floating-point
                // cancellation or an actual issue
                WARNING("Skipping RTYPE_SUM test\n");
                continue;
            }

            const VertexBufferHandle ftype = VTXBUF_UUX;

            // Scal
            ModelScalar model = model_reduce_scal(*modelmesh, ReductionType(rtype),
                                                  VertexBufferHandle(ftype));
            AcReal candidate  = acReduceScal(ReductionType(rtype), VertexBufferHandle(ftype));
            print_result(model, candidate, range, "UUX scal");

            bool is_acceptable = verify(model, candidate, range);
            if (!is_acceptable) {
                ++num_failures;

                // Print debug info
                printf("Scalar reduction type %d FAIL\n", rtype);
                print_debug_info(model, candidate, range);
            }

            // Vec
            model     = model_reduce_vec(*modelmesh, ReductionType(rtype), VTXBUF_UUX, VTXBUF_UUY,
                                         VTXBUF_UUZ);
            candidate = acReduceVec(ReductionType(rtype), VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
            print_result(model, candidate, range, "UUXYZ vec");

            is_acceptable = verify(model, candidate, range);
            if (!is_acceptable) {
                ++num_failures;

                // Print debug info
                printf("Vector reduction type %d FAIL\n", rtype);
                print_debug_info(model, candidate, range);
            }
        }

        printf("Acceptable relative error: < %Lg \u03B5, absolute error < %Lg\n",
               get_acceptable_relative_error(), get_acceptable_absolute_error(range));
    }
    acQuit();
    modelmesh_destroy(modelmesh);
    acmesh_destroy(mesh);

    return num_failures;
}

/** Finds the maximum and minimum in all meshes and computes the range.
 * Note! Potentially dangerous if all meshes do not interact with each other.
 * Otherwise the range may be too high.
 */
static ModelScalar
get_data_range(const ModelMesh& model)
{
    ModelScalar vertex_buffer_max_all = -INFINITY;
    ModelScalar vertex_buffer_min_all = INFINITY;
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const ModelScalar vertex_buffer_max = model_reduce_scal(model, RTYPE_MAX,
                                                                VertexBufferHandle(w));
        const ModelScalar vertex_buffer_min = model_reduce_scal(model, RTYPE_MIN,
                                                                VertexBufferHandle(w));

        if (vertex_buffer_max > vertex_buffer_max_all)
            vertex_buffer_max_all = vertex_buffer_max;
        if (vertex_buffer_min < vertex_buffer_min_all)
            vertex_buffer_min_all = vertex_buffer_min;
    }
    return fabsl(vertex_buffer_max_all - vertex_buffer_min_all);
}

// #define GEN_TEST_RESULT
#if GEN_TEST_RESULT == 1
static FILE* test_result = NULL;
#endif

static bool
verify_meshes(const ModelMesh& model, const AcMesh& candidate)
{
    bool retval = true;

#if GEN_TEST_RESULT == 1
    ErrorInfo err = ErrorInfo();
#endif

    const ModelScalar range = get_data_range(model);
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(model.info);

        // Maximum errors
        ErrorInfo max_abs_error = ErrorInfo();
        ErrorInfo max_rel_error = ErrorInfo();

        for (size_t i = 0; i < n; ++i) {
            const ModelScalar model_val = model.vertex_buffer[VertexBufferHandle(w)][i];
            const AcReal cand_val       = candidate.vertex_buffer[VertexBufferHandle(w)][i];

            if (!verify(model_val, cand_val, range)) {
                const int i0 = i % model.info.int_params[AC_mx];
                const int j0 = ((i %
                                 (model.info.int_params[AC_mx] * model.info.int_params[AC_my])) /
                                model.info.int_params[AC_mx]);
                const int k0 = i / (model.info.int_params[AC_mx] * model.info.int_params[AC_my]);
                printf("Index (%d, %d, %d)\n", i0, j0, k0);
                print_debug_info(model_val, cand_val, range);
                retval = false;
                printf("Breaking\n");
                break;
            }

            const ModelScalar abs_error = get_absolute_error(model_val, cand_val);
            if (abs_error > max_abs_error.error) {
                max_abs_error.error     = abs_error;
                max_abs_error.model     = model_val;
                max_abs_error.candidate = cand_val;
            }

            const ModelScalar rel_error = get_relative_error(model_val, cand_val);
            if (rel_error > max_rel_error.error) {
                max_rel_error.error     = rel_error;
                max_rel_error.model     = model_val;
                max_rel_error.candidate = cand_val;
            }

#if GEN_TEST_RESULT == 1
            if (abs_error > err.error) {
                err.error     = abs_error;
                err.model     = model_val;
                err.candidate = cand_val;
            }
#endif
        }
        // print_result(max_rel_error.model, max_rel_error.candidate, range,
        // vtxbuf_names[VertexBufferHandle(w)]);
        print_result(max_abs_error.model, max_abs_error.candidate, range,
                     vtxbuf_names[VertexBufferHandle(w)]);
    }

#if GEN_TEST_RESULT == 1
    const ModelScalar rel_err = get_relative_error(err.model, err.candidate);
    const ModelScalar abs_err = get_absolute_error(err.model, err.candidate);
    fprintf(test_result, "%.3Lg & %.3Lg\n", abs_err, rel_err);
#endif

    printf("Acceptable relative error: < %Lg \u03B5, absolute error < %Lg\n",
           get_acceptable_relative_error(), get_acceptable_absolute_error(range));

    return retval;
}

int
check_rk3(const AcMeshInfo& mesh_info)
{
    const int num_iterations = 1; // Note: should work up to at least 15 steps
    printf("Testing RK3 (running %d steps before checking the result)\n", num_iterations);
    int num_failures = 0;

    // Init CPU meshes
    AcMesh* gpu_mesh      = acmesh_create(mesh_info);
    ModelMesh* model_mesh = modelmesh_create(mesh_info);

    // Init GPU meshes
    acInit(mesh_info);

    for (unsigned int i = 0; i < ARRAY_SIZE(test_cases); ++i) {
        const InitType itype = test_cases[i];
        printf("Checking %s...\n", init_type_names[InitType(itype)]);

        // Init the mesh and figure out the acceptable range for error
        acmesh_init_to(InitType(itype), gpu_mesh);

        acLoad(*gpu_mesh);
        acmesh_to_modelmesh(*gpu_mesh, model_mesh);

        acBoundcondStep();
        boundconds(model_mesh->info, model_mesh);

        for (int j = 0; j < num_iterations; ++j) {
            // const AcReal umax = AcReal(acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY,
            // VTXBUF_UUZ));
            // const AcReal dt   = host_timestep(umax, mesh_info);
            const AcReal dt = AcReal(1e-2); // Use a small constant timestep to avoid instabilities

#if LFORCING
            const ForcingParams forcing_params = generateForcingParams(model_mesh->info);
            loadForcingParamsToHost(forcing_params, model_mesh);
            loadForcingParamsToDevice(forcing_params);
#endif

            acIntegrate(dt);

            model_rk3(dt, model_mesh);
        }
        boundconds(model_mesh->info, model_mesh);
        acBoundcondStep();
        acStore(gpu_mesh);

        bool is_acceptable = verify_meshes(*model_mesh, *gpu_mesh);
        if (!is_acceptable) {
            ++num_failures;
        }
    }

    acQuit();
    acmesh_destroy(gpu_mesh);
    modelmesh_destroy(model_mesh);

    return num_failures;
}

int
run_autotest(const char* config_path)
{
#if GEN_TEST_RESULT == 1
    char testresult_path[256];
    sprintf(testresult_path, "%s_fullstep_testresult.out",
            AC_DOUBLE_PRECISION ? "double" : "float");

    test_result = fopen(testresult_path, "w");
    ERRCHK(test_result);

    fprintf(test_result, "n, max abs error, corresponding rel error\n");
#endif

    /* Parse configs */
    AcMeshInfo config;
    load_config(config_path, &config);

    if (STENCIL_ORDER > 6)
        printf("WARNING!!! If the stencil order is larger than the computational domain some "
               "vertices may be done twice (f.ex. doing inner and outer domains separately and "
               "some of the front/back/left/right/etc slabs collide). The mesh must be large "
               "enough s.t. this doesn't happen.");

    const vec3i test_dims[] = {{32, 32, 32}, {64, 32, 32}, {32, 64, 32}, {32, 32, 64},
                               {64, 64, 32}, {64, 32, 64}, {32, 64, 64}};

    int num_failures = 0;
    for (size_t i = 0; i < ARRAY_SIZE(test_dims); ++i) {
        config.int_params[AC_nx] = test_dims[i].x;
        config.int_params[AC_ny] = test_dims[i].y;
        config.int_params[AC_nz] = test_dims[i].z;
        update_config(&config);

        printf("Testing mesh (%d, %d, %d):\n", //
               test_dims[i].x, test_dims[i].y, test_dims[i].z);

        num_failures += check_reductions(config);
        fflush(stdout);
    }

    for (size_t i = 0; i < ARRAY_SIZE(test_dims); ++i) {
        config.int_params[AC_nx] = test_dims[i].x;
        config.int_params[AC_ny] = test_dims[i].y;
        config.int_params[AC_nz] = test_dims[i].z;
        update_config(&config);

        printf("Testing mesh (%d, %d, %d):\n", //
               test_dims[i].x, test_dims[i].y, test_dims[i].z);

        num_failures += check_rk3(config);
        fflush(stdout);
    }

    printf("\n--------Testing done---------\n");
    printf("Failures found: %d\n", num_failures);

#if GEN_TEST_RESULT == 1
    fflush(test_result);
    fclose(test_result);
#endif

    if (num_failures > 0)
        return EXIT_FAILURE;
    else
        return EXIT_SUCCESS;
}

#elif TEST_TYPE ==                                                                                 \
    THOROUGH_TEST // GEN TEST FILE START HERE
                  // --------------------------------------------------------------------------------------------------------------
typedef struct {
    ModelScalar model;
    AcReal candidate;
    ModelScalar abs_error;
    ModelScalar ulp_error;
    ModelScalar rel_error;
    ModelScalar maximum_magnitude;
    ModelScalar minimum_magnitude;
} Error;

Error
get_error(ModelScalar model, AcReal candidate)
{
    Error error;
    error.abs_error = 0;

    error.model     = model;
    error.candidate = candidate;

    if (error.model == error.candidate || fabsl(model - candidate) == 0) { // If exact
        error.abs_error = 0;
        error.rel_error = 0;
        error.ulp_error = 0;
    }
    else if (!is_valid(error.model) || !is_valid(error.candidate)) {
        error.abs_error = INFINITY;
        error.rel_error = INFINITY;
        error.ulp_error = INFINITY;
    }
    else {
        const int base = 2;
        const int p    = sizeof(AcReal) == 4 ? 24 : 53; // Bits in the significant

        const ModelScalar e = floorl(logl(fabsl(error.model)) / logl(2));

        const ModelScalar ulp             = powl(base, e - (p - 1));
        const ModelScalar machine_epsilon = 0.5 * powl(base, -(p - 1));
        error.abs_error                   = fabsl(model - candidate);
        error.ulp_error                   = error.abs_error / ulp;
        error.rel_error                   = fabsl(1.0l - candidate / model) / machine_epsilon;
    }

    return error;
}

Error
get_max_abs_error_mesh(const ModelMesh& model_mesh, const AcMesh& candidate_mesh)
{
    Error error;
    error.abs_error = -1;

    for (size_t j = 0; j < NUM_VTXBUF_HANDLES; ++j) {
        for (size_t i = 0; i < acVertexBufferSize(model_mesh.info); ++i) {
            Error curr_error = get_error(model_mesh.vertex_buffer[j][i],
                                         candidate_mesh.vertex_buffer[j][i]);
            if (curr_error.abs_error > error.abs_error)
                error = curr_error;
        }
    }

    error.maximum_magnitude = -1; // Not calculated.
    error.minimum_magnitude = -1; // Not calculated.

    return error;
}

static ModelScalar
get_maximum_magnitude(const ModelScalar* field, const AcMeshInfo info)
{
    ModelScalar maximum = -INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        maximum = max(maximum, fabsl(field[i]));

    return maximum;
}

static ModelScalar
get_minimum_magnitude(const ModelScalar* field, const AcMeshInfo info)
{
    ModelScalar minimum = INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        minimum = min(minimum, fabsl(field[i]));

    return minimum;
}

Error
get_max_abs_error_vtxbuf(const VertexBufferHandle vtxbuf_handle, const ModelMesh& model_mesh,
                         const AcMesh& candidate_mesh)
{
    ModelScalar* model_vtxbuf = model_mesh.vertex_buffer[vtxbuf_handle];
    AcReal* candidate_vtxbuf  = candidate_mesh.vertex_buffer[vtxbuf_handle];

    Error error;
    error.abs_error = -1;

    for (size_t i = 0; i < acVertexBufferSize(model_mesh.info); ++i) {

        Error curr_error = get_error(model_vtxbuf[i], candidate_vtxbuf[i]);

        if (curr_error.abs_error > error.abs_error)
            error = curr_error;
    }

    error.maximum_magnitude = get_maximum_magnitude(model_vtxbuf, model_mesh.info);
    error.minimum_magnitude = get_minimum_magnitude(model_vtxbuf, model_mesh.info);

    return error;
}

void
print_error_to_file(const char* path, const int n, const Error error)
{
    FILE* file = fopen(path, "a");
    fprintf(file, "%d, %Lg, %Lg, %Lg, %Lg, %Lg\n", n, error.ulp_error, error.abs_error,
            error.rel_error, error.maximum_magnitude, error.minimum_magnitude);
    // fprintf(file, "%d, %Lg, %Lg, %Lg, %Lg, %Lg\n", n, error.maximum_magnitude,
    // error.minimum_magnitude, error.abs_error, error.ulp_error, error.rel_error);
    fclose(file);
}

#define MAX_PATH_LEN (256)

int
run_autotest(void)
{

#define N_MIN (32)
#define N_MAX (512)
    for (int n = N_MIN; n <= N_MAX; n += N_MIN) {
        AcMeshInfo config;
        load_config(&config);
        config.int_params[AC_nx] = config.int_params[AC_ny] = config.int_params[AC_nz] = n;
        update_config(&config);

        // Init host
        AcMesh* candidate_mesh = acmesh_create(config);
        ModelMesh* model_mesh  = modelmesh_create(config);

        // Init device
        acInit(config);

        // Check all initial conditions
        for (int i = 0; i < ARRAY_SIZE(test_cases); ++i) {
            const InitType init_type = test_cases[i];
            acmesh_init_to((InitType)init_type, candidate_mesh);
            acmesh_to_modelmesh(*candidate_mesh, model_mesh); // Load to Host
            acLoad(*candidate_mesh);                          // Load to Device

            boundconds(model_mesh->info, model_mesh);
            acBoundcondStep();

            { // Check boundconds
                acStore(candidate_mesh);
                Error boundcond_error = get_max_abs_error_mesh(*model_mesh, *candidate_mesh);
                char boundcond_path[MAX_PATH_LEN];
                sprintf(boundcond_path, "%s_boundcond_%s.testresult",
                        AC_DOUBLE_PRECISION ? "double" : "float",
                        init_type_names[(InitType)init_type]);
                print_error_to_file(boundcond_path, n, boundcond_error);
            }

            { // Check scalar max reduction
                ModelScalar model         = model_reduce_scal(*model_mesh, (ReductionType)RTYPE_MAX,
                                                              VTXBUF_UUX);
                AcReal candidate          = acReduceScal((ReductionType)RTYPE_MAX, VTXBUF_UUX);
                Error scalar_reduce_error = get_error(model, candidate);
                char scalar_reduce_path[MAX_PATH_LEN];
                sprintf(scalar_reduce_path, "%s_scalar_reduce_%s.testresult",
                        AC_DOUBLE_PRECISION ? "double" : "float",
                        init_type_names[(InitType)init_type]);
                print_error_to_file(scalar_reduce_path, n, scalar_reduce_error);
            }

            { // Check vector max reduction
                ModelScalar model = model_reduce_vec(*model_mesh, (ReductionType)RTYPE_MAX,
                                                     VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
                AcReal candidate  = acReduceVec((ReductionType)RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY,
                                                VTXBUF_UUZ);
                Error vector_reduce_error = get_error(model, candidate);
                char vector_reduce_path[MAX_PATH_LEN];
                sprintf(vector_reduce_path, "%s_vector_reduce_%s.testresult",
                        AC_DOUBLE_PRECISION ? "double" : "float",
                        init_type_names[(InitType)init_type]);
                print_error_to_file(vector_reduce_path, n, vector_reduce_error);
            }

            // Time advance
            {
                const AcReal umax = (AcReal)model_reduce_vec(*model_mesh, (ReductionType)RTYPE_MAX,
                                                             VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
                const AcReal dt   = host_timestep(umax, config);

#if LFORCING

                // CURRENTLY AUTOTEST NOT SUPPORTED WITH FORCING!!!

#endif

                // Host integration step
                model_rk3(dt, model_mesh);
                boundconds(config, model_mesh);

                // Device integration step
                acIntegrate(dt);
                acBoundcondStep();
                acStore(candidate_mesh);

                // Check fields
                for (int vtxbuf_handle = 0; vtxbuf_handle < NUM_VTXBUF_HANDLES; ++vtxbuf_handle) {
                    Error field_error = get_max_abs_error_vtxbuf((VertexBufferHandle)vtxbuf_handle,
                                                                 *model_mesh, *candidate_mesh);

                    printf("model %Lg, cand %Lg, abs %Lg, rel %Lg\n",
                           (ModelScalar)field_error.model, (ModelScalar)field_error.candidate,
                           (ModelScalar)field_error.abs_error, (ModelScalar)field_error.rel_error);

                    char field_path[MAX_PATH_LEN];
                    sprintf(field_path, "%s_integrationstep_%s_%s.testresult",
                            AC_DOUBLE_PRECISION ? "double" : "float",
                            init_type_names[(InitType)init_type],
                            vtxbuf_names[(VertexBufferHandle)vtxbuf_handle]);
                    print_error_to_file(field_path, n, field_error);
                }
            }
        }

        // Deallocate host
        acmesh_destroy(candidate_mesh);
        modelmesh_destroy(model_mesh);

        // Deallocate device
        acQuit();
    }

    return 0;
}
#endif
