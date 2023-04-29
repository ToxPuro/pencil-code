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

#include <stdlib.h> // EXIT_SUCCESS

#include "config_loader.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"
#include "timer_hires.h"

#include "errchk.h"
#include <algorithm>
#include <math.h>
#include <vector>

static bool
smaller_than(const double& a, const double& b)
{
    return a < b;
}

int
run_benchmark(const char* config_path)
{
    const int nn        = 512;
    const int num_iters = 100;
#define BENCH_STRONG_SCALING (1)
    const int num_processes = acGetNumDevicesPerNode();

    AcMeshInfo mesh_info;
    load_config(config_path, &mesh_info);
    mesh_info.int_params[AC_nx] = mesh_info.int_params[AC_ny] = nn;
    mesh_info.int_params[AC_nz] = BENCH_STRONG_SCALING ? nn : nn * num_processes;
    update_config(&mesh_info);

    AcMesh* mesh = acmesh_create(mesh_info);
    acmesh_init_to(INIT_TYPE_ABC_FLOW, mesh);

    acInit(mesh_info);
    acLoad(*mesh);

    std::vector<double> results;
    results.reserve(num_iters);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        acIntegrate(0);
    }
    acSynchronize();
    const AcReal dt = FLT_EPSILON;
    printf("Using dt = %g\n", (double)dt);

    Timer total_time;
    timer_reset(&total_time);

    Timer step_time;
    for (int i = 0; i < num_iters; ++i) {
        timer_reset(&step_time);

        acIntegrate(dt);
        acSynchronize();

        results.push_back(timer_diff_nsec(step_time) / 1e6);
    }
    acSynchronize();
    const double ms_elapsed     = timer_diff_nsec(total_time) / 1e6;
    const double nth_percentile = 0.90;
    std::sort(results.begin(), results.end(), smaller_than);

    printf("vertices: %d^3, iterations: %d\n", nn, num_iters);
    printf("Total time: %f ms\n", ms_elapsed);
    printf("Time per step: %f ms\n", ms_elapsed / num_iters);

    const size_t nth_index = int(nth_percentile * num_iters);
    printf("%dth percentile per step: %f ms\n", int(100 * nth_percentile), results[nth_index]);

    // Write out
    char buf[256];
    sprintf(buf, "nprocs_%d_result_%s.bench", num_processes,
            BENCH_STRONG_SCALING ? "strong" : "weak");
    FILE* fp = fopen(buf, "w");
    ERRCHK_ALWAYS(fp);
    fprintf(fp, "num_processes, percentile (%dth)\n", int(100 * nth_percentile));
    fprintf(fp, "%d, %g\n", num_processes, results[nth_index]);
    fclose(fp);

    acQuit();
    acmesh_destroy(mesh);

    return AC_SUCCESS;
}

#if 0 // Old single-GPU benchmark
static bool
smaller_than(const double& a, const double& b)
{
    return a < b;
}

static int
write_runningtimes(const char* path, const int n, const double min, const double max,
                   const double median, const double perc)
{
    FILE* fp;
    fp = fopen(path, "a");

    if (fp != NULL) {
        fprintf(fp, "%d, %f, %f, %f, %f\n", n, min, max, median, perc);
        fclose(fp);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}

static int
write_percentiles(const char* path, const int num_iters, const std::vector<double>& results)
{
    FILE* fp;
    fp = fopen(path, "w");

    if (fp != NULL) {
        for (int i = 0; i < 100; ++i) {
            fprintf(fp, "%f\n", results[(long unsigned)((i / 100.) * num_iters)]);
        }
        fclose(fp);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}

int
run_benchmark(void)
{
    char runningtime_path[256];
    sprintf(runningtime_path, "%s_%s_runningtimes.out", AC_DOUBLE_PRECISION ? "double" : "float",
            GEN_BENCHMARK_RK3 ? "rk3substep" : "fullstep");

    FILE* fp;
    fp = fopen(runningtime_path, "w");

    if (fp != NULL) {
        fprintf(fp, "n, min, max, median, perc\n");
        fclose(fp);
    }
    else {
        return EXIT_FAILURE;
    }

#define N_STEP_SIZE (128)
#define MAX_MESH_DIM (128)
#define NUM_ITERS (100)
    for (int n = N_STEP_SIZE; n <= MAX_MESH_DIM; n += N_STEP_SIZE) {
        /* Parse configs */
        AcMeshInfo mesh_info;
        load_config(&mesh_info);
        mesh_info.int_params[AC_nx] = n;
        mesh_info.int_params[AC_ny] = mesh_info.int_params[AC_nx];
        mesh_info.int_params[AC_nz] = mesh_info.int_params[AC_nx];
        update_config(&mesh_info);

        AcMesh* mesh = acmesh_create(mesh_info);
        acmesh_init_to(INIT_TYPE_ABC_FLOW, mesh);

        acInit(mesh_info);
        acLoad(*mesh);

        std::vector<double> results;
        results.reserve(NUM_ITERS);

        // Optimize
        // acAutoOptimize();

        // Warmup
        for (int i = 0; i < 10; ++i) {
            acIntegrate(0);
        }

        Timer t;
        for (int i = 0; i < NUM_ITERS; ++i) {

            timer_reset(&t);
            const AcReal dt = FLT_EPSILON; // TODO NOTE: time to timestep not measured
#if GEN_BENCHMARK_RK3 == 1
            acIntegrateStep(2, dt);
            acSynchronizeStream(STREAM_ALL);
#else // GEN_BENCHMARK_FULL
            acIntegrate(dt);
#endif
            const double ms_elapsed = timer_diff_nsec(t) / 1e6;
            results.push_back(ms_elapsed);
        }

#define NTH_PERCENTILE (0.95)
        std::sort(results.begin(), results.end(), smaller_than);
        write_runningtimes(runningtime_path, n, results[0], results[results.size() - 1],
                           results[int(0.5 * NUM_ITERS)], results[int(NTH_PERCENTILE * NUM_ITERS)]);

        char percentile_path[256];
        sprintf(percentile_path, "%d_%s_%s_percentiles.out", n,
                AC_DOUBLE_PRECISION ? "double" : "float",
                GEN_BENCHMARK_RK3 ? "rk3substep" : "fullstep");
        write_percentiles(percentile_path, NUM_ITERS, results);

        printf("%s running time %g ms, (%dth percentile, nx = %d) \n",
               GEN_BENCHMARK_RK3 ? "RK3 step" : "Fullstep",
               double(results[int(NTH_PERCENTILE * NUM_ITERS)]), int(NTH_PERCENTILE * 100),
               mesh_info.int_params[AC_nx]);

        acQuit();
        acmesh_destroy(mesh);
    }

    return 0;
}
#endif // single-GPU benchmark

/*

#if AUTO_OPTIMIZE
const char* benchmark_path = "benchmark.out";

#include "kernels/rk3_threadblock.conf"
static int
write_result_to_file(const float& ms_per_step)
{
    FILE* fp;
    fp = fopen(benchmark_path, "a");

    if (fp != NULL) {
        fprintf(fp,
                "(%d, %d, %d), %d elems per thread, launch bound %d, %f ms\n",
                RK_THREADS_X, RK_THREADS_Y, RK_THREADS_Z, RK_ELEMS_PER_THREAD,
                RK_LAUNCH_BOUND_MIN_BLOCKS, double(ms_per_step));
        fclose(fp);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
#endif

#if GENERATE_BENCHMARK_DATA != 1
int
run_benchmark(void)
{
    // Parse configs
    AcMeshInfo mesh_info;
    load_config(&mesh_info);
    mesh_info.int_params[AC_nx] = 128;
    mesh_info.int_params[AC_ny] = mesh_info.int_params[AC_nx];
    mesh_info.int_params[AC_nz] = mesh_info.int_params[AC_nx];
    update_config(&mesh_info);

    AcMesh* mesh = acmesh_create(mesh_info);
    acmesh_init_to(INIT_TYPE_ABC_FLOW, mesh);

    acInit(mesh_info);
    acLoad(*mesh);

    Timer t;
    timer_reset(&t);

    int steps           = 0;
    const int num_steps = 100;
    while (steps < num_steps) {
        // Advance the simulation
        const AcReal umax = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY,
                                        VTXBUF_UUZ);
        const AcReal dt   = host_timestep(umax, mesh_info);
        acIntegrate(dt);
        ++steps;
    }
    acSynchronize();
    const float wallclock = timer_diff_nsec(t) / 1e9f;
    printf("%d steps. Wallclock time %f s per step\n", steps,
           double(wallclock) / num_steps);
    #if AUTO_OPTIMIZE
    write_result_to_file(wallclock * 1e3f / steps);
    #endif

    acQuit();
    acmesh_destroy(mesh);

    return 0;
}

#else
//////////////////////////////////////////////////////////////////////////GENERATE_BENCHMARK_DATA




int
run_benchmark(void)
{
    const char path[] = "result.out";
    FILE* fp;
    fp = fopen(path, "w");

    if (fp != NULL) {
        fprintf(fp, "n, min, max, median, perc\n");
        fclose(fp);
    } else {
        return EXIT_FAILURE;
    }

    #define N_STEP_SIZE (256)
    #define MAX_MESH_DIM (256)
    #define NUM_ITERS (1000)
    for (int n = N_STEP_SIZE; n <= MAX_MESH_DIM; n += N_STEP_SIZE) {
        // Parse configs
        AcMeshInfo mesh_info;
        load_config(&mesh_info);
        mesh_info.int_params[AC_nx] = n;
        mesh_info.int_params[AC_ny] = mesh_info.int_params[AC_nx];
        mesh_info.int_params[AC_nz] = mesh_info.int_params[AC_nx];
        update_config(&mesh_info);

        AcMesh* mesh = acmesh_create(mesh_info);
        acmesh_init_to(INIT_TYPE_ABC_FLOW, mesh);

        acInit(mesh_info);
        acLoad(*mesh);

        std::vector<double> results;
        results.reserve(NUM_ITERS);


        // Warmup
        for (int i = 0; i < 10; ++i) {
            acIntegrate(0);
            acSynchronize();
        }

        Timer t;

        const AcReal dt = AcReal(1e-5);
        for (int i = 0; i < NUM_ITERS; ++i) {

            timer_reset(&t);
            //acIntegrate(dt);
            acIntegrateStep(2, dt);
            acSynchronize();

            const double ms_elapsed = timer_diff_nsec(t) / 1e6;
            results.push_back(ms_elapsed);
        }



        #define NTH_PERCENTILE (0.95)
        std::sort(results.begin(), results.end(), smaller_than);
        write_result(n, results[0], results[results.size()-1], results[int(0.5 * NUM_ITERS)],
results[int(NTH_PERCENTILE * NUM_ITERS)]); write_percentiles(n, NUM_ITERS, results);
    }

    return 0;
}
#endif
*/
