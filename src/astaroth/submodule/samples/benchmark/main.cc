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
    Running: benchmark -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"

#include "errchk.h"
#include "timer_hires.h"

#include <string>
#include <unistd.h> // getopt

static const bool verify = false;

#if AC_MPI_ENABLED

#include <mpi.h>

#include <algorithm>
#include <string.h>
#include <vector>

typedef enum {
    TEST_STRONG_SCALING,
    TEST_WEAK_SCALING,
    NUM_TESTS,
} TestType;

#include <stdint.h>

typedef struct {
    uint64_t x, y, z;
} uint3_64;

static uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
    return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    for (int bit = 0; bit <= 21; ++bit) {
        const uint64_t mask = 0x1l << 3 * bit;
        k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
        j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
        i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
    }

    return (uint3_64){i, j, k};
}

static uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    printf("getting decomp: %d,%d,%d\n", p.x.,p.y,p.z);
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

#include "timer_hires.h"

#include <stdarg.h>

static Timer timer;

static void
timer_event_launch(void)
{
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    acGridSynchronizeStream(STREAM_ALL);
    if (pid == 0) {
        timer_reset(&timer);
    }
}

static void
timer_event_stop(const char* format, ...)
{
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    acGridSynchronizeStream(STREAM_ALL);
    if (pid == 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);

        timer_diff_print(timer);
        fflush(stdout);
    }
}

int
main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    TestType test = TEST_STRONG_SCALING;

    int opt;
    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
        case 't':
            if (std::string("strong").find(optarg) == 0) {
                test = TEST_STRONG_SCALING;
            }
            else if (std::string("weak").find(optarg) == 0) {
                test = TEST_WEAK_SCALING;
            }
            // else if (std::string("verify").find(optarg) == 0) {
            //     verify = true;
            // }
            else {
                fprintf(stderr, "Could not parse option -t <type>. <type> should be \"strong\" or "
                                "\"weak\"\n");
                exit(EXIT_FAILURE);
            }
            break;
        default:
            fprintf(stderr, "Could not parse arguments. Usage: ./benchmark <nx> <ny> <nz>.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (argc - optind > 0) {
        if (argc - optind == 3) {
            const int nx           = atoi(argv[optind]);
            const int ny           = atoi(argv[optind + 1]);
            const int nz           = atoi(argv[optind + 2]);
            info.int_params[AC_nx] = nx;
            info.int_params[AC_ny] = ny;
            info.int_params[AC_nz] = nz;
            acHostUpdateBuiltinParams(&info);
            printf("Benchmark mesh dimensions: (%d, %d, %d)\n", nx, ny, nz);
        }
        else {
            fprintf(stderr, "Could not parse arguments. Usage: ./benchmark <nx> <ny> <nz>.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (test == TEST_WEAK_SCALING) {
        fprintf(stdout, "Running weak scaling benchmarks.\n");
        uint3_64 decomp = decompose(nprocs);
        info.int_params[AC_nx] *= decomp.x;
        info.int_params[AC_ny] *= decomp.y;
        info.int_params[AC_nz] *= decomp.z;
    }
    else {
        fprintf(stdout, "Running strong scaling benchmarks.\n");
    }

    // Device init
    acGridInit(info);
    acGridRandomize();

    // Constant timestep
    const AcReal dt = (AcReal)FLT_EPSILON;

    // Dryrun
    acGridIntegrate(STREAM_DEFAULT, dt);

    if (verify) {
        // Host init
        AcMesh model, candidate;
        if (!pid) {
            acHostMeshCreate(info, &model);
            acHostMeshCreate(info, &candidate);
            acHostMeshRandomize(&model);
            acHostMeshRandomize(&candidate);
        }
        acGridLoadMesh(STREAM_DEFAULT, model);
        acGridPeriodicBoundconds(STREAM_DEFAULT);

        // Verification run
        const size_t nsteps = 10;
        for (size_t i = 0; i < nsteps; ++i) {
            acGridIntegrate(STREAM_DEFAULT, dt);

            if (!pid) {
                printf("Host integration step %lu\n", i);
                fflush(stdout);

                acHostIntegrateStep(model, dt);
            }
        }
        acHostMeshApplyPeriodicBounds(&model);
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        acGridStoreMesh(STREAM_DEFAULT, &candidate);
        acGridSynchronizeStream(STREAM_ALL);

        // Verify
        if (!pid) {
            printf("Verifying...\n");
            fflush(stdout);

            AcResult retval = acVerifyMesh("Integration", model, candidate);
            acHostMeshDestroy(&model);
            acHostMeshDestroy(&candidate);

            if (retval != AC_SUCCESS) {
                fprintf(stderr, "Failures found, benchmark invalid. Skipping\n");
                return EXIT_FAILURE;
            }
            printf("Verification done - everything OK\n");
        }
    }

    // Percentiles
    const size_t num_iters      = 100;
    const double nth_percentile = 0.90;
    std::vector<double> results; // ms
    results.reserve(num_iters);

    // Warmup
    for (size_t i = 0; i < 5; ++i)
        acGridIntegrate(STREAM_DEFAULT, dt);

    // Benchmark
    Timer t;
    for (size_t i = 0; i < num_iters; ++i) {
        acGridSynchronizeStream(STREAM_ALL);
        timer_reset(&t);
        acGridSynchronizeStream(STREAM_ALL);
        acGridIntegrate(STREAM_DEFAULT, dt);
        acGridSynchronizeStream(STREAM_ALL);
        results.push_back(timer_diff_nsec(t) / 1e6); // ms
        acGridSynchronizeStream(STREAM_ALL);
    }

    if (!pid) {
        std::sort(results.begin(), results.end(),
                  [](const double& a, const double& b) { return a < b; });
        fprintf(stdout,
                "Integration step time %g ms (%gth "
                "percentile)--------------------------------------\n",
                results[(size_t)(nth_percentile * num_iters)], 100 * nth_percentile);

        // char path[4096] = "";
        // sprintf(path, "%s_%d.csv", test == TEST_STRONG_SCALING ? "strong" : "weak", nprocs);
        const char path[] = "scaling-benchmark.csv";

        FILE* fp = fopen(path, "a");
        ERRCHK_ALWAYS(fp);
// Format
// nprocs, min, 50th perc, 90th perc, max
// Format
// devices,millisecondsmin,milliseconds50thpercentile,milliseconds90thpercentile,millisecondsmax,usedistributedcommunication,nx,ny,nz,dostrongscaling
// devices, minmilliseconds, 50th perc (ms), 90th perc (ms), max (ms)
#if USE_DISTRIBUTED_IO
        const bool use_distributed_io = true;
#else
        const bool use_distributed_io = false;
#endif
        fprintf(fp, "%d,%g,%g,%g,%g,%d,%d,%d,%d,%d\n", nprocs, results[0],
                results[(size_t)(0.5 * num_iters)], results[(size_t)(nth_percentile * num_iters)],
                results[num_iters - 1], use_distributed_io, info.int_params[AC_nx],
                info.int_params[AC_ny], info.int_params[AC_nz], test == TEST_STRONG_SCALING);
        // fprintf(fp, "%d, %g, %g, %g, %g\n", nprocs, results[0],
        //         results[(size_t)(0.5 * num_iters)],
        //         results[(size_t)(nth_percentile * num_iters)], results[num_iters - 1]);
        fclose(fp);
    }

    // Sanity check start
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!pid)
        fprintf(stderr, "\nSanity performance check:\n");

    timer_event_launch();
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    timer_event_stop("acGridPeriodicBoundconds: ");

    timer_event_launch();
    acGridIntegrate(STREAM_DEFAULT, dt);
    timer_event_stop("acGridIntegrate: ");

    timer_event_launch();
    AcReal candval;
    acGridReduceScal(STREAM_DEFAULT, (ReductionType)0, (Field)0, &candval);
    timer_event_stop("acGridReduceScal");

    ERRCHK_ALWAYS(NUM_FIELDS >= 3);
    timer_event_launch();
    acGridReduceVec(STREAM_DEFAULT, (ReductionType)0, (Field)0, (Field)1, (Field)2, &candval);
    timer_event_stop("acGridReduceVec");
    // Sanity check end

    acGridQuit();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
