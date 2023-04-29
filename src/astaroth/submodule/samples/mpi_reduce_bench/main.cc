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

#if AC_MPI_ENABLED

#include <mpi.h>

#include <algorithm>
#include <string.h>

#include <chrono>
#include <ctime>
#include <string>
#include <vector>

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

    char* benchmark_label;

    if (argc > 1) {
        benchmark_label = argv[1];
    }
    else {
        auto timestamp  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        benchmark_label = std::ctime(&timestamp);
        benchmark_label[strcspn(benchmark_label, "\n")] = '\0';
    }

    //clang-format off
    std::vector<AcScalReductionTestCase>
        scalarReductionTests{acCreateScalReductionTestCase("Scalar MAX", VTXBUF_UUX, RTYPE_MAX),
                             acCreateScalReductionTestCase("Scalar MIN", VTXBUF_UUX, RTYPE_MIN),
                             acCreateScalReductionTestCase("Scalar RMS", VTXBUF_UUX, RTYPE_RMS),
                             acCreateScalReductionTestCase("Scalar RMS_EXP", VTXBUF_UUX,
                                                           RTYPE_RMS_EXP),
                             acCreateScalReductionTestCase("Scalar SUM", VTXBUF_UUX, RTYPE_SUM)};

    std::vector<AcVecReductionTestCase>
        vectorReductionTests{acCreateVecReductionTestCase("Vector MAX", VTXBUF_UUX, VTXBUF_UUY,
                                                          VTXBUF_UUZ, RTYPE_MAX),
                             acCreateVecReductionTestCase("Vector MIN", VTXBUF_UUX, VTXBUF_UUY,
                                                          VTXBUF_UUZ, RTYPE_MIN),
                             acCreateVecReductionTestCase("Vector RMS", VTXBUF_UUX, VTXBUF_UUY,
                                                          VTXBUF_UUZ, RTYPE_RMS),
                             acCreateVecReductionTestCase("Vector RMS_EXP", VTXBUF_UUX, VTXBUF_UUY,
                                                          VTXBUF_UUZ, RTYPE_RMS_EXP),
                             acCreateVecReductionTestCase("Vector SUM", VTXBUF_UUX, VTXBUF_UUY,
                                                          VTXBUF_UUZ, RTYPE_SUM)};
    //clang-format on

    // GPU alloc & compute
    acGridInit(info);

    const size_t num_iters      = 100;
    const double nth_percentile = 0.90;
    std::vector<double> results; // ms
    Timer t;
    // Scalar benchmarks
    for (auto& testCase : scalarReductionTests) {
        results.clear();
        results.reserve(num_iters);

        for (size_t i = 0; i < num_iters; ++i) {
            acGridSynchronizeStream(STREAM_ALL);
            timer_reset(&t);
            acGridSynchronizeStream(STREAM_ALL);
            acGridReduceScal(STREAM_DEFAULT, testCase.rtype, testCase.vtxbuf, &testCase.candidate);
            acGridSynchronizeStream(STREAM_ALL);
            results.push_back(timer_diff_nsec(t) / 1e6);
            acGridSynchronizeStream(STREAM_ALL);
        }

        if (!pid) {
            std::sort(results.begin(), results.end(),
                      [](const double& a, const double& b) { return a < b; });
            fprintf(stdout,
                    "Reduction time %g ms (%gth "
                    "percentile)--------------------------------------\n",
                    results[nth_percentile * num_iters], 100 * nth_percentile);

            char path[4096] = "mpi_reduction_benchmark.csv";

            FILE* fp = fopen(path, "a");
            ERRCHK_ALWAYS(fp);

            // Format
            // benchmark label, test label, nprocs, measured (ms)
            fprintf(fp, "\"%s\",\"%s\", %d, %g\n", benchmark_label, testCase.label, nprocs,
                    results[nth_percentile * num_iters]);
            fclose(fp);
        }
    }

    // Vector benchmarks
    for (auto& testCase : vectorReductionTests) {
        results.clear();
        results.reserve(num_iters);

        for (size_t i = 0; i < num_iters; ++i) {
            acGridSynchronizeStream(STREAM_ALL);
            timer_reset(&t);
            acGridSynchronizeStream(STREAM_ALL);
            acGridReduceVec(STREAM_DEFAULT, testCase.rtype, testCase.a, testCase.b, testCase.c,
                            &testCase.candidate);
            acGridSynchronizeStream(STREAM_ALL);
            results.push_back(timer_diff_nsec(t) / 1e6);
            acGridSynchronizeStream(STREAM_ALL);
        }

        if (!pid) {
            std::sort(results.begin(), results.end(),
                      [](const double& a, const double& b) { return a < b; });
            fprintf(stdout,
                    "Reduction time %g ms (%gth "
                    "percentile)--------------------------------------\n",
                    results[nth_percentile * num_iters], 100 * nth_percentile);

            char path[4096] = "mpi_reduction_benchmark.csv";

            FILE* fp = fopen(path, "a");
            ERRCHK_ALWAYS(fp);

            // Format
            // benchmark label, test label, nprocs, measured (ms)
            fprintf(fp, "\"%s\",\"%s\", %d, %g\n", benchmark_label, testCase.label, nprocs,
                    results[nth_percentile * num_iters]);
            fclose(fp);
        }
    }
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
