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
    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"
#include "timer_hires.h"

#if !AC_MPI_ENABLED
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#else

#include <mpi.h>

static const bool verify = false;

int
main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int job_id = 0;

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    // Set mesh dimensions
    if (argc < 4 || argc > 5) {
        fprintf(stderr,
                "Usage: ./mpi-io <nx> <ny> <nz> <(Optional) unique job id for output files>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = atoi(argv[3]);
        acHostUpdateBuiltinParams(&info);

        if (argc == 5)
            job_id = atoi(argv[4]);
    }
    char job_dir[4096];
    snprintf(job_dir, 4096, "mpi-io-tmpdir-%d", job_id);

    // Init device
    acGridInit(info);

    AcMesh model, candidate;
    if (verify) {
        // Alloc host
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
        acHostMeshApplyPeriodicBounds(&model);

        acGridLoadMesh(STREAM_DEFAULT, model);
        acGridPeriodicBoundconds(STREAM_DEFAULT);

        acGridStoreMesh(STREAM_DEFAULT, &candidate);
        if (!pid) {
            const AcResult res = acVerifyMesh("CPU-GPU Load/store", model, candidate);
            ERRCHK_ALWAYS(res == AC_SUCCESS);
        }
    }

    // Make tmpdir for output
    char cmd[4096];
    snprintf(cmd, 4096, "mkdir -p %s", job_dir);
    system(cmd);

    // Write
    Timer t;
    timer_reset(&t);
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char label[4096] = "";
        sprintf(label, "field-%lu", i);
        // printf("Storing %s/%s\n", job_dir, label);
        // acGridStoreFieldToFile(label, (VertexBufferHandle)i);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, job_dir, label, ACCESS_WRITE);
    }
    double write_milliseconds = 0;
    double write_bandwidth    = 0; // bytes per second
    if (!pid) {
        write_milliseconds   = (double)timer_diff_nsec(t) / 1e6;
        const double seconds = (double)timer_diff_nsec(t) / 1e9;
        const size_t bytes   = NUM_VTXBUF_HANDLES * acVertexBufferCompdomainSizeBytes(info);
        write_bandwidth      = bytes / seconds;
        timer_diff_print(t);
    }

    if (verify) {
        // Scramble
        acHostMeshRandomize(&candidate);
        acGridLoadMesh(STREAM_DEFAULT, candidate);
        // acGridStoreFieldToFile("field-tmp.out", 0);
        for (size_t i = 0; i < NUM_FIELDS; ++i)
            acGridAccessMeshOnDiskSynchronous(i, job_dir, "field-tmp",
                                              ACCESS_WRITE); // Hacky, indirectly scramble vba.out
                                                             // to catch false positives if the MPI
                                                             // calls fail completely.
    }

    // Read
    timer_reset(&t);
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char label[4096] = "";
        sprintf(label, "field-%lu", i);
        // acGridLoadFieldFromFile(label, (VertexBufferHandle)i);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, job_dir, label, ACCESS_READ);
    }
    double read_milliseconds = 0;
    double read_bandwidth    = 0; // bytes per second
    if (!pid) {
        read_milliseconds    = (double)timer_diff_nsec(t) / 1e6;
        const double seconds = (double)timer_diff_nsec(t) / 1e9;
        const size_t bytes   = NUM_VTXBUF_HANDLES * acVertexBufferCompdomainSizeBytes(info);
        read_bandwidth       = bytes / seconds;
        timer_diff_print(t);
    }

    if (verify) {
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        acGridStoreMesh(STREAM_DEFAULT, &candidate);

        if (!pid) {
            const AcResult res = acVerifyMesh("MPI-IO disk read/write", model, candidate);
            ERRCHK_ALWAYS(res == AC_SUCCESS);
        }
    }

    // Write out
    // Format:
    // devices,writemilliseconds,writebandwidth,readmilliseconds,readbandwidth,usedistributedio,nx,ny,nz
    if (!pid) {
        FILE* fp = fopen("scaling-io-benchmark.csv", "a");
        ERRCHK_ALWAYS(fp);

#if USE_DISTRIBUTED_IO
        const bool use_distributed_io = true;
#else
        const bool use_distributed_io = false;
#endif
        fprintf(fp, "%d,%g,%g,%g,%g,%d,%d,%d,%d\n", nprocs, write_milliseconds, write_bandwidth,
                read_milliseconds, read_bandwidth, use_distributed_io, info.int_params[AC_nx],
                info.int_params[AC_ny], info.int_params[AC_nz]);
        fclose(fp);
    }

    acGridQuit();

    if (verify) {
        // Deallocate
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    // Remove old files
    if (!pid) {
        printf("Removing fields\n");
        // sprintf(cmd, "rm %s/*.mesh", job_dir);
        sprintf(cmd, "rm -r %s", job_dir);
        system(cmd);
        printf("Done.\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
#endif
