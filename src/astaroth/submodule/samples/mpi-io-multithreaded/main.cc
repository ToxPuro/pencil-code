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

    May need to allocate >= 2 cores per task to get tryly parallel compute and disk IO
    SRUN="srun --account=project_2000403 --gres=gpu:v100:2 --mem=24000 -t 00:14:59 -p gputest
   --ntasks-per-socket=1 -n 2 -N 1 --cpus-per-task=2"
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

void
timer_print(const char* str, const Timer t)
{
    const double ms = timer_diff_nsec(t) / 1e6;
    printf("%s: %g ms\n", str, ms);
}

int
main(int argc, char** argv)
{
    //////////////// FUNNELED
    /*
    int thread_support_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &thread_support_level);
    if (thread_support_level < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED not supported by the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    */
    ////////////////////

    //////////////////// MULTIPLE
    int thread_support_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_support_level);
    if (thread_support_level < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not supported by the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    ////////////////////
    // MPI_Init(NULL, NULL);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    // Set mesh dimensions
    if (argc != 4) {
        fprintf(stderr, "Usage: ./mpi-io <nx> <ny> <nz>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = atoi(argv[3]);
        acHostUpdateBuiltinParams(&info);
    }

    AcMesh model, candidate;
    if (!pid) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);

        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    // Test load/store and boundconds
    if (!pid) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("CPU-GPU Load/store", model, candidate);
        WARNCHK_ALWAYS(res == AC_SUCCESS);
    }

    // Test integration step
    const AcReal dt = (AcReal)FLT_EPSILON;

    // Warmup
    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridLoadMesh(STREAM_DEFAULT,
                   model); // Workaround to avoid cluttering the buffers with autotuning
    acGridSwapBuffers();
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (!pid) {
        acHostIntegrateStep(model, dt);
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration step", model, candidate);
        WARNCHK_ALWAYS(res == AC_SUCCESS);
    }

    // Declare timer
    Timer t;

#if 0
    // Test synchronous read/write
    //// Write
    timer_reset(&t);
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char buf[4096] = "";
        sprintf(buf, "%s.out", vtxbuf_names[i]);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, buf, ACCESS_WRITE);
    }
    timer_print("Wrote mesh to disc (synchronous", t);
    //// Scramble buffers
    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)0, "test.out", ACCESS_WRITE);
    acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)0, "test.out", ACCESS_READ);
    //// Read
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char buf[4096] = "";
        sprintf(buf, "%s.out", vtxbuf_names[i]);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, buf, ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (!pid) {
        const AcResult res = acVerifyMesh("Synchronous read/write", model, candidate);
        WARNCHK_ALWAYS(res == AC_SUCCESS);
    }
#endif

    // Test asynchronous read/write
    //// Write
    timer_reset(&t);
    if (!pid)
        timer_print("Timer reset", t);
    acGridDiskAccessLaunch(ACCESS_WRITE);
    if (!pid)
        timer_print("Disk access launched", t);

    //// Scramble buffers
    for (size_t i = 0; i < 10; ++i) {
        acGridIntegrate(STREAM_DEFAULT, dt);
        if (!pid)
            timer_print("Integration step complete", t);
    }
    acGridDiskAccessSync();
    if (!pid)
        timer_print("Disk access synced", t);

    //// Read
    // acGridDiskAccessLaunch(ACCESS_READ);
    // acGridDiskAccessSync();
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char buf[4096] = "";
        sprintf(buf, "%s.out", vtxbuf_names[i]);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, ".", buf, ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (!pid) {
        const AcResult res = acVerifyMesh("Asynchronous read/write", model, candidate);
        WARNCHK_ALWAYS(res == AC_SUCCESS);
    }

    if (!pid) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }
    acGridQuit();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#endif
