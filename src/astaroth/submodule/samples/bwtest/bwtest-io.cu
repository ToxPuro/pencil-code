#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep

#include <thread>

#include <omp.h>

#include "common.h"

#include "timer_hires.h" // From acc-runtime/api

#if !AC_MPI_ENABLED
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#else
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

#include <mpi.h>

int
main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: ./benchmark-io <compute time in ms (integer)> <communication bytes "
                        "(integer)>\n");
        fprintf(stderr, "       ./benchmark 0 0 # To use the defaults\n");
        return EXIT_FAILURE;
    }
    const size_t arg0 = (size_t)atol(argv[1]);
    const size_t arg1 = (size_t)atol(argv[2]);

    const size_t compute_time = arg0 ? arg0 : 25;        // 25 ms by default
    const size_t comm_size    = arg1 ? arg1 : 268435456; // 256 MiB default

    MPI_Init(NULL, NULL);

    // MPI info
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    printf("Hello from proc %d of %d\n", pid, nprocs);

    // Allocate
    uint8_t* data = (uint8_t*)malloc(comm_size);
    for (size_t i = 0; i < comm_size; ++i)
        data[i] = (uint8_t)rand();
    assert(data);

    /*
    const auto fn = [](const int i){ usleep(2 * 1e6); printf("tid %d\n", i); };
    std::thread t1(fn, 0);
    printf("This happens async\n");
    t1.join();
    MPI_Finalize();
    return EXIT_SUCCESS;
    */

    // Synchronous IO
    printf("Synchronous IO\n");
    MPI_File file;
    int retval;
    const size_t buflen = 4096;
    char path[buflen];
    snprintf(path, buflen, "proc%d.out", pid);
    printf("Writing out to %s\n", path);
    int mode = MPI_MODE_CREATE // Create if not already exists
               | MPI_MODE_WRONLY;

    retval = MPI_File_open(MPI_COMM_SELF, path, mode, MPI_INFO_NULL, &file);
    assert(retval == MPI_SUCCESS);

    Timer t;
    timer_reset(&t);
    MPI_Status status;
    retval = MPI_File_write(file, data, comm_size, MPI_UINT8_T, &status);
    assert(retval == MPI_SUCCESS);
    timer_diff_print(t);

    retval = MPI_File_close(&file);
    assert(retval == MPI_SUCCESS);

    // Asynchronous IO
    printf("Asynchronous IO\n");
    retval = MPI_File_open(MPI_COMM_SELF, path, mode, MPI_INFO_NULL, &file);
    assert(retval == MPI_SUCCESS);

    timer_reset(&t);
    MPI_Request req;
    retval = MPI_File_iwrite(file, data, comm_size, MPI_UINT8_T, &req);
    assert(retval == MPI_SUCCESS);
    printf("Returned from MPI_File_iwrite\n");
    timer_diff_print(t);

    int complete;
    retval = MPI_Request_get_status(req, &complete, &status);
    assert(retval == MPI_SUCCESS);
    while (!complete) {
        printf("Process %d not yet complete...\n", pid);
        timer_diff_print(t);
        fflush(stdout);
        usleep(compute_time * 1000);

        retval = MPI_Request_get_status(req, &complete, &status);
        assert(retval == MPI_SUCCESS);
    }
    printf("Process %d complete\n", pid);
    timer_diff_print(t);
    retval = MPI_Wait(&req, &status);
    assert(retval == MPI_SUCCESS);
    printf("Wait complete\n");
    timer_diff_print(t);

    retval = MPI_File_close(&file);
    assert(retval == MPI_SUCCESS);

// Does not always (ever?) allocate two threads in this case
#pragma omp parallel num_threads(2)
    {
        const int tid = omp_get_thread_num();
        printf("Hello from proc %d, tid %d\n", pid, 1 * tid);
    }

    // Synchronous IO with C++ threads
    printf("Synchronous IO with C++ threads\n");
    retval = MPI_File_open(MPI_COMM_SELF, path, mode, MPI_INFO_NULL, &file);
    assert(retval == MPI_SUCCESS);

    const auto write = [](const MPI_File file, const uint8_t* data, const size_t bytes) {
        MPI_Status status;
        const int retval = MPI_File_write(file, data, bytes, MPI_UINT8_T, &status);
        assert(retval == MPI_SUCCESS);
    };

    timer_reset(&t);
    std::thread write_thread(write, file, data, comm_size);
    printf("C++ thread started\n");
    timer_diff_print(t);

    for (int i = 0; i < 10; ++i) {
        printf("Doing something else in the meanwhile\n");
        timer_diff_print(t);
        fflush(stdout);
        usleep(compute_time * 1000);
    }

    write_thread.join();
    printf("C++ thread joined\n");
    timer_diff_print(t);

    retval = MPI_File_close(&file);
    assert(retval == MPI_SUCCESS);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
#endif
