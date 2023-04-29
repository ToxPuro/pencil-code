#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <cuda.h> // CUDA driver API
#include <cuda_runtime_api.h>

#include "timer_hires.h" // From src/common

//#define BLOCK_SIZE (100 * 1024 * 1024) // Bytes
#define BLOCK_SIZE (256 * 256 * 3 * 8 * 8)

#define errchk(x)                                                                                  \
    {                                                                                              \
        if (!(x)) {                                                                                \
            fprintf(stderr, "errchk(%s) failed", #x);                                              \
            assert(x);                                                                             \
        }                                                                                          \
    }

/*
  Findings:
    - MUST ALWAYS SET DEVICE. Absolutely kills performance if device is not set explicitly
    - Need to use cudaMalloc for intranode comm for P2P to trigger with MPI
    - For internode one should use pinned memory (RDMA is staged through pinned, gives full
    network speed iff pinned)
    - Both the sending and receiving arrays must be pinned to see performance improvement
    in internode comm
*/

static uint8_t*
allocHost(const size_t bytes)
{
    uint8_t* arr = malloc(bytes);
    errchk(arr);
    return arr;
}

static void
freeHost(uint8_t* arr)
{
    free(arr);
}

static uint8_t*
allocDevice(const size_t bytes)
{
    uint8_t* arr;
    // Standard (20 GiB/s internode, 85 GiB/s intranode)
    const cudaError_t retval = cudaMalloc((void**)&arr, bytes);
    // Unified mem (5 GiB/s internode, 6 GiB/s intranode)
    // const cudaError_t retval = cudaMallocManaged((void**)&arr, bytes, cudaMemAttachGlobal);
    // Pinned (40 GiB/s internode, 10 GiB/s intranode)
    // const cudaError_t retval = cudaMallocHost((void**)&arr, bytes);
    errchk(retval == cudaSuccess);
    return arr;
}

static uint8_t*
allocDevicePinned(const size_t bytes)
{
#define USE_CUDA_DRIVER_PINNING (1)
#if USE_CUDA_DRIVER_PINNING
    uint8_t* arr = allocDevice(bytes);

    unsigned int flag = 1;
    CUresult retval   = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                              (CUdeviceptr)arr);

    errchk(retval == CUDA_SUCCESS);
    return arr;

#else
    uint8_t* arr;
    // Standard (20 GiB/s internode, 85 GiB/s intranode)
    // const cudaError_t retval = cudaMalloc((void**)&arr, bytes);
    // Unified mem (5 GiB/s internode, 6 GiB/s intranode)
    // const cudaError_t retval = cudaMallocManaged((void**)&arr, bytes, cudaMemAttachGlobal);
    // Pinned (40 GiB/s internode, 10 GiB/s intranode)
    const cudaError_t retval = cudaMallocHost((void**)&arr, bytes);
    errchk(retval == cudaSuccess);
    return arr;
#endif
}

/*
static uint8_t*
allocDevicePinned(const size_t bytes)
{
    uint8_t* arr;
    // Standard (20 GiB/s internode, 85 GiB/s intranode)
    // const cudaError_t retval = cudaMalloc((void**)&arr, bytes);
    // Unified mem (5 GiB/s internode, 6 GiB/s intranode)
    // const cudaError_t retval = cudaMallocManaged((void**)&arr, bytes, cudaMemAttachGlobal);
    // Pinned (40 GiB/s internode, 10 GiB/s intranode)
    const cudaError_t retval = cudaMallocHost((void**)&arr, bytes);
    errchk(retval == cudaSuccess);
    return arr;
}*/

static void
freeDevice(uint8_t* arr)
{
    cudaFree(arr);
}

static void
sendrecv_blocking(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    if (!pid) {
        MPI_Status status;
        MPI_Send(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD);
        MPI_Recv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &status);
    }
    else {
        MPI_Status status;
        MPI_Recv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &status);
        MPI_Send(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD);
    }
}

static void
sendrecv_nonblocking(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    MPI_Request recv_request, send_request;
    MPI_Irecv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &recv_request);
    MPI_Isend(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD, &send_request);

    MPI_Status status;
    MPI_Wait(&recv_request, &status);
    MPI_Wait(&send_request, &status);
}

static void
sendrecv_twoway(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    MPI_Status status;
    MPI_Sendrecv(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, dst, BLOCK_SIZE, MPI_BYTE, nback, nback,
                 MPI_COMM_WORLD, &status);
}

static void
sendrecv_nonblocking_multiple(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Request recv_requests[nprocs], send_requests[nprocs];
    for (int i = 1; i < nprocs; ++i) {
        int nfront = (pid + i) % nprocs;
        int nback  = (((pid - i) % nprocs) + nprocs) % nprocs;

        MPI_Irecv(dst, BLOCK_SIZE, MPI_BYTE, nback, pid, MPI_COMM_WORLD, &recv_requests[i]);
        MPI_Isend(src, BLOCK_SIZE, MPI_BYTE, nfront, nfront, MPI_COMM_WORLD, &send_requests[i]);
    }

    for (int i = 1; i < nprocs; ++i) {
        MPI_Status status;
        MPI_Wait(&recv_requests[i], &status);
        MPI_Wait(&send_requests[i], &status);
    }
}

/*
static void
sendrecv_nonblocking_multiple_parallel(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Request send_requests[nprocs];
    for (int i = 1; i < nprocs; ++i) {
        int nfront = (pid + i) % nprocs;
        MPI_Isend(src, BLOCK_SIZE, MPI_BYTE, nfront, nfront, MPI_COMM_WORLD, &send_requests[i]);
    }

    static bool error_shown = false;
    if (!pid && !error_shown) {
        fprintf(stderr, "\tWARNING: make sure you init MPI_Init_thread for OpenMP support (no "
                        "supported on puhti atm "
                        "2020-04-05\n");
        error_shown = true;
    }
#pragma omp parallel for
    for (int i = 1; i < nprocs; ++i) {
        int nback = (((pid - i) % nprocs) + nprocs) % nprocs;

        MPI_Status status;
        MPI_Recv(dst, BLOCK_SIZE, MPI_BYTE, nback, pid, MPI_COMM_WORLD, &status);
    }

    for (int i = 1; i < nprocs; ++i) {
        MPI_Status status;
        MPI_Wait(&send_requests[i], &status);
    }
}
*/

static void
sendrecv_nonblocking_multiple_rt_pinning(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    static uint8_t* src_pinned = NULL;
    static uint8_t* dst_pinned = NULL;
    if (!src_pinned)
        src_pinned = allocDevicePinned(BLOCK_SIZE); // Note: Never freed
    if (!dst_pinned)
        dst_pinned = allocDevicePinned(BLOCK_SIZE); // Note: Never freed

    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    MPI_Request recv_requests[nprocs], send_requests[nprocs];
    for (int i = 1; i < nprocs; ++i) {
        int nfront = (pid + i) % nprocs;
        int nback  = (((pid - i) % nprocs) + nprocs) % nprocs;

        if (nback / devices_per_node != pid / devices_per_node) // Not on the same node
            MPI_Irecv(dst_pinned, BLOCK_SIZE, MPI_BYTE, nback, pid, MPI_COMM_WORLD,
                      &recv_requests[i]);
        else
            MPI_Irecv(dst, BLOCK_SIZE, MPI_BYTE, nback, pid, MPI_COMM_WORLD, &recv_requests[i]);

        if (nfront / devices_per_node != pid / devices_per_node) // Not on the same node
            MPI_Isend(src_pinned, BLOCK_SIZE, MPI_BYTE, nfront, nfront, MPI_COMM_WORLD,
                      &send_requests[i]);
        else
            MPI_Isend(src, BLOCK_SIZE, MPI_BYTE, nfront, nfront, MPI_COMM_WORLD, &send_requests[i]);
    }

    for (int i = 1; i < nprocs; ++i) {
        MPI_Status status;
        MPI_Wait(&recv_requests[i], &status);
        MPI_Wait(&send_requests[i], &status);
    }
}

static void
send_d2h(uint8_t* src, uint8_t* dst)
{
    cudaMemcpy(dst, src, BLOCK_SIZE, cudaMemcpyDeviceToHost);
}

static void
send_h2d(uint8_t* src, uint8_t* dst)
{
    cudaMemcpy(dst, src, BLOCK_SIZE, cudaMemcpyHostToDevice);
}

static void
sendrecv_d2h2d(uint8_t* dsrc, uint8_t* hdst, uint8_t* hsrc, uint8_t* ddst)
{
    cudaStream_t d2h, h2d;
    cudaStreamCreate(&d2h);
    cudaStreamCreate(&h2d);

    cudaMemcpyAsync(hdst, dsrc, BLOCK_SIZE, cudaMemcpyDeviceToHost, d2h);
    cudaMemcpyAsync(ddst, hsrc, BLOCK_SIZE, cudaMemcpyHostToDevice, h2d);

    cudaStreamSynchronize(d2h);
    cudaStreamSynchronize(h2d);

    cudaStreamDestroy(d2h);
    cudaStreamDestroy(h2d);
}

#define PRINT                                                                                      \
    if (!pid)                                                                                      \
    printf

static void
measurebw(const char* msg, const size_t bytes, void (*sendrecv)(uint8_t*, uint8_t*), uint8_t* src,
          uint8_t* dst)
{
    const size_t num_samples = 100;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    PRINT("%s\n", msg);
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT("\tWarming up... ");
    for (size_t i = 0; i < num_samples / 10; ++i)
        sendrecv(src, dst);

    MPI_Barrier(MPI_COMM_WORLD);
    PRINT("Done\n");

    PRINT("\tBandwidth... ");
    fflush(stdout);

    Timer t;
    MPI_Barrier(MPI_COMM_WORLD);
    timer_reset(&t);
    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < num_samples; ++i)
        sendrecv(src, dst);

    MPI_Barrier(MPI_COMM_WORLD);
    const long double time_elapsed = timer_diff_nsec(t) / 1e9l; // seconds
    PRINT("%Lg GiB/s\n", num_samples * bytes / time_elapsed / (1024 * 1024 * 1024));
    PRINT("\tTransfer time: %Lg ms\n", time_elapsed * 1000 / num_samples);
    MPI_Barrier(MPI_COMM_WORLD);
}

static void
measurebw2(const char* msg, const size_t bytes,
           void (*sendrecv)(uint8_t*, uint8_t*, uint8_t*, uint8_t*), uint8_t* dsrc, uint8_t* hdst,
           uint8_t* hsrc, uint8_t* ddst)
{
    const size_t num_samples = 100;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    PRINT("%s\n", msg);
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT("\tWarming up... ");
    for (size_t i = 0; i < num_samples / 10; ++i)
        sendrecv(dsrc, hdst, hsrc, ddst);

    MPI_Barrier(MPI_COMM_WORLD);
    PRINT("Done\n");

    PRINT("\tBandwidth... ");
    fflush(stdout);

    Timer t;
    MPI_Barrier(MPI_COMM_WORLD);
    timer_reset(&t);
    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < num_samples; ++i)
        sendrecv(dsrc, hdst, hsrc, ddst);

    MPI_Barrier(MPI_COMM_WORLD);
    const long double time_elapsed = timer_diff_nsec(t) / 1e9l; // seconds
    PRINT("%Lg GiB/s\n", num_samples * bytes / time_elapsed / (1024 * 1024 * 1024));
    PRINT("\tTransfer time: %Lg ms\n", time_elapsed * 1000 / num_samples);
    MPI_Barrier(MPI_COMM_WORLD);
}

int
main(void)
{
    MPI_Init(NULL, NULL);
    // int provided;
    // MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    // errchk(provided >= MPI_THREAD_MULTIPLE);

    // Disable stdout buffering
    setbuf(stdout, NULL);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    errchk(nprocs >= 2); // Require at least one neighbor

    MPI_Barrier(MPI_COMM_WORLD);
    if (!pid) {
        printf("Do we have threads? The following should not be ordered (unless very lucky)\n");
#pragma omp parallel for
        for (int i = 0; i < 10; ++i)
            printf("%d, ", i);
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);
    const int device_id = pid % devices_per_node;
    cudaSetDevice(device_id);

    printf("Process %d of %d running.\n", pid, nprocs);
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT("Block size: %u MiB\n", BLOCK_SIZE / (1024 * 1024));

#if 1
    {
        uint8_t* src = allocHost(BLOCK_SIZE);
        uint8_t* dst = allocHost(BLOCK_SIZE);

        measurebw("Unidirectional bandwidth, blocking (Host)", //
                  2 * BLOCK_SIZE, sendrecv_blocking, src, dst);
        measurebw("Bidirectional bandwidth, async (Host)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking, src, dst);
        measurebw("Bidirectional bandwidth, twoway (Host)", //
                  2 * BLOCK_SIZE, sendrecv_twoway, src, dst);
        measurebw("Bidirectional bandwidth, async multiple (Host)", //
                  2 * (nprocs - 1) * BLOCK_SIZE, sendrecv_nonblocking_multiple, src, dst);
        // measurebw("Bidirectional bandwidth, async multiple parallel (Host)", //
        //          2 * (nprocs-1) * BLOCK_SIZE, sendrecv_nonblocking_multiple_parallel, src, dst);

        freeHost(src);
        freeHost(dst);
    }
    PRINT("\n------------------------\n");

    {
        uint8_t* src = allocDevice(BLOCK_SIZE);
        uint8_t* dst = allocDevice(BLOCK_SIZE);

        measurebw("Unidirectional bandwidth, blocking (Device)", //
                  2 * BLOCK_SIZE, sendrecv_blocking, src, dst);
        measurebw("Bidirectional bandwidth, async (Device)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking, src, dst);
        measurebw("Bidirectional bandwidth, twoway (Device)", //
                  2 * BLOCK_SIZE, sendrecv_twoway, src, dst);
        measurebw("Bidirectional bandwidth, async multiple (Device)", //
                  2 * (nprocs - 1) * BLOCK_SIZE, sendrecv_nonblocking_multiple, src, dst);
        // measurebw("Bidirectional bandwidth, async multiple parallel (Device)", //
        //          2 *  (nprocs-1) *BLOCK_SIZE, sendrecv_nonblocking_multiple_parallel, src, dst);
        measurebw("Bidirectional bandwidth, async multiple (Device, rt pinning)", //
                  2 * (nprocs - 1) * BLOCK_SIZE, sendrecv_nonblocking_multiple_rt_pinning, src,
                  dst);

        freeDevice(src);
        freeDevice(dst);
    }
    PRINT("\n------------------------\n");

    {
        uint8_t* src = allocDevicePinned(BLOCK_SIZE);
        uint8_t* dst = allocDevicePinned(BLOCK_SIZE);

        measurebw("Unidirectional bandwidth, blocking (Device, pinned)", //
                  2 * BLOCK_SIZE, sendrecv_blocking, src, dst);
        measurebw("Bidirectional bandwidth, async (Device, pinned)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking, src, dst);
        measurebw("Bidirectional bandwidth, twoway (Device, pinned)", //
                  2 * BLOCK_SIZE, sendrecv_twoway, src, dst);
        measurebw("Bidirectional bandwidth, async multiple (Device, pinned)", //
                  2 * (nprocs - 1) * BLOCK_SIZE, sendrecv_nonblocking_multiple, src, dst);

        freeDevice(src);
        freeDevice(dst);
    }
    PRINT("\n------------------------\n");

    {
        uint8_t* hsrc = allocHost(BLOCK_SIZE);
        uint8_t* hdst = allocHost(BLOCK_SIZE);
        uint8_t* dsrc = allocDevice(BLOCK_SIZE);
        uint8_t* ddst = allocDevice(BLOCK_SIZE);

        measurebw("Unidirectional D2H", BLOCK_SIZE, send_d2h, dsrc, hdst);
        measurebw("Unidirectional H2D", BLOCK_SIZE, send_h2d, hsrc, ddst);

        measurebw2("Bidirectional D2H & H2D", 2 * BLOCK_SIZE, sendrecv_d2h2d, dsrc, hdst, hsrc,
                   ddst);

        freeDevice(dsrc);
        freeDevice(ddst);
        freeHost(hsrc);
        freeHost(hdst);
    }
    PRINT("\n------------------------\n");

    {
        uint8_t* hsrc = allocHost(BLOCK_SIZE);
        uint8_t* hdst = allocHost(BLOCK_SIZE);
        uint8_t* dsrc = allocDevicePinned(BLOCK_SIZE);
        uint8_t* ddst = allocDevicePinned(BLOCK_SIZE);

        measurebw("Unidirectional D2H (pinned)", BLOCK_SIZE, send_d2h, dsrc, hdst);
        measurebw("Unidirectional H2D (pinned)", BLOCK_SIZE, send_h2d, hsrc, ddst);

        measurebw2("Bidirectional D2H & H2D (pinned)", 2 * BLOCK_SIZE, sendrecv_d2h2d, dsrc, hdst,
                   hsrc, ddst);

        freeDevice(dsrc);
        freeDevice(ddst);
        freeHost(hsrc);
        freeHost(hdst);
    }
    PRINT("\n------------------------\n");
#else
    { // Final run for easy identification with the profiler
        uint8_t* src = allocDevice(BLOCK_SIZE);
        uint8_t* dst = allocDevice(BLOCK_SIZE);

        measurebw("Bidirectional bandwidth, async multiple (Device, rt pinning)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking_multiple_rt_pinning, src, dst);

        freeDevice(src);
        freeDevice(dst);
    }
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
