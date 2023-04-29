#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//#include "acc_runtime.h" // For CUDA/HIP support
#include "errchk.h"
#include "datatypes.h"

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#endif

#define HALO (440)
#if AC_USE_HIP
#define MAX_SMEM (192 * 1024)
#else
#define MAX_SMEM (16 * 1024)
#endif
#define USE_SMEM (0)

typedef struct {
    size_t count;
    AcReal* data;
} Array;

Array
acArrayCreate(const size_t count)
{
    Array a;

    a.count            = count;
    const size_t bytes = count * sizeof(a.data[0]);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&a.data, bytes));

    return a;
}

void
acArrayDestroy(Array* a)
{
    cudaFree(a->data);
    a->data  = NULL;
    a->count = 0;
}

__global__ void
acArraySetToTID(Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out.count)
        out.data[tid] = tid;
}

__global__ void
acArraySet(const AcReal value, Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out.count)
        out.data[tid] = value;
}

void
printDeviceInfo(const int device_id)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    printf("--------------------------------------------------\n");
    printf("Device Number: %d\n", device_id);
    const size_t bus_id_max_len = 128;
    char bus_id[bus_id_max_len];
    cudaDeviceGetPCIBusId(bus_id, bus_id_max_len, device_id);
    printf("  PCI bus ID: %s\n", bus_id);
    printf("    Device name: %s\n", props.name);
    printf("    Compute capability: %d.%d\n", props.major, props.minor);

    // Compute
    printf("  Compute\n");
    printf("    Clock rate (GHz): %g\n", props.clockRate / 1e6); // KHz -> GHz
    printf("    Stream processors: %d\n", props.multiProcessorCount);
#if !AC_USE_HIP
    printf("    SP to DP flops performance ratio: %d:1\n", props.singleToDoublePrecisionPerfRatio);
#endif
    printf(
        "    Compute mode: %d\n",
        (int)props
            .computeMode); // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
    // Memory
    printf("  Global memory\n");
    printf("    Memory Clock Rate (MHz): %d\n", props.memoryClockRate / (1000));
    printf("    Memory Bus Width (bits): %d\n", props.memoryBusWidth);
    printf("    Peak Memory Bandwidth (GiB/s): %f\n",
           2 * (props.memoryClockRate * 1e3) * props.memoryBusWidth / (8. * 1024. * 1024. * 1024.));
    printf("    ECC enabled: %d\n", props.ECCEnabled);

    // Memory usage
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes = total_bytes - free_bytes;
    printf("    Total global mem: %.2f GiB\n", props.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("    Gmem used (GiB): %.2f\n", used_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory free (GiB): %.2f\n", free_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory total (GiB): %.2f\n", total_bytes / (1024.0 * 1024 * 1024));
    printf("  Caches\n");
#if !AC_USE_HIP
    printf("    Local L1 cache supported: %d\n", props.localL1CacheSupported);
    printf("    Global L1 cache supported: %d\n", props.globalL1CacheSupported);
#endif
    printf("    L2 size: %d KiB\n", props.l2CacheSize / (1024));
    // MV: props.totalConstMem and props.sharedMemPerBlock cause assembler error
    // MV: while compiling in TIARA gp cluster. Therefore commeted out.
    //!!    printf("    Total const mem: %ld KiB\n", props.totalConstMem / (1024));
    //!!    printf("    Shared mem per block: %ld KiB\n", props.sharedMemPerBlock / (1024));
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    // printf("    Single to double perf. ratio: %dx\n",
    // props.singleToDoublePrecisionPerfRatio); //Not supported with older CUDA
    // versions
#if !AC_USE_HIP
    printf("    Stream priorities supported: %d\n", props.streamPrioritiesSupported);
#endif
    printf("    AcReal precision: %lu bits\n", 8 * sizeof(AcReal));
    printf("--------------------------------------------------\n");
}

#if USE_SMEM
__global__ void
kernel(const Array in, Array out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x + HALO;

    extern __shared__ AcReal smem[];
    smem[HALO + threadIdx.x] = in.data[tid];

    if (threadIdx.x < HALO) {
        smem[threadIdx.x]                     = in.data[tid - HALO];
        smem[HALO + blockDim.x + threadIdx.x] = in.data[tid + blockDim.x];
    }
    __syncthreads();

    if (tid >= HALO && tid < in.count - HALO) {
        AcReal tmp = 0;
#pragma unroll
        for (int i = 0; i <= 2 * HALO; ++i)
            tmp += 2.0 * smem[threadIdx.x + i];

        out.data[tid] += tmp;
    }
}
#else
__global__ void
kernel(const Array in, const Array out)
{
    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);

    if (tid >= HALO && tid < (int)in.count - HALO) {
        AcReal tmp = 0;

#pragma unroll
        for (int i = -HALO; i <= HALO; ++i)
            tmp += 2.0 * in.data[tid + i];

        out.data[tid] += tmp;
    }
}
/*
__global__ void
kernel(const Array in, const Array out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x + HALO;
    if (tid >= HALO + in.count)
        return;

    AcReal tmp = 0;

#pragma unroll
    for (int i = -HALO; i <= HALO; ++i)
        tmp += 2.0 * in.data[tid + i];

    out.data[tid] += tmp;
}
*/
/*
__global__ void
kernel(const Array in, const Array out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= HALO && tid < in.count - HALO) {
        AcReal tmp[] = {0.0, 0.0};

#pragma unroll
        for (int i = -HALO; i <= HALO; ++i) {
            tmp[0] += 2.0 * in.data[tid + i];
            if (tid + i * blockDim.x >= 0 && tid + i * blockDim.x < in.count)
                tmp[1] += 2.0 * in.data[tid + i * blockDim.x];
        }

        out.data[tid] += tmp[0] + 0 * tmp[1];
    }
}
*/
/*
__global__ void
kernel(const Array in, const Array out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= HALO && tid < in.count - HALO) {
        AcReal tmp[2 * HALO + 1];

#pragma unroll
        for (int i = 0; i < 2 * HALO + 1; ++i)
            tmp[i] = 2.0 * in.data[tid - HALO + i];

#pragma unroll
        for (int i = 1; i < 2 * HALO + 1; ++i)
            tmp[0] += tmp[i];

        out.data[tid] += tmp[0];
    }
}
*/
#endif

void
validate(const Array darr, const AcReal mult)
{
    const size_t bytes = darr.count * sizeof(darr.data[0]);
    AcReal* data       = (AcReal*)malloc(bytes);
    assert(data);

    ERRCHK_CUDA_ALWAYS(cudaMemcpy(data, darr.data, bytes, cudaMemcpyDeviceToHost));

    for (int i = HALO; i < (int)darr.count - HALO; ++i) {
        AcReal expected = 0;
        for (int j = -HALO; j <= HALO; ++j)
            expected += mult * (i + j);
        if (data[i] != expected) {
            fprintf(stderr, "Validation failed at %d: expected %g, got %g\n", i, expected, data[i]);
            exit(EXIT_FAILURE);
        }
    }
    free(data);
}

typedef struct {
    size_t tpb;
    size_t bpg;
    size_t smem;
} KernelConfig;

/** Returns the optimal threadblock dimensions for a given problem size */
static KernelConfig
autotune(const Array a, const Array b)
{
    cudaEvent_t tstart, tstop;
    const size_t count     = a.count;
    const size_t num_iters = 3;

    float best_time = INFINITY;
    KernelConfig c  = {
         .tpb  = 256,
         .bpg  = (size_t)ceil(1.0 * count / 256),
         .smem = 0,
    };
#if USE_SMEM
    c.smem = (c.tpb + 2 * HALO) * sizeof(AcReal);
#endif

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const size_t warp_size             = (size_t)props.warpSize;
    const size_t max_threads_per_block = (size_t)props.maxThreadsPerBlock;

    // Warmup
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart); // Timing start
    for (size_t i = 0; i < 10; ++i)
        kernel<<<c.bpg, c.tpb, c.smem>>>(a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaDeviceSynchronize();

    // Tune
    for (size_t tpb = 1; tpb <= max_threads_per_block; ++tpb) {
        if (tpb > max_threads_per_block)
            break;

        const size_t bpg = (size_t)ceil(1.0 * count / tpb);
#if USE_SMEM
        const size_t smem = (tpb + 2 * HALO) * sizeof(AcReal);
#else
        if ((tpb % warp_size))
            continue;
        const size_t smem = 0;
#endif

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (size_t i = 0; i < num_iters; ++i)
            kernel<<<bpg, tpb, smem>>>(a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
            // Exit in case of unrecoverable error that needs a device reset
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Unrecoverable CUDA error\n");
                exit(EXIT_FAILURE);
            }
            continue;
        }

        // printf("KernelConfig {.tpb = %lu, .bpg = %lu, .smem = %lu}\n", tpb, bpg, smem);
        // printf("\tTime elapsed: %g ms\n", (double)milliseconds);
        if (milliseconds < best_time) {
            best_time = milliseconds;
            c         = (KernelConfig){.tpb = tpb, .bpg = bpg, .smem = smem};
        }
    }
    printf("KernelConfig {.tpb = %lu, .bpg = %lu, .smem = %lu}\n", c.tpb, c.bpg, c.smem);

    if (c.smem > MAX_SMEM) {
        fprintf(stderr, "Attempted to use too much shared memory (%lu > %d)\n", c.smem, MAX_SMEM);
        exit(EXIT_FAILURE);
    }
    return c;
}

void
benchmark(const Array a, const Array b, const KernelConfig c)
{
    const size_t count = a.count;

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart); // Timing start
    kernel<<<c.bpg, c.tpb, c.smem>>>(a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);

    ERRCHK_CUDA_KERNEL_ALWAYS();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tstart, tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);

    // Validate
    acArraySetToTID<<<c.bpg, c.tpb>>>(a);
    acArraySet<<<c.bpg, c.tpb>>>((AcReal)0.0, b);
    kernel<<<c.bpg, c.tpb, c.smem>>>(a, b);
    validate(b, 2.0);

    const size_t bytes   = (count + 2 * (count - 2 * HALO)) * sizeof(a.data[0]);
    const double seconds = (double)milliseconds / 1e3;
    printf("Bandwidth: %g GiB/s\n", bytes / seconds / pow(1024, 3));
    printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
    printf("\tTime elapsed: %g ms\n", (double)milliseconds);
}

int
main(void)
{
    printDeviceInfo(0);

#if 0
    const size_t count = 4 * pow(1024, 3) / sizeof(AcReal);
#elif 0
    const size_t halo   = 3;
    const size_t nn     = 128;
    const size_t mm     = nn + 2 * halo;
    const size_t fields = 8;
    const size_t count  = fields * (pow(mm, 3) + 2 * pow(nn, 3));
#elif 1
    const size_t nn     = 64;
    const size_t fields = 8;
    const size_t count  = fields * (size_t)pow(nn, 3); // Approx what we do (lower bound)
#elif 0
    const size_t count = 10;
#endif

    Array a = acArrayCreate(count);
    Array b = acArrayCreate(count);
    assert(a.count == count);
    assert(b.count == count);
    const KernelConfig c = autotune(a, b);

    const size_t num_iters = 5;
    for (size_t i = 0; i < num_iters; ++i)
        benchmark(a, b, c);

    acArrayDestroy(&a);
    acArrayDestroy(&b);
    return EXIT_SUCCESS;
}
