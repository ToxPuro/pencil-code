#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//#include "acc_runtime.h" // For CUDA/HIP support
#include "errchk.h"
#include "math_utils.h"

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#endif

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

void
acArraySet(const uint8_t value, Array* a)
{
    cudaMemset(a->data, value, a->count * sizeof(a->data[0]));
}

__global__ void
array_set_to_tid(Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out.count)
        out.data[tid] = tid;
}

__global__ void
array_set(const AcReal value, Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out.count)
        out.data[tid] = value;
}

__global__ void
kernel(const Array in, Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < in.count)
        out.data[tid] += 2.0 * in.data[tid];
}

__global__ void
kernel_vectorized(const Array in, Array out)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (2 * tid < in.count)
        reinterpret_cast<AcReal2*>(
            &out.data[2 * tid])[0] = reinterpret_cast<AcReal2*>(&out.data[2 * tid])[0] +
                                     2.0 * reinterpret_cast<AcReal2*>(&in.data[2 * tid])[0];
}

void
validate(const Array darr, const AcReal mult)
{
    const size_t bytes = darr.count * sizeof(darr.data[0]);
    AcReal* data       = (AcReal*)malloc(bytes);
    assert(data);

    ERRCHK_CUDA_ALWAYS(cudaMemcpy(data, darr.data, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < darr.count; ++i) {
        const AcReal expected = mult * i;
        if (data[i] != expected) {
            fprintf(stderr, "Validation failed at %lu: expected %g, got %g\n", i, expected,
                    data[i]);
            exit(EXIT_FAILURE);
        }
    }
    free(data);
}

void
benchmark(const size_t count)
{
    Array a = acArrayCreate(count);
    Array b = acArrayCreate(count);

    const size_t tpb = 512;
    const size_t bpg = (size_t)ceil(1.0 * count / tpb);
    assert(a.count == count);
    assert(b.count == count);

    // Warmup
    for (size_t i = 0; i < 10; ++i)
        kernel<<<bpg, tpb>>>(a, b);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart); // Timing start
    kernel<<<bpg, tpb>>>(a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);

    ERRCHK_CUDA_KERNEL_ALWAYS();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tstart, tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);

    // Validate
    array_set_to_tid<<<bpg, tpb>>>(a);
    array_set<<<bpg, tpb>>>((AcReal)0.0, b);
    kernel<<<bpg, tpb>>>(a, b);
    validate(b, 2.0);

    const size_t bytes   = 3 * count * sizeof(a.data[0]);
    const double seconds = (double)milliseconds / 1e3;
    printf("Bandwidth: %g GiB/s\n", bytes / seconds / pow(1024, 3));
    printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
    printf("\tTime elapsed: %g ms\n", (double)milliseconds);

    acArrayDestroy(&a);
    acArrayDestroy(&b);
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
    const size_t nn     = 128;
    const size_t fields = 8;
    const size_t count  = fields * (size_t)pow(nn, 3); // Approx what we do (lower bound)
#elif 0
    const size_t count = 10;
#endif

    const size_t num_iters = 10;
    for (size_t i = 0; i < num_iters; ++i)
        benchmark(count);

    return EXIT_SUCCESS;
}
