/**
    Microbenchmark the GPU caches in 1D stencil computations and generate a plottable .csv output

    Examples:
        # Usage
        ./bwtest-benchmark <problem size in bytes> <working set size in bytes>

        # 256 MiB problem size and working set of size 8 (one double), i.e. halo r=0
        ./bwtest-benchmark 268435456 8

        # 3-point von Neumann stencil
        ./bwtest-benchmark 268435456 24

        # Profiling
        cmake -DUSE_HIP=ON .. &&\
        make -j &&\
        rocprof --trace-start off -i ~/rocprof-input-metrics.txt ./bwtest-benchmark 268435456 256

cat ~/rocprof-input-metrics.txt
```
# Perf counters group 1
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
# Perf counters group 2
pmc : TCC_HIT[0], TCC_MISS[0], TCC_HIT_sum, TCC_MISS_sum
# Perf counters group 3
pmc: L2CacheHit MemUnitBusy LDSBankConflict

# Filter by dispatches range, GPU index and kernel names
# supported range formats: "3:9", "3:", "3"
#range: 0 : 16
gpu: 0 1 2 3
#kernel: singlepass_solve
```
*/
#include <stdio.h>
#include <stdlib.h>

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <roctracer_ext.h>   // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // cudaStream_t
#endif

#include "common.h"

// #define USE_SMEM (0) // Set with cmake
// #define MAX_THREADS_PER_BLOCK (0) // Set with cmake

#if USE_SMEM
static size_t
get_smem(const int tpb, const int halo)
{
    return (tpb + 2 * halo) * sizeof(double);
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const int halo, const Array in, Array out)
{
    extern __shared__ double smem[];

    const int base_idx = blockIdx.x * blockDim.x;
    for (int sid = threadIdx.x; sid < (int)(blockDim.x + 2 * halo); sid += blockDim.x)
        if (sid + base_idx < in.count)
            smem[sid] = in.data[sid + base_idx];
    __syncthreads();

    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x) + halo;
    if (tid < in.count - halo) {

        double tmp = 0.0;
        for (int i = 0; i < 2 * halo + 1; ++i)
            tmp += smem[threadIdx.x + i];

        out.data[tid] = tmp;
    }
}
#else
static size_t
get_smem(const int tpb, const int halo)
{
    (void)tpb;  // Unused
    (void)halo; // Unused
    return 0;
}

__global__ void
#if MAX_THREADS_PER_BLOCK
__launch_bounds__(MAX_THREADS_PER_BLOCK)
#endif
    kernel(const int halo, const Array in, Array out)
{
    const int tid = (int)(threadIdx.x + blockIdx.x * blockDim.x);

    if (halo <= tid && tid < (int)in.count - halo) {
        double tmp = 0.0;

        for (int i = -halo; i <= halo; ++i)
            tmp += in.data[tid + i];

        out.data[tid] = tmp;
    }
}
#endif

void
model_kernel(const int halo, const Array in, Array out)
{
    for (int tid = 0; tid < (int)in.count; ++tid) {
        if (halo <= tid && tid < (int)in.count - halo) {

            double tmp = 0.0;
            for (int i = -halo; i <= halo; ++i)
                tmp += in.data[tid + i];

            out.data[tid] = tmp;
        }
    }
}

typedef struct {
    size_t count;
    int halo;
    size_t tpb;
    size_t bpg;
    size_t smem;
} KernelConfig;

/** Returns the optimal threadblock dimensions for a given problem size */
static KernelConfig
autotune(const size_t count, const int halo)
{
    Array a = arrayCreate(count, true);
    Array b = arrayCreate(count, true);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const size_t warp_size             = (size_t)props.warpSize;
    const size_t max_smem              = (size_t)props.sharedMemPerBlock;
    const size_t max_threads_per_block = MAX_THREADS_PER_BLOCK
                                             ? (size_t)min(props.maxThreadsPerBlock,
                                                           MAX_THREADS_PER_BLOCK)
                                             : (size_t)props.maxThreadsPerBlock;

    // Warmup
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart); // Timing start
    for (size_t i = 0; i < 1; ++i)
        kernel<<<1, 1, max_smem>>>(halo, a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaDeviceSynchronize();

    // Tune
    KernelConfig c  = {.count = count, .halo = halo, .tpb = 0, .bpg = 0, .smem = 0};
    float best_time = INFINITY;
    for (size_t tpb = 1; tpb <= max_threads_per_block; ++tpb) {

        if (tpb > max_threads_per_block)
            break;

        if (tpb % warp_size)
            continue;

        const size_t bpg  = (size_t)ceil(1. * count / tpb);
        const size_t smem = get_smem(tpb, halo);

        if (smem > max_smem)
            continue;

        printf("Current KernelConfig {.count = %lu, .halo = %d, .tpb = %lu, .bpg = %lu, .smem = "
               "%lu}",
               c.count, c.halo, tpb, bpg, smem);

        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < 3; ++i)
            kernel<<<bpg, tpb, smem>>>(halo, a, b);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

        ERRCHK_CUDA_KERNEL_ALWAYS();
        //  Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
            // Exit in case of unrecoverable error that needs a device reset
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "Unrecoverable CUDA error\n");
                exit(EXIT_FAILURE);
            }
            continue;
        }

        // printf("KernelConfig {.tpb = %lu, .bpg = %lu}\n", tpb, bpg);
        printf(", Time elapsed: %g ms\n", (double)milliseconds);
        if (milliseconds < best_time) {
            best_time = milliseconds;
            c.tpb     = tpb;
            c.bpg     = bpg;
            c.smem    = smem;
        }
    }
    printf("KernelConfig {.count = %lu, .halo = %d, .tpb = %lu, .bpg = %lu, .smem = %lu}\n",
           c.count, c.halo, c.tpb, c.bpg, c.smem);

    arrayDestroy(&a);
    arrayDestroy(&b);

#if USE_SMEM
    ERRCHK_ALWAYS(c.smem);
#endif

    return c;
}

void
verify(const KernelConfig c)
{
    const size_t count = c.count;
    const size_t tpb   = c.tpb;
    const size_t bpg   = c.bpg;
    const size_t smem  = c.smem;
    const int halo     = c.halo;

    Array ahost = arrayCreate(count, false);
    Array bhost = arrayCreate(count, false);
    Array a     = arrayCreate(count, true);
    Array b     = arrayCreate(count, true);

    arrayRandomize(&ahost);
    model_kernel(halo, ahost, bhost);

    const size_t bytes = count * sizeof(ahost.data[0]);
    cudaMemcpy(a.data, ahost.data, bytes, cudaMemcpyHostToDevice);
    kernel<<<bpg, tpb, smem>>>(halo, a, b);
    cudaMemcpy(ahost.data, b.data, bytes, cudaMemcpyDeviceToHost);

    const double* candidate = ahost.data;
    const double* model     = bhost.data;

    for (size_t i = halo; i < ahost.count - halo; ++i) {
        if (model[i] != candidate[i]) {
            fprintf(stderr, "Failure at %lu: %g (host) and %g (device)\n", i, model[i],
                    candidate[i]);
        }
    }

    arrayDestroy(&a);
    arrayDestroy(&b);
    arrayDestroy(&ahost);
    arrayDestroy(&bhost);

    printf("Results verified\n");
}

static void
benchmark(const KernelConfig c)
{
    const size_t num_iters = 5;

    // Allocate
    Array a = arrayCreate(c.count, true);
    Array b = arrayCreate(c.count, true);

    // Benchmark
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart); // Timing start
    for (size_t i = 0; i < num_iters; ++i)
        kernel<<<c.bpg, c.tpb, c.smem>>>(c.halo, a, b);
    cudaEventRecord(tstop); // Timing stop
    cudaEventSynchronize(tstop);
    ERRCHK_CUDA_KERNEL_ALWAYS();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, tstart, tstop);
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);

    const size_t bytes     = num_iters * sizeof(a.data[0]) * (a.count + b.count - 2 * c.halo);
    const double seconds   = (double)milliseconds / 1e3;
    const double bandwidth = bytes / seconds;
    printf("Effective bandwidth: %g GiB/s\n", bandwidth / pow(1024, 3));
    printf("\tBytes transferred: %g GiB\n", bytes / pow(1024, 3));
    printf("\tTime elapsed: %g ms\n", (double)milliseconds);

    // File
    const char* benchmark_dir = "microbenchmark.csv";
    FILE* fp                  = fopen(benchmark_dir, "a");
    ERRCHK_ALWAYS(fp);
    ERRCHK_ALWAYS(fp);
    // format
    // 'usesmem, maxthreadsperblock, problemsize, workingsetsize, milliseconds, effectivebandwidth'
    fprintf(fp, "%d,%d,%lu,%lu,%g,%g\n", USE_SMEM, MAX_THREADS_PER_BLOCK, c.count * sizeof(double),
            (2 * c.halo + 1) * sizeof(double), (double)milliseconds, bandwidth);
    fclose(fp);

    // Free
    arrayDestroy(&a);
    arrayDestroy(&b);
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
    printf("    Shared memory per block: %lu\n", props.sharedMemPerBlock);
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    printf("--------------------------------------------------\n");
}

int
main(int argc, char* argv[])
{
    cudaProfilerStop();
    if (argc != 3) {
        fprintf(stderr, "Usage: ./benchmark <problem size> <working set size>\n");
        fprintf(stderr, "       ./benchmark 0 0 # To use the defaults\n");
        return EXIT_FAILURE;
    }
    const size_t arg0 = (size_t)atol(argv[1]);
    const size_t arg1 = (size_t)atol(argv[2]);

    const size_t problem_size     = arg0 ? arg0 : 268435456; // 256 MiB default
    const size_t working_set_size = arg1 ? arg1 : 8;         // 8 byte default (r=0)
    const int halo                = ((working_set_size / sizeof(double)) - 1) / 2;
    const size_t count            = problem_size / sizeof(double);
    ERRCHK(working_set_size <= problem_size);

    if (working_set_size > problem_size) {
        fprintf(stderr, "Invalid working set size: %lu > %lu\n", working_set_size, problem_size);
        return EXIT_FAILURE;
    }

    printDeviceInfo(0);
    printf("USE_SMEM=%d\n", USE_SMEM);
    printf("MAX_THREADS_PER_BLOCK=%d\n", MAX_THREADS_PER_BLOCK);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    KernelConfig c = autotune(count, halo);
    verify(c);
    cudaProfilerStart();
    benchmark(c);
    cudaProfilerStop();
    return EXIT_SUCCESS;
}