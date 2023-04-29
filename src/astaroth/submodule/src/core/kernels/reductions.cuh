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
#pragma once
#include <assert.h>

// Function pointer definitions
typedef AcReal (*MapFn)(const AcReal&);
typedef AcReal (*MapVecFn)(const AcReal&, const AcReal&, const AcReal&);
typedef AcReal (*MapVecScalFn)(const AcReal&, const AcReal&, const AcReal&, const AcReal&);
typedef AcReal (*ReduceFn)(const AcReal&, const AcReal&);

// Map functions
static __device__ inline AcReal
map_value(const AcReal& a)
{
    return a;
}

static __device__ inline AcReal
map_square(const AcReal& a)
{
    return a * a;
}

static __device__ inline AcReal
map_exp_square(const AcReal& a)
{
    return exp(a) * exp(a);
}

static __device__ inline AcReal
map_length_vec(const AcReal& a, const AcReal& b, const AcReal& c)
{
    return sqrt(a * a + b * b + c * c);
}

static __device__ inline AcReal
map_square_vec(const AcReal& a, const AcReal& b, const AcReal& c)
{
    return map_square(a) + map_square(b) + map_square(c);
}

static __device__ inline AcReal
map_exp_square_vec(const AcReal& a, const AcReal& b, const AcReal& c)
{
    return map_exp_square(a) + map_exp_square(b) + map_exp_square(c);
}

static __device__ inline AcReal
map_length_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d)
{
    return sqrt(a * a + b * b + c * c) / sqrt(exp(d));
}

static __device__ inline AcReal
map_square_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d)
{
    return (map_square(a) + map_square(b) + map_square(c)) / (exp(d));
}

// Reduce functions
static __device__ inline AcReal
reduce_max(const AcReal& a, const AcReal& b)
{
    return a > b ? a : b;
}

static __device__ inline AcReal
reduce_min(const AcReal& a, const AcReal& b)
{
    return a < b ? a : b;
}

static __device__ inline AcReal
reduce_sum(const AcReal& a, const AcReal& b)
{
    return a + b;
}

/** Map data from a 3D array into a 1D array */
template <MapFn map_fn>
__global__ void
map(const AcReal* in, const int3 start, const int3 end, AcReal* out)
{
    assert((start >= (int3){0, 0, 0}));
    assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));

    const int3 tid = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    const int3 in_idx3d = start + tid;
    const size_t in_idx = IDX(in_idx3d);

    const int3 dims      = end - start;
    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;

    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
    if (within_bounds)
        out[out_idx] = map_fn(in[in_idx]);
}

template <MapVecFn map_fn>
__global__ void
map_vec(const AcReal* in0, const AcReal* in1, const AcReal* in2, const int3 start, const int3 end,
        AcReal* out)
{
    assert((start >= (int3){0, 0, 0}));
    assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));

    const int3 tid = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    const int3 in_idx3d = start + tid;
    const size_t in_idx = IDX(in_idx3d);

    const int3 dims      = end - start;
    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;

    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
    if (within_bounds)
        out[out_idx] = map_fn(in0[in_idx], in1[in_idx], in2[in_idx]);
}

template <MapVecScalFn map_fn>
__global__ void
map_vec_scal(const AcReal* in0, const AcReal* in1, const AcReal* in2, const AcReal* in3,
             const int3 start, const int3 end, AcReal* out)
{
    assert((start >= (int3){0, 0, 0}));
    assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));

    const int3 tid = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    const int3 in_idx3d = start + tid;
    const size_t in_idx = IDX(in_idx3d);

    const int3 dims      = end - start;
    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;

    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
    if (within_bounds)
        out[out_idx] = map_fn(in0[in_idx], in1[in_idx], in2[in_idx], in3[in_idx]);
}

template <ReduceFn reduce_fn>
__global__ void
reduce(const AcReal* in, const size_t count, AcReal* out)
{
    // Note: possible integer overflow when GPU memory becomes large enough
    const size_t curr = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ AcReal smem[];
    if (curr < count)
        smem[threadIdx.x] = in[curr];
    else
        smem[threadIdx.x] = AC_REAL_INVALID_VALUE;

    __syncthreads();

    size_t offset = blockDim.x / 2;
    while (offset > 0) {
        if (threadIdx.x < offset) {
            const AcReal a = smem[threadIdx.x];
            const AcReal b = smem[threadIdx.x + offset];

            // If the mesh dimensions are not divisible by mapping tbdims, and mapping tb dims are
            // not divisible by reduction tb dims, then it is possible for `a` to be invalid but `b`
            // to be valid
            if (a != AC_REAL_INVALID_VALUE && b != AC_REAL_INVALID_VALUE)
                smem[threadIdx.x] = reduce_fn(a, b);
            else if (a != AC_REAL_INVALID_VALUE)
                smem[threadIdx.x] = a;
            else
                smem[threadIdx.x] = b;
        }

        offset /= 2;
        __syncthreads();
    }

    if (!threadIdx.x)
        out[blockIdx.x] = smem[threadIdx.x];
}

static void
swap_ptrs(AcReal** a, AcReal** b)
{
    AcReal* tmp = *a;
    *a          = *b;
    *b          = tmp;
}

static Volume
get_map_tpb(void)
{
    return (Volume){32, 4, 1};
}

static Volume
get_map_bpg(const int3 dims, const Volume tpb)
{
    return (Volume){
        as_size_t(int(ceil(double(dims.x) / tpb.x))),
        as_size_t(int(ceil(double(dims.y) / tpb.y))),
        as_size_t(int(ceil(double(dims.z) / tpb.z))),
    };
}

size_t
acKernelReduceGetMinimumScratchpadSize(const int3 max_dims)
{
    const Volume tpb   = get_map_tpb();
    const Volume bpg   = get_map_bpg(max_dims, tpb);
    const size_t count = tpb.x * bpg.x * tpb.y * bpg.y * tpb.z * bpg.z;
    return count;
}

size_t
acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims)
{
    return sizeof(AcReal) * acKernelReduceGetMinimumScratchpadSize(max_dims);
}

AcReal
acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const AcReal* vtxbuf,
                   const int3 start, const int3 end, AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                   const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();

    ERRCHK_ALWAYS(NUM_REDUCE_SCRATCHPADS >= 2);
    AcReal* in  = scratchpads[0];
    AcReal* out = scratchpads[1];

    // Compute block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    // Map
    {
        const Volume tpb = get_map_tpb();
        const Volume bpg = get_map_bpg(dims, tpb);

        switch (rtype) {
        case RTYPE_MAX: /* Fallthrough */
        case RTYPE_MIN: /* Fallthrough */
        case RTYPE_SUM:
            map<map_value><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf, start, end, out);
            break;
        case RTYPE_RMS:
            map<map_square><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf, start, end, out);
            break;
        case RTYPE_RMS_EXP:
            map<map_exp_square><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf, start, end, out);
            break;
        default:
            ERROR("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };
        swap_ptrs(&in, &out);
    }

    // Reduce
    size_t count = initial_count;
    do {
        const size_t tpb  = 128;
        const size_t bpg  = as_size_t(ceil(double(count) / tpb));
        const size_t smem = tpb * sizeof(in[0]);

        switch (rtype) {
        case RTYPE_MAX:
            reduce<reduce_max><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_MIN:
            reduce<reduce_min><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_SUM: /* Fallthrough */
        case RTYPE_RMS: /* Fallthrough */
        case RTYPE_RMS_EXP:
            reduce<reduce_sum><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        default:
            ERROR("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };
        ERRCHK_CUDA_KERNEL();
        swap_ptrs(&in, &out);

        count = bpg;
    } while (count > 1);

    // Copy the result back to host
    AcReal result;
    cudaMemcpyAsync(&result, in, sizeof(in[0]), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();
    // fprintf(stderr, "%s device result %g\n", rtype_names[rtype], result);
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                  const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                  const AcReal* vtxbuf2, AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                  const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();

    ERRCHK_ALWAYS(NUM_REDUCE_SCRATCHPADS >= 2);
    AcReal* in  = scratchpads[0];
    AcReal* out = scratchpads[1];

    // Set thread block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    // Map
    {
        const Volume tpb = get_map_tpb();
        const Volume bpg = get_map_bpg(dims, tpb);
        switch (rtype) {
        case RTYPE_MAX: /* Fallthrough */
        case RTYPE_MIN: /* Fallthrough */
        case RTYPE_SUM:
            map_vec<map_length_vec><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf0, vtxbuf1,
                                                                               vtxbuf2, start, end,
                                                                               out);
            break;
        case RTYPE_RMS:
            map_vec<map_square_vec><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf0, vtxbuf1,
                                                                               vtxbuf2, start, end,
                                                                               out);
            break;
        case RTYPE_RMS_EXP:
            map_vec<map_exp_square_vec><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf0, vtxbuf1,
                                                                                   vtxbuf2, start,
                                                                                   end, out);
            break;
        default:
            ERROR("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };
        swap_ptrs(&in, &out);
    }

    // Reduce
    size_t count = initial_count;
    do {
        const size_t tpb  = 128;
        const size_t bpg  = as_size_t(ceil(double(count) / tpb));
        const size_t smem = tpb * sizeof(in[0]);

        switch (rtype) {
        case RTYPE_MAX:
            reduce<reduce_max><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_MIN:
            reduce<reduce_min><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_SUM: /* Fallthrough */
        case RTYPE_RMS: /* Fallthrough */
        case RTYPE_RMS_EXP:
            reduce<reduce_sum><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        default:
            ERROR("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };
        ERRCHK_CUDA_KERNEL();
        swap_ptrs(&in, &out);

        count = bpg;
    } while (count > 1);

    // Copy the result back to host
    AcReal result;
    cudaMemcpyAsync(&result, in, sizeof(in[0]), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();
    return result;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                      const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                      const AcReal* vtxbuf2, const AcReal* vtxbuf3,
                      AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS], const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();

    ERRCHK_ALWAYS(NUM_REDUCE_SCRATCHPADS >= 2);
    AcReal* in  = scratchpads[0];
    AcReal* out = scratchpads[1];

    // Set thread block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    // Map
    {
        const Volume tpb = get_map_tpb();
        const Volume bpg = get_map_bpg(dims, tpb);
        switch (rtype) {
        case RTYPE_ALFVEN_MAX: /* Fallthrough */
        case RTYPE_ALFVEN_MIN:
            map_vec_scal<map_length_alf>
                <<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                                            start, end, out);
            break;
        case RTYPE_ALFVEN_RMS:
            map_vec_scal<map_square_alf>
                <<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                                            start, end, out);
            break;
        default:
            fprintf(stderr, "Rtype %s (%d)\n", rtype_names[rtype], rtype);
            ERROR("Invalid reduction type in acKernelReduceVecScal");
            return AC_FAILURE;
        };
        swap_ptrs(&in, &out);
    }

    // Reduce
    size_t count = initial_count;
    do {
        const size_t tpb  = 128;
        const size_t bpg  = as_size_t(ceil(double(count) / tpb));
        const size_t smem = tpb * sizeof(in[0]);

        switch (rtype) {
        case RTYPE_ALFVEN_MAX:
            reduce<reduce_max><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_ALFVEN_MIN:
            reduce<reduce_min><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        case RTYPE_ALFVEN_RMS:
            reduce<reduce_sum><<<bpg, tpb, smem, stream>>>(in, count, out);
            break;
        default:
            ERROR("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };
        ERRCHK_CUDA_KERNEL();
        swap_ptrs(&in, &out);

        count = bpg;
    } while (count > 1);

    // Copy the result back to host
    AcReal result;
    cudaMemcpyAsync(&result, in, sizeof(in[0]), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();
    return result;
}