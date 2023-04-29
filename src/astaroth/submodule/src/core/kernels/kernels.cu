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
#include "kernels.h"

#include "acc_runtime.cu"

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    // Commented out until issues on lumi sorted
    // acComplex a = exp(acComplex(1, 1) * AcReal(1));
    AcReal3 a = (AcReal)2.0 * (AcReal3){1, 2, 3};
    (void)a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

// Built-in kernels
#include "boundconds.cuh"
#include "boundconds_miikka_GBC.cuh"
#include "packing.cuh"
#include "reductions.cuh"
#include "volume_copy.cuh"
#include "pack_unpack.cuh"

AcResult
acKernel(const KernelParameters params, VertexBufferArray vba)
{
#ifdef AC_INTEGRATION_ENABLED
    // TODO: Why is AC_step_number loaded here??
    acLoadIntUniform(params.stream, AC_step_number, params.step_number);
    acLaunchKernel(params.kernel, params.stream, params.start, params.end, vba);
    return AC_SUCCESS;
#else
    (void)params; // Unused
    (void)vba;    // Unused
    ERROR("acKernel() called but AC_step_number not defined!");
    return AC_FAILURE;
#endif
}
#if PACKED_DATA_TRANSFERS
void
acUnpackPlate(const Device device, int3 start, int3 end, int block_size, const Stream stream, PlateType plate)
{
    const dim3 tpb(256, 1, 1);
    const dim3 bpg((uint)ceil((block_size * NUM_VTXBUF_HANDLES) / (float)tpb.x), 1, 1);

    packUnpackPlate<AC_H2D><<<bpg, tpb, 0, device->streams[stream]>>>(device->plate_buffers[plate], device->vba, start, end);
}

void
acPackPlate(const Device device, int3 start, int3 end, int block_size, const Stream stream, PlateType plate)
{                               
    const dim3 tpb(256, 1, 1);
    const dim3 bpg((uint)ceil((block_size * NUM_VTXBUF_HANDLES) / (float)tpb.x), 1, 1);
    acKernelPackData(device->streams[stream], device->vba, start, end - start, device->plate_buffers[plate]);
    //packUnpackPlate<AC_D2H><<<bpg, tpb, 0, device->streams[stream]>>>(device->plate_buffers[plate], device->vba, start, end);
}
#endif

