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
#pragma once

static __global__ void
kernel_volume_copy(const AcReal* in, const int3 in_offset, const int3 in_volume, //
                   AcReal* out, const int3 out_offset, const int3 out_volume)
{
    const int3 idx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };
    if (idx.x > min(in_volume.x, out_volume.x) || //
        idx.y > min(in_volume.y, out_volume.y) || //
        idx.z > min(in_volume.z, out_volume.z))
        return;

    const int3 in_pos  = idx + in_offset;
    const int3 out_pos = idx + out_offset;

    const size_t in_idx = in_pos.x +               //
                          in_pos.y * in_volume.x + //
                          in_pos.z * in_volume.x * in_volume.y;
    const size_t out_idx = out_pos.x +                //
                           out_pos.y * out_volume.x + //
                           out_pos.z * out_volume.x * out_volume.y;

    out[out_idx] = in[in_idx];
}

AcResult
acKernelVolumeCopy(const cudaStream_t stream,                                    //
                   const AcReal* in, const int3 in_offset, const int3 in_volume, //
                   AcReal* out, const int3 out_offset, const int3 out_volume)
{
    const int3 nn = min(in_volume, out_volume);
    const dim3 tpb(min(512, nn.x), 1, 1);
    const dim3 bpg((unsigned int)ceil(nn.x / double(tpb.x)),
                   (unsigned int)ceil(nn.y / double(tpb.y)),
                   (unsigned int)ceil(nn.z / double(tpb.z)));

    kernel_volume_copy<<<bpg, tpb, 0, stream>>>(in, in_offset, in_volume, //
                                                out, out_offset, out_volume);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}
