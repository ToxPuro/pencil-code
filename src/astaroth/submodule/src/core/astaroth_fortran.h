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
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Utils
 */
void achostupdatebuiltinparams_(AcMeshInfo* info);

void acgetdevicecount_(int* count);

/**
 * Device
 */
void acdevicecreate_(const int* id, const AcMeshInfo* info, Device* handle);

void acdevicedestroy_(Device* device);

void acdeviceprintinfo_(const Device* device);

void acdeviceloadmeshinfo_(const Device* device, const AcMeshInfo* info);

void acdeviceloadmesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                       const int* num_farrays, AcReal* farray);

void acdevicestoremesh_(const Device* device, const Stream* stream, const AcMeshInfo* info,
                        const int* num_farrays, AcReal* farray);

void acdeviceintegratesubstep_(const Device* device, const Stream* stream, const int* step_number,
                               const int3* start, const int3* end, const AcReal* dt);
void acdeviceperiodicboundconds_(const Device* device, const Stream* stream, const int3* start,
                                 const int3* end);

void acdeviceswapbuffers_(const Device* device);

void acdevicereducescal_(const Device* device, const Stream* stream, const ReductionType* rtype,
                         const VertexBufferHandle* vtxbuf_handle, AcReal* result);

void acdevicereducevec_(const Device* device, const Stream* stream, const ReductionType* rtype,
                        const VertexBufferHandle* vtxbuf0, const VertexBufferHandle* vtxbuf1,
                        const VertexBufferHandle* vtxbuf2, AcReal* result);

void acdevicesynchronizestream_(const Device* device, const Stream* stream);

#ifdef __cplusplus
} // extern "C"
#endif
