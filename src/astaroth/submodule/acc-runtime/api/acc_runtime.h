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
#include <stdio.h>

#if AC_USE_HIP
#include "hip.h"

#include <hip/hip_runtime_api.h> // Streams
#include <roctracer_ext.h>       // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // Streams
#endif

#include "datatypes.h"
#include "errchk.h"

#include "user_defines.h"

#define NUM_REDUCE_SCRATCHPADS (2)

typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

typedef struct {
  AcReal* in[NUM_FIELDS];
  AcReal* out[NUM_FIELDS];
  size_t bytes;
} VertexBufferArray;

typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);

#ifdef __cplusplus
extern "C" {
#endif

#include "user_declarations.h"

AcResult acKernelFlush(AcReal* arr, const size_t n);

VertexBufferArray acVBACreate(const size_t count);

void acVBADestroy(VertexBufferArray* vba);

AcResult acRandInit(const uint64_t seed, const Volume m_local,
                    const Volume m_global, const Volume global_offset);

AcResult acRandInitAlt(const uint64_t seed, const size_t count,
                       const size_t rank);

void acRandQuit(void);

AcResult acLaunchKernel(Kernel func, const cudaStream_t stream,
                        const int3 start, const int3 end,
                        VertexBufferArray vba);

AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t stream,
              const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

AcResult
acStoreStencil(const Stencil stencil, const cudaStream_t stream,
               AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

AcResult acLoadRealUniform(const cudaStream_t stream, const AcRealParam param,
                           const AcReal value);
AcResult acLoadReal3Uniform(const cudaStream_t stream, const AcReal3Param param,
                            const AcReal3 value);
AcResult acLoadIntUniform(const cudaStream_t stream, const AcIntParam param,
                          const int value);
AcResult acLoadInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                           const int3 value);

AcResult acStoreRealUniform(const cudaStream_t stream, const AcRealParam param,
                            AcReal* value);
AcResult acStoreReal3Uniform(const cudaStream_t stream,
                             const AcReal3Param param, AcReal3* value);
AcResult acStoreIntUniform(const cudaStream_t stream, const AcIntParam param,
                           int* value);
AcResult acStoreInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                            int3* value);

#ifdef __cplusplus
} // extern "C"
#endif
