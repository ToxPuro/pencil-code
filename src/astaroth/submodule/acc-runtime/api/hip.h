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

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

#define CUresult hipError_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cuFloatComplex hipFloatComplex
#define make_cuFloatComplex make_hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSetDevice hipSetDevice
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbolAsync hipMemcpyFromSymbolAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemset hipMemset
#define cudaMalloc hipMalloc
#define cudaMallocHost hipMallocHost
#define cudaFree hipFree
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaMemGetInfo hipMemGetInfo
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaGetDeviceProperties hipGetDeviceProperties

#define cudaStreamQuery hipStreamQuery
#define cudaErrorNotReady hipErrorNotReady
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange

#define cudaProfilerStart roctracer_start
#define cudaProfilerStop roctracer_stop

#define cudaDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
#define cudaFuncSetSharedMemConfig hipFuncSetSharedMemConfig
#define cudaSharedMemBankSizeEightByte hipSharedMemBankSizeEightByte
#define cudaSharedMemBankSizeFourByte hipSharedMemBankSizeFourByte
#define cudaDeviceSetCacheConfig hipDeviceSetCacheConfig
#define cudaFuncCachePreferShared hipFuncCachePreferShared
#define cudaFuncCachePreferL1 hipFuncCachePreferL1

#define curandStateXORWOW_t hiprandStateXORWOW_t
#define curandStateMRG32k3a_t hiprandStateMRG32k3a_t
#define curandState hiprandState
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_uniform_double hiprand_uniform_double
