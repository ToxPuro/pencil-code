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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#pragma once
#include <stdbool.h>
#include <stdint.h> // SIZE_MAX
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h> // cuda_assert
#endif

/*
 * =============================================================================
 * General error checking
 * =============================================================================
 */
// clang-format off
#define ERROR(str)                                                             \
  {                                                                            \
    time_t terr;                                                               \
    time(&terr);                                                               \
    fprintf(stderr, "\n\n\n\n┌──────────────────────── ERROR ───────────────────────────┐\n\n"); \
    fprintf(stderr, "%s", ctime(&terr));                                       \
    fprintf(stderr, "Error in file %s line %d: %s\n", __FILE__, __LINE__, str); \
    fprintf(stderr, "\n└──────────────────────── ERROR ───────────────────────────┘\n\n\n\n"); \
    fflush(stderr);                                                            \
    exit(EXIT_FAILURE);                                                        \
    abort();                                                                   \
  }
// clang-format on

#define WARNING(str)                                                           \
  {                                                                            \
    time_t terr;                                                               \
    time(&terr);                                                               \
    fprintf(stderr, "%s", ctime(&terr));                                       \
    fprintf(stderr, "\tWarning in file %s line %d: %s\n", __FILE__, __LINE__,  \
            str);                                                              \
    fflush(stderr);                                                            \
  }

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(retval)                                                         \
  {                                                                            \
    if (!(retval))                                                             \
      ERROR(#retval " was false");                                             \
  }
#define WARNCHK(retval)                                                        \
  {                                                                            \
    if (!(retval))                                                             \
      WARNING(#retval " was false");                                           \
  }
#define WARNCHK_ALWAYS(retval)                                                 \
  {                                                                            \
    if (!(retval))                                                             \
      WARNING(#retval " was false");                                           \
  }
#define ERRCHK_ALWAYS(retval)                                                  \
  {                                                                            \
    if (!(retval))                                                             \
      ERROR(#retval " was false");                                             \
  }

/*
 * =============================================================================
 * CUDA-specific error checking
 * =============================================================================
 */
// #if defined(__CUDA_RUNTIME_API_H__)
static inline void
cuda_assert(cudaError_t code, const char* file, int line, bool abort)
{
  if (code != cudaSuccess) {
    time_t terr;
    time(&terr);
    fprintf(stderr, "%s", ctime(&terr));
    fprintf(stderr, "\tCUDA error in file %s line %d: %s\n", file, line,
            cudaGetErrorString(code));
    fflush(stderr);

    if (abort)
      exit(code);
  }
}

#ifdef NDEBUG
#undef ERRCHK
#undef WARNCHK
#define ERRCHK(params)
#define WARNCHK(params)
#define ERRCHK_CUDA(params) params
#define WARNCHK_CUDA(params) params
#define ERRCHK_CUDA_KERNEL()                                                   \
  {                                                                            \
  }
#else
#define ERRCHK_CUDA(params)                                                    \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }
#define WARNCHK_CUDA(params)                                                   \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, false);                          \
  }

#define ERRCHK_CUDA_KERNEL()                                                   \
  {                                                                            \
    ERRCHK_CUDA(cudaPeekAtLastError());                                        \
    ERRCHK_CUDA(cudaDeviceSynchronize());                                      \
  }
#endif

#define ERRCHK_CUDA_ALWAYS(params)                                             \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }

#define ERRCHK_CUDA_KERNEL_ALWAYS()                                            \
  {                                                                            \
    ERRCHK_CUDA_ALWAYS(cudaPeekAtLastError());                                 \
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());                               \
  }

#define WARNCHK_CUDA_ALWAYS(params)                                            \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, false);                          \
  }
// #endif // __CUDA_RUNTIME_API_H__

#ifdef __cplusplus
template <typename T>
static inline size_t
as_size_t(const T i)
{
  ERRCHK_ALWAYS(i >= 0);
  ERRCHK_ALWAYS(static_cast<long double>(i) <
                static_cast<long double>(SIZE_MAX));
  return static_cast<size_t>(i);
}
#else
static inline int
as_size_t(const int i)
{
  ERRCHK_ALWAYS(i >= 0);
  ERRCHK_ALWAYS((long double)(i) < (long double)(SIZE_MAX));
  return (size_t)(i);
}
#endif