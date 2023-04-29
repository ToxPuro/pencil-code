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

#include <float.h> // DBL/FLT_EPSILON

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_complex.h>
#else
#include <cuComplex.h>    // CUDA complex types
#include <vector_types.h> // CUDA vector types
#endif

#if AC_DOUBLE_PRECISION
typedef double AcReal;
typedef double2 AcReal2;
typedef double3 AcReal3;
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#define AC_REAL_MAX (DBL_MAX)
#define AC_REAL_MIN (DBL_MIN)
#define AcReal3(x, y, z) make_double3(x, y, z)
#define AC_REAL_EPSILON (DBL_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_DOUBLE)
#define AC_REAL_INVALID_VALUE (DBL_MAX)
#else
typedef float AcReal;
typedef float2 AcReal2;
typedef float3 AcReal3;
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#define AC_REAL_MAX (FLT_MAX)
#define AC_REAL_MIN (FLT_MIN)
#define AcReal3(x, y, z) make_float3(x, y, z)
#define AC_REAL_EPSILON (FLT_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_FLOAT)
#define AC_REAL_INVALID_VALUE (FLT_MAX)
#endif

#define AC_REAL_PI ((AcReal)M_PI)

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

typedef struct {
  size_t x, y, z;
} Volume;
