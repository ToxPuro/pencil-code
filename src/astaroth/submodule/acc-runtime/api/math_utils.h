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
#include <math.h>   // isnan, isinf
#include <stdint.h> // uint64_t
#include <stdlib.h> // rand

#include "datatypes.h"
#include "errchk.h"

#if AC_DOUBLE_PRECISION != 1
#define exp(x) expf(x)
#define sin(x) sinf(x)
#define cos(x) cosf(x)
#define sqrt(x) sqrtf(x)
#define fabs(x) fabsf(x)
#endif

#define UNUSED __attribute__((unused))

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__ UNUSED
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__ UNUSED
#else
#define HOST_DEVICE UNUSED
#define HOST_DEVICE_INLINE inline UNUSED
#endif // __CUDACC__

// Disabled for now, issues on lumi (exp, cos, sin ambiguous)
#define ENABLE_COMPLEX_DATATYPE (0)
#if ENABLE_COMPLEX_DATATYPE
static __device__ AcReal
cos(const AcReal& val)
{
  return cos(val);
}

static __device__ AcReal
sin(const AcReal& val)
{
  return sin(val);
}

static __device__ AcReal
exp(const AcReal& val)
{
  return exp(val);
}

static __device__ inline acComplex
expc(const acComplex& val)
{
  return acComplex(exp(val.x) * cos(val.y), exp(val.x) * sin(val.y));
}

#if defined(__CUDACC__)
// These are already overloaded in the HIP API
/*
static HOST_DEVICE_INLINE acComplex
operator*(const AcReal& a, const acComplex& b)
{
  return (acComplex){a * b.x, a * b.y};
}

static HOST_DEVICE_INLINE acComplex
operator*(const acComplex& b, const AcReal& a)
{
  return (acComplex){a * b.x, a * b.y};
}
*/

static HOST_DEVICE_INLINE acComplex
operator*(const acComplex& a, const acComplex& b)
{
  return (acComplex){a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
#endif
#endif // ENABLE_COMPLEX_DATATYPE

typedef struct uint3_64 {
  uint64_t x, y, z;
  explicit inline constexpr operator int3() const
  {
    return (int3){(int)x, (int)y, (int)z};
  }
} uint3_64;

template <class T>
static HOST_DEVICE_INLINE constexpr const T
val(const T& a)
{
  return a;
}

template <class T>
static HOST_DEVICE_INLINE constexpr const T
sum(const T& a, const T& b)
{
  return a + b;
}

template <class T>
static HOST_DEVICE_INLINE constexpr const T
max(const T& a, const T& b)
{
  return a > b ? a : b;
}

template <class T>
static HOST_DEVICE_INLINE constexpr const T
min(const T& a, const T& b)
{
  return a < b ? a : b;
}

static inline const int3
max(const int3& a, const int3& b)
{
  return (int3){max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

static inline const int3
min(const int3& a, const int3& b)
{
  return (int3){min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

template <class T>
static inline const T
clamp(const T& val, const T& min, const T& max)
{
  return val < min ? min : val > max ? max : val;
}

static inline uint64_t
mod(const int a, const int b)
{
  const int r = a % b;
  return r < 0 ? as_size_t(r + b) : as_size_t(r);
}

static inline AcReal
randr()
{
  return AcReal(rand()) / AcReal(RAND_MAX);
}

static inline bool
is_power_of_two(const unsigned val)
{
  return val && !(val & (val - 1));
}

/*
 * INT3
 */
static HOST_DEVICE_INLINE int3
operator+(const int3& a, const int3& b)
{
  return (int3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a, const int3& b)
{
  return (int3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a)
{
  return (int3){-a.x, -a.y, -a.z};
}

static HOST_DEVICE_INLINE int3
operator*(const int3& a, const int3& b)
{
  return (int3){a.x * b.x, a.y * b.y, a.z * b.z};
}

static HOST_DEVICE_INLINE int3
operator*(const int& a, const int3& b)
{
  return (int3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE bool
operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

static HOST_DEVICE_INLINE bool
operator!=(const int3& a, const int3& b)
{
  return !(a == b);
}

static HOST_DEVICE_INLINE bool
operator>=(const int3& a, const int3& b)
{
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

static HOST_DEVICE_INLINE bool
operator<=(const int3& a, const int3& b)
{
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

/*
 * UINT3_64
 */
static HOST_DEVICE_INLINE uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_DEVICE_INLINE uint3_64
operator-(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_DEVICE_INLINE uint3_64
operator*(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x * b.x, a.y * b.y, a.z * b.z};
}

static inline uint3_64
operator*(const int& a, const uint3_64& b)
{
  return (uint3_64){as_size_t(a) * b.x, as_size_t(a) * b.y, as_size_t(a) * b.z};
}

static HOST_DEVICE_INLINE bool
operator==(const uint3_64& a, const uint3_64& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

/*
 * Volume
 */
template <class T>
static Volume
to_volume(const T a)
{
  return (Volume){as_size_t(a.x), as_size_t(a.y), as_size_t(a.z)};
}

static inline dim3
to_dim3(const Volume v)
{
  return dim3(v.x, v.y, v.z);
}

/*
 * AcReal
 */
static HOST_DEVICE_INLINE bool
is_valid(const AcReal a)
{
  return !isnan(a) && !isinf(a);
}

/*
 * AcReal2
 */
#if defined(__CUDACC__)
static HOST_DEVICE_INLINE AcReal2
operator+(const AcReal2& a, const AcReal2& b)
{
  return (AcReal2){a.x + b.x, a.y + b.y};
}

static HOST_DEVICE_INLINE void
operator+=(AcReal2& lhs, const AcReal2& rhs)
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
}

static HOST_DEVICE_INLINE AcReal2
operator-(const AcReal2& a, const AcReal2& b)
{
  return (AcReal2){a.x - b.x, a.y - b.y};
}

static HOST_DEVICE_INLINE AcReal2
operator-(const AcReal2& a)
{
  return (AcReal2){-a.x, -a.y};
}

static HOST_DEVICE_INLINE void
operator-=(AcReal2& lhs, const AcReal2& rhs)
{
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
}

static HOST_DEVICE_INLINE AcReal2
operator*(const AcReal& a, const AcReal2& b)
{
  return (AcReal2){a * b.x, a * b.y};
}

static HOST_DEVICE_INLINE AcReal2
operator*(const AcReal2& b, const AcReal& a)
{
  return (AcReal2){a * b.x, a * b.y};
}

static HOST_DEVICE_INLINE AcReal2
operator/(const AcReal2& a, const AcReal& b)
{
  return (AcReal2){a.x / b, a.y / b};
}

#endif

static HOST_DEVICE_INLINE AcReal
dot(const AcReal2& a, const AcReal2& b)
{
  return a.x * b.x + a.y * b.y;
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal2& a)
{
  return is_valid(a.x) && is_valid(a.y);
}

/*
 * AcReal3
 */
static HOST_DEVICE_INLINE AcReal3
operator+(const AcReal3& a, const AcReal3& b)
{
  return (AcReal3){a.x + b.x, a.y + b.y, a.z + b.z};
}


static HOST_DEVICE_INLINE int3
operator+(const int3& a, const int b)
{
    return (int3){a.x + b, a.y + b, a.z + b};
}

static HOST_DEVICE_INLINE int3
operator+(const int a, const int3& b)
{
    return (int3){a + b.x, a + b.y, a + b.z};
}

static HOST_DEVICE_INLINE void
operator+=(AcReal3& lhs, const AcReal3& rhs)
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
}

static HOST_DEVICE_INLINE AcReal3
operator-(const AcReal3& a, const AcReal3& b)
{
  return (AcReal3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator-(const AcReal3& a)
{
  return (AcReal3){-a.x, -a.y, -a.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a, const int b)
{
    return (int3){a.x - b, a.y - b, a.z - b};
}

static HOST_DEVICE_INLINE int3
operator-(const int a, const int3& b)
{
    return (int3){a - b.x, a - b.y, a - b.z};
}

static HOST_DEVICE_INLINE void
operator-=(AcReal3& lhs, const AcReal3& rhs)
{
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
}

static HOST_DEVICE_INLINE AcReal3
operator*(const AcReal& a, const AcReal3& b)
{
  return (AcReal3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator*(const AcReal3& b, const AcReal& a)
{
  return (AcReal3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator/(const AcReal3& a, const AcReal& b)
{
  return (AcReal3){a.x / b, a.y / b, a.z / b};
}

static HOST_DEVICE_INLINE AcReal
dot(const AcReal3& a, const AcReal3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static HOST_DEVICE_INLINE AcReal3
cross(const AcReal3& a, const AcReal3& b)
{
  AcReal3 c;

  c.x = a.y * b.z - a.z * b.y;
  c.y = a.z * b.x - a.x * b.z;
  c.z = a.x * b.y - a.y * b.x;

  return c;
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal3& a)
{
  return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}

/*
 * AcMatrix
 */
typedef struct AcMatrix {
  AcReal data[3][3] = {{0}};

  HOST_DEVICE AcMatrix() {}

  HOST_DEVICE AcMatrix(const AcReal3 row0, const AcReal3 row1,
                       const AcReal3 row2)
  {
    data[0][0] = row0.x;
    data[0][1] = row0.y;
    data[0][2] = row0.z;

    data[1][0] = row1.x;
    data[1][1] = row1.y;
    data[1][2] = row1.z;

    data[2][0] = row2.x;
    data[2][1] = row2.y;
    data[2][2] = row2.z;
  }

  HOST_DEVICE AcReal3 row(const int row) const
  {
    return (AcReal3){data[row][0], data[row][1], data[row][2]};
  }

  HOST_DEVICE AcReal3 operator*(const AcReal3& v) const
  {
    return (AcReal3){
        dot(row(0), v),
        dot(row(1), v),
        dot(row(2), v),
    };
  }

  HOST_DEVICE AcMatrix operator-() const
  {
    return AcMatrix(-row(0), -row(1), -row(2));
  }
} AcMatrix;

static HOST_DEVICE AcMatrix
operator*(const AcReal v, const AcMatrix& m)
{
  AcMatrix out;

  out.data[0][0] = v * m.data[0][0];
  out.data[0][1] = v * m.data[0][1];
  out.data[0][2] = v * m.data[0][2];

  out.data[1][0] = v * m.data[1][0];
  out.data[1][1] = v * m.data[1][1];
  out.data[1][2] = v * m.data[1][2];

  out.data[2][0] = v * m.data[2][0];
  out.data[2][1] = v * m.data[2][1];
  out.data[2][2] = v * m.data[2][2];

  return out;
}

static HOST_DEVICE AcMatrix
operator-(const AcMatrix& A, const AcMatrix& B)
{
  return AcMatrix(A.row(0) - B.row(0), //
                  A.row(1) - B.row(1), //
                  A.row(2) - B.row(2));
}