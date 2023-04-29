#pragma once
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
  size_t x, y, z;
} Volume;
#include "math.h"

#define ORIGINAL (0)
#define ORIGINAL_WITH_ILP (1)
#define EXPL_REG_VARS (2)
#define FULLY_EXPL_REG_VARS (3)
#define EXPL_REG_VARS_AND_CT_CONST_STENCILS (4)
#define FULLY_EXPL_REG_VARS_AND_PINGPONG_REGISTERS (5)
#define SMEM_AND_VECTORIZED_LOADS (6)
#define SMEM_AND_VECTORIZED_LOADS_PINGPONG (7)
#define SMEM_AND_VECTORIZED_LOADS_FULL (8)
#define SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC (9)
#define SMEM_HIGH_OCCUPANCY (10)
#define SMEM_HIGH_OCCUPANCY_CT_CONST_TB (11)
#define SMEM_GENERIC_BLOCKED (12)
#define SMEM_GENERIC_BLOCKED_1D (13)
#define FULLY_EXPL_REG_VARS_AND_HALO_THREADS (14)
#define FULLY_EXPL_REG_VARS_AND_HALO_THREADS_1D_SHUFFLE (15)
#define SMEM_AND_VECTORIZED_LOADS_AND_PINGPONG_AND_ONDEMAND_STENCIL_COMPUTATION \
  (16)
#define NUM_IMPLEMENTATIONS (17) // Note last implementation define

// NOTE: need to do a clean build when switching to/from smem (suspected
// cmake dependency issue)
//#define IMPLEMENTATION (3)
//#define MAX_THREADS_PER_BLOCK (256) // If 0, disables __launch_bounds__
//#define MAX_THREADS_PER_BLOCK (0)

#if IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS ||                             \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL ||                        \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC ||                  \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_PINGPONG

#define VECTORIZED_LOADS (1)
const char* realtype   = "double";
const char* veclen_str = "4";
const size_t veclen    = 4;

#if IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS
const size_t buffers = 1;
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_PINGPONG
const size_t buffers = 2;
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL ||                      \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC
const size_t buffers = NUM_FIELDS;
#endif

size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  return buffers * (x + stencil_order) * (y + stencil_order) *
         (z + stencil_order) * bytes_per_elem;
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  // TODO recheck if needed
  if (veclen != 1 && STENCIL_ORDER != 4)
    return false;

  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return true;
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#elif IMPLEMENTATION ==                                                        \
    SMEM_AND_VECTORIZED_LOADS_AND_PINGPONG_AND_ONDEMAND_STENCIL_COMPUTATION
size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)x;              // Unused
  (void)y;              // Unused
  (void)z;              // Unused
  (void)stencil_order;  // Unused
  (void)bytes_per_elem; // Unused
  return 0;
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return true;
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#elif IMPLEMENTATION == SMEM_HIGH_OCCUPANCY
size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)y; // Unused
  (void)z; // Unused
  return bytes_per_elem * (x + stencil_order);
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return true;
}
Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#elif IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
const size_t nx = 32;
const size_t ny = 2;
const size_t nz = 2;

size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)y; // Unused
  (void)z; // Unused
  return bytes_per_elem * (x + stencil_order);
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return (x == nx) && (y == ny) && (z == nz);
}
#elif IMPLEMENTATION == SMEM_GENERIC_BLOCKED
#define VECTORIZED_LOADS (1)
const char* realtype   = "double";
const char* veclen_str = "2";
const size_t veclen    = 2;
const size_t buffers   = 1;

size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  return buffers * (x + stencil_order) * (y + stencil_order) * bytes_per_elem;
  // return (x + stencil_order) * bytes_per_elem;
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return z == 1;
}
Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#elif IMPLEMENTATION == SMEM_GENERIC_BLOCKED_1D
#define VECTORIZED_LOADS (1)
const char* realtype   = "double";
const char* veclen_str = "2";
const size_t veclen    = 2;
const size_t buffers   = 1;

size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  return buffers * (x + stencil_order) * bytes_per_elem;
}

bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return (y == 1) && (z == 1);
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#else
size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)x;              // Unused
  (void)y;              // Unused
  (void)z;              // Unused
  (void)stencil_order;  // Unused
  (void)bytes_per_elem; // Unused
  return 0;
}

#if IMPLEMENTATION == FULLY_EXPL_REG_VARS_AND_HALO_THREADS
bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  if (x < STENCIL_WIDTH)
    return false;
  if (y < STENCIL_HEIGHT)
    return false;
  if (z < STENCIL_DEPTH)
    return false;

  return true;
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  const size_t rx = (STENCIL_WIDTH - 1) / 2;
  const size_t ry = (STENCIL_HEIGHT - 1) / 2;
  const size_t rz = (STENCIL_DEPTH - 1) / 2;
  const size_t tx = tpb.x - 2 * rx;
  const size_t ty = tpb.y - 2 * ry;
  const size_t tz = tpb.z - 2 * rz;
  return (Volume){
      (size_t)ceil(1. * dims.x / tx),
      (size_t)ceil(1. * dims.y / ty),
      (size_t)ceil(1. * dims.z / tz),
  };
}
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS_AND_HALO_THREADS_1D_SHUFFLE
bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  /*
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (x != props.warpSize)
    return false;
    */
  if (x != 32)
    return false;

  return true;
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  const size_t rx = (STENCIL_WIDTH - 1) / 2;
  const size_t tx = tpb.x - 2 * rx;
  return (Volume){
      (size_t)ceil(1. * dims.x / tx),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#else
bool
is_valid_configuration(const size_t x, const size_t y, const size_t z)
{
  if (MAX_THREADS_PER_BLOCK && x * y * z > MAX_THREADS_PER_BLOCK)
    return false;

  return true;
}

Volume
get_bpg(const Volume dims, const Volume tpb)
{
  return (Volume){
      (size_t)ceil(1. * dims.x / tpb.x),
      (size_t)ceil(1. * dims.y / tpb.y),
      (size_t)ceil(1. * dims.z / tpb.z),
  };
}
#endif
#endif
