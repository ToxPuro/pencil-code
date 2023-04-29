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
#if AC_MPI_ENABLED
#include "astaroth.h"

#include <stdint.h> //uint64_t

#include "errchk.h"
#include "math_utils.h"

#define MPI_DECOMPOSITION_AXES (3)

static inline uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 3 * bit;
            k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
            i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
        }
    }
    // Just a quick copy/paste for other decomp dims
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 2 * bit;
            j |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            k |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 1 * bit;
            k |= ((pid & (mask << 0)) >> 0 * bit) >> 0;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return (uint3_64){i, j, k};
}

static inline uint64_t
morton1D(const uint3_64 pid)
{
    uint64_t i = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 2 * bit;
            i |= ((pid.y & mask) << 1) << 2 * bit;
            i |= ((pid.x & mask) << 2) << 2 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.y & mask) << 0) << 1 * bit;
            i |= ((pid.z & mask) << 1) << 1 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 0 * bit;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return i;
}

static inline uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};
    printf("getting decomp: %d,%d,%d\n", p.x,p.y,p.z);

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

static inline uint3_64
wrap(const int3 i, const uint3_64 n)
{
    return (uint3_64){
        mod(i.x, n.x),
        mod(i.y, n.y),
        mod(i.z, n.z),
    };
}

static inline int
getPid(const int3 pid_raw, const uint3_64 decomp)
{
    // printf("pid_raw: %d, %d, %d\n", pid_raw.x, pid_raw.y, pid_raw.z);
    // printf("decomp: %d, %d, %d\n", decomp.x, decomp.y, decomp.z);
    const uint3_64 pid = wrap(pid_raw, decomp);
    // return (int)morton1D(pid);
    return pid.z*decomp.x*decomp.y + pid.y*decomp.x + pid.x;

    // Pencil Code decomposition
    // if (lprocz_slowest) then
    //     find_proc = ipz * nprocxy + ipy * nprocx + ipx
    // else
    //     find_proc = ipy * nprocxz + ipz * nprocx + ipx
    // endif
}

static inline int3
getPid3D(const uint64_t pid, const uint3_64 decomp)
{
    // const uint3_64 pid3D = morton3D(pid);
    // ERRCHK_ALWAYS(getPid(static_cast<int3>(pid3D), decomp) == (int)pid);
    // return (int3){(int)pid3D.x, (int)pid3D.y, (int)pid3D.z};

    // int ipx = modulo(iproc, nprocx)
    // int ipy = modulo(iproc/nprocx, nprocy)
    // int ipz = iproc/nprocxy    
    int i_pid = (int)pid;
    int3 pid3D = {i_pid%decomp.x,(i_pid/decomp.x)%decomp.y,i_pid/(decomp.x*decomp.y)};
    ERRCHK_ALWAYS(getPid(pid3D, decomp) == i_pid)  
    return pid3D;

}

/** Assumes that contiguous pids are on the same node and there is one process per GPU. */
static inline bool
onTheSameNode(const uint64_t pid_a, const uint64_t pid_b)
{
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    const uint64_t node_a = pid_a / devices_per_node;
    const uint64_t node_b = pid_b / devices_per_node;

    return node_a == node_b;
}
#endif // AC_MPI_ENABLED
