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
    Building and running:
        cmake -DDSL_MODULE_DIR=samples/multikerneltest \
              -DPROGRAM_MODULE_DIR=samples/multikerneltest \
              -DBUILD_SAMPLES=OFF -DBUILD_UTILS=OFF -DMPI_ENABLED=OFF ..
        make -j
        ./multikerneltest
*/
#include "astaroth.h"

static const int nn = 64;
static const int mm = nn + 2 * NGHOST;

void
fibostep(const Device device)
{
    const int3 start = (int3){0, 0, 0};
    const int3 end   = (int3){mm, mm, mm};

    acDevice_step(device, STREAM_DEFAULT, start, end);
    acDeviceSwapBuffer(device, VTXBUF_FIBO);

    AcReal val;
    acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, VTXBUF_FIBO, &val);
    printf("%f\n", (double)val);
}

int
main(void)
{
    AcMeshInfo info;
    info.int_params[AC_nx] = info.int_params[AC_ny] = info.int_params[AC_nz] = nn;
    acHostUpdateBuiltinParams(&info);

    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    const int3 start = (int3){0, 0, 0};
    const int3 end   = (int3){mm, mm, mm};

    // acDevice_solve(device, STREAM_DEFAULT, start, end);
    acDevice_clear(device, STREAM_DEFAULT, start, end);
    acDeviceSwapBuffer(device, VTXBUF_FIBO);
    // MV NOTE: Boundary conditions after swapping
    acDevice_set(device, STREAM_DEFAULT, start, end);
    acDeviceSwapBuffer(device, VTXBUF_FIBO);

    for (size_t i = 0; i < 20; ++i)
        fibostep(device);

    acDeviceDestroy(device);
}
