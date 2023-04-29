#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "timer_hires.h"

static const size_t nx = 256;
static const size_t ny = nx;
static const size_t nz = nx;

int
main(void)
{
    AcMeshInfo info;
    acSetMeshDims(nx, ny, nz, &info);

    Device device;
    acDeviceCreate(0, info, &device);

    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dx, 2 * AC_REAL_PI / nx);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dy, 2 * AC_REAL_PI / ny);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dz, 2 * AC_REAL_PI / nz);

    AcMeshDims dims = acGetMeshDims(info);
    acPrintIntParams(AC_mx, AC_my, AC_mz, info);
    acPrintIntParams(AC_nx, AC_ny, AC_nz, info);

    // Init & dryrun
    const size_t pid   = 0;
    const size_t count = acVertexBufferSize(info);
    acRandInitAlt(1234UL, count, pid);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, init, dims.n0, dims.n1);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, dt, 1e-3);

    acDeviceSynchronizeStream(device, STREAM_ALL);
    Timer t;
    timer_reset(&t);
    acDeviceLaunchKernel(device, STREAM_DEFAULT, solve, dims.n0, dims.n1);
    acDeviceSynchronizeStream(device, STREAM_ALL);
    const double elems_per_second = (nx * ny * nz) / (1e-9 * timer_diff_nsec(t));
    timer_diff_print(t);
    printf("%g M elements per second\n", elems_per_second / 1e6);

    acRandQuit();
    acDeviceDestroy(device);

    return EXIT_SUCCESS;
}