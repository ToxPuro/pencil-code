#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

int
main(void)
{
    // Setup the mesh configuration
    AcMeshInfo info;
    acLoadConfig("../samples/plasma-meets-ai-workshop/astaroth.conf", &info);

    // Allocate memory on the GPU
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    const AcMeshDims dims = acGetMeshDims(info);

    // Setup initial conditions
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh);
    // acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, 2);
    // acDeviceLaunchKernel(device, STREAM_DEFAULT, hydro, dims.n0, dims.n1);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

    // Write the initial snapshot to a file
    acHostMeshApplyPeriodicBounds(&mesh);
    acHostMeshWriteToFile(mesh, 0);

    // Scale velocity to [-1, 1]
    acDeviceLaunchKernel(device, STREAM_DEFAULT, scale_velocity, dims.n0, dims.n1);
    acDeviceSwapBuffer(device, UUX);
    acDeviceSwapBuffer(device, UUY);
    acDeviceSwapBuffer(device, UUZ);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

    for (size_t i = 1; i < 2000; ++i) {
        // Update timestep
        acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-3);

        // Apply forcing
        acDeviceLaunchKernel(device, STREAM_DEFAULT, forcing, dims.n0, dims.n1);
        acDeviceSwapBuffer(device, UUX);
        acDeviceSwapBuffer(device, UUY);
        acDeviceSwapBuffer(device, UUZ);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

        // Smooth velocity
        acDeviceLaunchKernel(device, STREAM_DEFAULT, smooth, dims.n0, dims.n1);
        acDeviceSwapBuffer(device, UUX);
        acDeviceSwapBuffer(device, UUY);
        acDeviceSwapBuffer(device, UUZ);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

        for (size_t substep = 0; substep < 3; ++substep) {

            acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep);

            // Compute SGS stresses
            acDeviceLaunchKernel(device, STREAM_DEFAULT, compute_sgs_stress, dims.n0, dims.n1);
            acDeviceSwapBuffer(device, T00);
            acDeviceSwapBuffer(device, T01);
            acDeviceSwapBuffer(device, T02);
            acDeviceSwapBuffer(device, T11);
            acDeviceSwapBuffer(device, T12);
            acDeviceSwapBuffer(device, T22);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

            // Compute
            acDeviceLaunchKernel(device, STREAM_DEFAULT, hydro_sgs, dims.n0, dims.n1);
            acDeviceSwapBuffer(device, UUX);
            acDeviceSwapBuffer(device, UUY);
            acDeviceSwapBuffer(device, UUZ);
            acDeviceSwapBuffer(device, LNRHO);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
        }

        if (!(i % 100)) {
            // Store to host memory and write to a file
            acDeviceSynchronizeStream(device, STREAM_DEFAULT);
            acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
            acDeviceSynchronizeStream(device, STREAM_DEFAULT);
            acHostMeshWriteToFile(mesh, i);
        }
    }

    // Deallocate memory on the GPU
    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);
    return EXIT_SUCCESS;
}
// B211