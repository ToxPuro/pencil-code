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

    for (size_t i = 1; i < 2000; ++i) {
        // EXERCISE: Load some real value (1e-3 is god) to the device constant handle `AC_dt`
        // acDeviceLoadScalarUniform(device, STREAM_DEFAULT, <device constant handle>, 1e-3);
        for (size_t substep = 0; substep < 3; ++substep) {
            // EXERCISE: Load an int corresponding to the RK3 substep to `AC_step_number`
            // acDeviceLoadIntUniform(device, STREAM_DEFAULT, <device constant handle>, substep);

            // Compute
            // EXERCISE: Launch kernel `hydro`
            // acDeviceLaunchKernel(device, STREAM_DEFAULT, <kernel name>, dims.n0, dims.n1);
            acDeviceSwapBuffers(device);
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