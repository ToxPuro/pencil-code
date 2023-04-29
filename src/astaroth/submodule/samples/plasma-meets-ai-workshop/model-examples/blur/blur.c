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
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);

    // Write the initial snapshot to a file
    acHostMeshWriteToFile(mesh, 0);

    // Compute
    acDeviceLaunchKernel(device, STREAM_DEFAULT, blur, dims.n0, dims.n1);
    acDeviceSwapBuffers(device);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    // Store to host memory and write to a file
    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    acHostMeshWriteToFile(mesh, 1);

    for (size_t i = 2; i < 20; ++i) {
        // Compute
        acDeviceLaunchKernel(device, STREAM_DEFAULT, blur, dims.n0, dims.n1);
        acDeviceSwapBuffers(device);
        acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        // Store to host memory and write to a file
        acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
        acHostMeshWriteToFile(mesh, i);
    }

    // Deallocate memory on the GPU
    acDeviceDestroy(device);
    acHostMeshDestroy(&mesh);
    return EXIT_SUCCESS;
}