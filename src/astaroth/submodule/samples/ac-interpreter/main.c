#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "astaroth.h"
#include "astaroth_utils.h"

/*
    cmdline interface:

    acInit
    acLaunchKernel
    acSaveToFile
    acLoadFromFile
    acLoadFromArray
    acStoreToArray
    acQuit
*/

AcResult
cmdInit(const size_t nx, const size_t ny, const size_t nz, //
        AcMeshInfo* info, AcMesh* host_mesh, Device* device)
{
    acSetMeshDims(nx, ny, nz, info);
    acHostMeshCreate(*info, host_mesh);
    acDeviceCreate(0, *info, device);

    acPrintMeshInfo(*info);

    return AC_SUCCESS;
}

AcResult
cmdLoadRandom(const Device device, AcMesh host_mesh)
{
    acHostMeshRandomize(&host_mesh);
    acHostMeshApplyPeriodicBounds(&host_mesh);
    return acDeviceLoadMesh(device, STREAM_DEFAULT, host_mesh);
}

AcResult
cmdWriteToFile(const Device device, AcMesh host_mesh, const size_t id)
{
    acDeviceStoreMesh(device, STREAM_DEFAULT, &host_mesh);
    acHostMeshWriteToFile(host_mesh, id);

    return AC_SUCCESS;
}

AcResult
cmdReadFromFile(const Device device, AcMesh host_mesh, const size_t id)
{
    acHostMeshReadFromFile(id, &host_mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, host_mesh);

    return AC_SUCCESS;
}

AcResult
cmdReduce(const Device device, const char* str)
{
    if (!str) {
        fprintf(stderr, "Malformed string (NULL) passed as reduction type\n");
        return AC_FAILURE;
    }

    ReductionType type = NUM_RTYPES;
    for (size_t i = 0; i < NUM_RTYPES; ++i) {
        if (!strcmp(str, rtype_names[i])) {
            type = (ReductionType)i;
            break;
        }
    }
    if (type >= NUM_RTYPES) {
        fprintf(stderr, "Unknown reduction type %s\n", str);
        printf("Available reduction types:\n");
        acQueryRtypes();
        return AC_FAILURE;
    }

    printf("%s:\n", rtype_names[type]);
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        AcReal res;
        acDeviceReduceScal(device, STREAM_DEFAULT, type, i, &res);
        printf("\t%-15s... [%.3g]\n", field_names[i], (double)res);
    }
    return AC_SUCCESS;
}

AcResult
cmdReduceAll(const Device device)
{
    AcResult retval = AC_SUCCESS;

    retval |= cmdReduce(device, "RTYPE_MAX");
    retval |= cmdReduce(device, "RTYPE_MIN");
    /*
    for (size_t j = 0; j < NUM_RTYPES; ++j) {
        for (size_t i = 0; i < NUM_FIELDS; ++i) {
            AcReal res;
            acDeviceReduceScal(device, STREAM_DEFAULT, j, i, &res);
            printf("\t%-15s... [%.3g]\n", field_names[i], res);
        }
    }
    */
    return retval;
}

AcResult
cmdLaunchKernel(const Device device, const AcMeshInfo info, const char* str)
{
    if (!str) {
        fprintf(stderr, "Malformed string (NULL) passed as kernel type\n");
        return AC_FAILURE;
    }

    size_t kernel = NUM_KERNELS;
    for (size_t i = 0; i < NUM_KERNELS; ++i) {
        if (!strcmp(str, kernel_names[i])) {
            kernel = i;
            break;
        }
    }
    if (kernel >= NUM_KERNELS) {
        fprintf(stderr, "Unknown kernel type %s\n", str);
        printf("Available kernels:\n");
        acQueryKernels();
        return AC_FAILURE;
    }

    const int3 start      = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 end        = acConstructInt3Param(AC_nx_max, AC_ny_max, AC_nz_max, info);
    const AcResult retval = acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[kernel], start,
                                                 end);

    if (retval != AC_SUCCESS)
        return retval;

    acDeviceSwapBuffers(device);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
    return AC_SUCCESS;
}

AcResult
cmdQuit(AcMesh* host_mesh, Device* device)
{
    acHostMeshDestroy(host_mesh);
    acDeviceDestroy(*device);

    return AC_SUCCESS;
}

int
main(void)
{
    AcMeshInfo info;
    AcMesh host_mesh;
    Device device;

    bool running = true;
    while (running) {
        printf("Ready\n");

        const size_t MAX_CMD_LEN = 4096;
        char line[MAX_CMD_LEN];
        scanf("%s", line);

        AcResult retval = AC_SUCCESS;
        if (!strcmp(line, "exit") || !strcmp(line, "quit")) {
            retval  = cmdQuit(&host_mesh, &device);
            running = false;
        }
        else if (!strcmp(line, "init")) {

            size_t input0, input1, input2;
            scanf("%lu", &input0);
            scanf("%lu", &input1);
            scanf("%lu", &input2);

            retval = cmdInit(input0, input1, input2, &info, &host_mesh, &device);
        }
        else if (!strcmp(line, "read")) {
            size_t input = 0;
            retval       = cmdReadFromFile(device, host_mesh, input);
        }
        else if (!strcmp(line, "write")) {
            size_t input = 0;
            retval       = cmdWriteToFile(device, host_mesh, input);
        }
        else if (!strcmp(line, "reduce")) {

            scanf("%s", line);
            retval = cmdReduce(device, line);
        }
        else if (!strcmp(line, "reduce_all")) {

            retval = cmdReduceAll(device);
        }
        else if (!strcmp(line, "load_random")) {
            retval = cmdLoadRandom(device, host_mesh);
        }
        else if (!strcmp(line, "launch")) {
            scanf("%s", line);
            retval = cmdLaunchKernel(device, info, line);
        }
        else {
            fprintf(stderr, "Unknown command: '%s'\n", line);
        }
        WARNCHK_ALWAYS(retval == AC_SUCCESS);
    }
    return EXIT_SUCCESS;
}