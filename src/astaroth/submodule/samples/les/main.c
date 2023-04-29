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
#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"
#include "user_defines.h"

static AcMesh tmp_mesh;

void
save_slice(const Device device, const size_t id)
{
    acDeviceStoreMesh(device, STREAM_DEFAULT, &tmp_mesh);
    acHostMeshWriteToFile(tmp_mesh, id);
    /*
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    acHostMeshApplyPeriodicBounds(&mesh);

    acHostMeshWriteToFile(mesh, id);
    acHostMeshDestroy(&mesh);
    */

#define WRITE_FILES_WITH_PYTHON (0)
#if WRITE_FILES_WITH_PYTHON
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        const size_t len = 4096;
        char buf[len];
        snprintf(buf, len, "../samples/les/analysis.py data-format.csv %s-%.05lu.dat",
                 field_names[i], id);

        FILE* proc = popen(buf, "r");
        ERRCHK_ALWAYS(proc);
        fclose(proc);
    }
#endif
}

int
main(void)
{
    ERRCHK_ALWAYS(acCheckDeviceAvailability() == AC_SUCCESS);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const int nn = 32;
    acSetMeshDims(nn, nn, nn, &info);
    acPrintMeshInfo(info);

    AcResult retval;
    const int3 n0 = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 n1 = acConstructInt3Param(AC_nx_max, AC_ny_max, AC_nz_max, info);
    const int3 m0 = (int3){0, 0, 0};
    const int3 m1 = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    acHostMeshCreate(info, &tmp_mesh);

    // Alloc
    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    acHostMeshRandomize(&mesh);
    acHostMeshApplyPeriodicBounds(&mesh);

    // Init device
    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-3);
    acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, 2);

    // Verify that the mesh was loaded and stored correctly
    AcMesh candidate;
    acHostMeshCreate(info, &candidate);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    retval = acVerifyMesh("Load/Store", mesh, candidate);
    ERRCHK_ALWAYS(retval == AC_SUCCESS);

    // Verify that reading and writing to file works correctly
    acHostMeshWriteToFile(candidate, 0);
    acHostMeshReadFromFile(0, &candidate);
    retval = acVerifyMesh("Read/Write", mesh, candidate);
    ERRCHK_ALWAYS(retval == AC_SUCCESS);

    // Verify that boundconds work correctly
    acHostMeshRandomize(&mesh);
    acHostMeshRandomize(&candidate);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);

    acHostMeshApplyPeriodicBounds(&mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    retval = acVerifyMesh("Boundconds", mesh, candidate);
    ERRCHK_ALWAYS(retval == AC_SUCCESS);

    // Warmup
    for (size_t i = 0; i < NUM_KERNELS; ++i) {
        printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], n0, n1);
    }

    // Benchmark
    cudaProfilerStart();
    for (size_t i = 0; i < NUM_KERNELS; ++i) {
        printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
        acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], n0, n1);
    }
    cudaProfilerStop();

    acHostMeshRandomize(&mesh);
    acHostVertexBufferSet(LNRHO, 1, &mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);

    /*
    // acHostVertexBufferSet(LNRHO, 1, &mesh);
    acHostMeshApplyPeriodicBounds(&mesh);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);

    acHostMeshRandomize(&candidate);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    acVerifyMesh("Boundconds", mesh, candidate);
    acHostMeshDestroy(&candidate);
    */

    printf("VTXBUF ranges before integration:\n");
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        AcReal min, max;
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
        printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
    }

    save_slice(device, 0);
    for (size_t step = 1; step < 2500; ++step) {
        /*
        for (size_t i = 0; i < NUM_KERNELS; ++i) {
            printf("Launching kernel %s (%p)...\n", kernel_names[i], kernels[i]);
            acDeviceLaunchKernel(device, STREAM_DEFAULT, kernels[i], n0, n1);
            acDeviceSwapBuffers(device);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
        }
        */
        /*
         for (size_t substep = 0; substep < 3; ++substep) {
             acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-2);
             acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep);

             acDeviceLaunchKernel(device, STREAM_DEFAULT, compute_stress_tensor_tau, n0, n1);
             acDeviceSwapBuffer(device, T00);
             acDeviceSwapBuffer(device, T01);
             acDeviceSwapBuffer(device, T02);
             acDeviceSwapBuffer(device, T11);
             acDeviceSwapBuffer(device, T12);
             acDeviceSwapBuffer(device, T22);
             acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
             // Note: the above boundcond step does all fields instead of just the stress tensor
             acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, n0, n1);
             acDeviceSwapBuffer(device, UUX);
             acDeviceSwapBuffer(device, UUY);
             acDeviceSwapBuffer(device, UUZ);
             acDeviceSwapBuffer(device, LNRHO);
             acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
         }
         */

#if 0 // WORKS
        acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-2);
        for (int substep = 0; substep < 3; ++substep) {
            acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep);

            acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, n0, n1);
            acDeviceSwapBuffers(device);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
        }
#endif
#if 1 // with stress tensor
        acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-2);
        for (int substep = 0; substep < 3; ++substep) {
            acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep);

            acDeviceLaunchKernel(device, STREAM_DEFAULT, compute_stress_tensor_tau, n0, n1);
            acDeviceSwapBuffer(device, T00);
            acDeviceSwapBuffer(device, T01);
            acDeviceSwapBuffer(device, T02);
            acDeviceSwapBuffer(device, T11);
            acDeviceSwapBuffer(device, T12);
            acDeviceSwapBuffer(device, T22);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);

            acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, n0, n1);
            acDeviceSwapBuffer(device, UUX);
            acDeviceSwapBuffer(device, UUY);
            acDeviceSwapBuffer(device, UUZ);
            acDeviceSwapBuffer(device, LNRHO);
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, m0, m1);
        }
#endif

        // Write to disk
        if (step % 25 == 0) {
            acDeviceSynchronizeStream(device, STREAM_ALL);
            save_slice(device, step);
        }

        printf("Step %lu\n", step);
        for (size_t i = 0; i < NUM_FIELDS; ++i) {
            AcReal min, max;
            acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
            acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
            printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
            if (isnan(min) || isnan(max)) {
                exit(EXIT_FAILURE);
                return EXIT_FAILURE;
            }
        }
    }

    printf("Done.\nVTXBUF ranges after integration:\n");
    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        AcReal min, max;
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, i, &min);
        acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MAX, i, &max);
        printf("\t%-15s... [%.3g, %.3g]\n", field_names[i], min, max);
    }

    acHostMeshDestroy(&candidate);
    acHostMeshDestroy(&mesh);
    acHostMeshDestroy(&tmp_mesh);
    acDeviceDestroy(device);

    return EXIT_SUCCESS;
}
