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
    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#include <vector>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (100)

int
main(void)
{
    int retval    = 0;
    const int pid = 0;

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(32, 32, 32, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    //
    const int3 mmin = (int3){0, 0, 0};
    const int3 mmax = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    const int3 nmin = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 nmax = acConstructInt3Param(AC_nx_max, AC_ny_max, AC_nz_max, info);

    // GPU alloc & compute
    Device device;
    acDeviceCreate(0, info, &device);

    // Boundconds
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, mmin, mmax);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
        acHostMeshRandomize(&model);
        acHostMeshApplyPeriodicBounds(&model);
    }

    // Dryrun
    const AcReal dt = (AcReal)FLT_EPSILON;
    for (int i = 0; i < 3; ++i)
        acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, nmin, nmax, dt);

    // Integration
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDeviceSwapBuffers(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);

    for (size_t j = 0; j < NUM_INTEGRATION_STEPS; ++j) {
        for (int i = 0; i < 3; ++i) {
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, mmin, mmax);
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, nmin, nmax, dt);
            acDeviceSwapBuffers(device);
        }
    }

    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, mmin, mmax);
    acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate);
    if (pid == 0) {

        // Host integrate
        for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
            acHostIntegrateStep(model, dt);

        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }

        srand(123567);
        acHostMeshRandomize(&model);
        // acHostMeshSet((AcReal)1.0, &model);
        acHostMeshApplyPeriodicBounds(&model);
    }

    // Scalar reductions
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    acDevicePeriodicBoundconds(device, STREAM_DEFAULT, mmin, mmax);

    if (pid == 0)
        printf("---Test: Scalar reductions---\n");

    const ReductionType scal_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                             RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(scal_reductions); ++i) { // NOTE: not using NUM_RTYPES here
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        AcReal candval;

        const ReductionType rtype = scal_reductions[i];

        acDeviceReduceScal(device, STREAM_DEFAULT, rtype, v0, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceScal(model, rtype, v0);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);

            if (!acEvalError(rtype_names[rtype], error)) {
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    // Vector reductions
    if (pid == 0)
        printf("---Test: Vector reductions---\n");

    const ReductionType vec_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                            RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(vec_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        AcReal candval;

        const ReductionType rtype = vec_reductions[i];
        acDeviceReduceVec(device, STREAM_DEFAULT, rtype, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceVec(model, rtype, v0, v1, v2);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);

            if (!acEvalError(rtype_names[rtype], error)) {
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    // Alfven reductions
    if (pid == 0)
        printf("---Test: Alfven reductions---\n");

    const ReductionType alf_reductions[] = {RTYPE_ALFVEN_MAX, RTYPE_ALFVEN_MIN, RTYPE_ALFVEN_RMS};
    for (size_t i = 0; i < ARRAY_SIZE(alf_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        const VertexBufferHandle v3 = (VertexBufferHandle)3;
        AcReal candval;

        const ReductionType rtype = alf_reductions[i];
        acDeviceReduceVecScal(device, STREAM_DEFAULT, rtype, v0, v1, v2, v3, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceVecScal(model, rtype, v0, v1, v2, v3);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MAX, v0, v1, v2, v3);
            error.minimum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MIN, v0, v1, v1, v3);

            if (!acEvalError(rtype_names[rtype], error)) {
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acDeviceDestroy(device);

    fflush(stdout);

    fprintf(stderr, "devicetest complete: %s\n",
            retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return EXIT_SUCCESS;
}
