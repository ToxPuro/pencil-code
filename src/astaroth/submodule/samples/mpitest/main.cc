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

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (100)

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    const int max_devices = 2 * 2 * 4;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d). Please modify "
                "mpitest/main.cc to use a larger mesh.\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(2 * 9, 2 * 11, 4 * 7, &info);
    // acSetMeshDims(32, 32, 32, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // Load/Store
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        const AcResult res = acVerifyMesh("Load/Store", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    // Boundconds
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Periodic boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    // Dryrun
    const AcReal dt = (AcReal)FLT_EPSILON;
    acGridIntegrate(STREAM_DEFAULT, dt);

    // Integration
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    // Device integrate
    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        acGridIntegrate(STREAM_DEFAULT, dt);

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);

        // Host integrate
        for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
            acHostIntegrateStep(model, dt);

        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    // Scalar reductions
    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        acHostMeshRandomize(&model);
        acHostMeshApplyPeriodicBounds(&model);
    }
    fflush(stdout);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    const ReductionType scal_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                             RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(scal_reductions); ++i) { // NOTE: not using NUM_RTYPES here
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const ReductionType rtype   = scal_reductions[i];

        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, rtype, v0, &candval);

        if (pid == 0) {
            const AcReal modelval = acHostReduceScal(model, rtype, v0);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);

            if (!acEvalError(rtype_names[rtype], error)) {
                fprintf(stderr, "Scalar %s: cand %g model %g\n", rtype_names[i], candval, modelval);
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
    }
    fflush(stdout);

    const ReductionType vec_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                            RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(vec_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        AcReal candval;

        const ReductionType rtype = vec_reductions[i];
        acGridReduceVec(STREAM_DEFAULT, rtype, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval = acHostReduceVec(model, rtype, v0, v1, v2);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);

            if (!acEvalError(rtype_names[rtype], error)) {
                fprintf(stderr, "Vector %s: cand %g model %g\n", rtype_names[i], candval, modelval);
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    if (pid == 0) {
        printf("---Test: Alfven reductions---\n");
    }
    fflush(stdout);

    const ReductionType alf_reductions[] = {RTYPE_ALFVEN_MAX, RTYPE_ALFVEN_MIN, RTYPE_ALFVEN_RMS};
    for (size_t i = 0; i < ARRAY_SIZE(alf_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        const VertexBufferHandle v3 = (VertexBufferHandle)3;
        AcReal candval;

        const ReductionType rtype = alf_reductions[i];
        acGridReduceVecScal(STREAM_DEFAULT, rtype, v0, v1, v2, v3, &candval);
        if (pid == 0) {
            const AcReal modelval = acHostReduceVecScal(model, rtype, v0, v1, v2, v3);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MAX, v0, v1, v2, v3);
            error.minimum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MIN, v0, v1, v1, v3);

            if (!acEvalError(rtype_names[rtype], error)) {
                fprintf(stderr, "Alfven %s: cand %g model %g\n", rtype_names[i], candval, modelval);
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

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "MPITEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return EXIT_SUCCESS;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
