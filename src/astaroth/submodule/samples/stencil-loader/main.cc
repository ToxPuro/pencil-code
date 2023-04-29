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

#include "stencil_loader.h"

#include <mpi.h>
#include <vector>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);
    load_stencil_from_config(info);

    // Boundconds
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Boundconds", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        acHostMeshRandomize(&model);
    }

    // Integration
    const AcReal dt = (AcReal)FLT_EPSILON;
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostIntegrateStep(model, dt);
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        acHostMeshRandomize(&model);
    }

    // Scalar reductions
    acGridLoadMesh(STREAM_DEFAULT, model);

    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, (ReductionType)i, v0, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceScal(model, (ReductionType)i, v0);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        const VertexBufferHandle v1 = VTXBUF_UUY;
        const VertexBufferHandle v2 = VTXBUF_UUZ;
        AcReal candval;
        acGridReduceVec(STREAM_DEFAULT, (ReductionType)i, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval   = acHostReduceVec(model, (ReductionType)i, v0, v1, v2);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    MPI_Finalize();
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
