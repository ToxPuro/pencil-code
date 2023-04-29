#include "astaroth.h"
#include "astaroth_debug.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <mpi.h>
#include <string>

int
main(void)
{
    // Setup
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    srand(321654987);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }
    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, model);
    // End setup

    // Setup trace
    AcTaskGraph* default_tasks = acGridGetDefaultTaskGraph();

    std::string dependencies_file_path = "dependencies_pid_" + std::to_string(pid);
    acGraphWriteDependencies(dependencies_file_path.c_str(), default_tasks);

    // warm up the kernels and get all the auto-optimizations out of the way
    for (int i = 0; i < 100; i++) {
        acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    }

    std::string trace_file_path = "trace_pid_" + std::to_string(pid);
    acGraphEnableTrace(trace_file_path.c_str(), default_tasks);
    // End trace setup

    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);

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
