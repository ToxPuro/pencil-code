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
    MPI_Init(NULL, NULL);
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acGridInit(info);

    AcTaskGraph* default_tasks = acGridGetDefaultTaskGraph();
    // std::string tasks_file_path = "tasks_pid_" + std::to_string(pid);
    // acGraphWriteTasks(tasks_file_path.c_str(), default_tasks);
    std::string dependencies_file_path = "dependencies_pid_" + std::to_string(pid);
    acGraphWriteDependencies(dependencies_file_path.c_str(), default_tasks);

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
