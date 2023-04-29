#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <iostream>
#include <mpi.h>

int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh mesh;
    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
        // Create randomized initial conditions
        acHostMeshRandomize(&mesh);
    }

    // Creates a global grid variable and default task graph
    acGridInit(info);

    std::cout << "Loading mesh" << std::endl;
    acGridLoadMesh(STREAM_DEFAULT, mesh);

    // Example: This does the same as acGridIntegrate()

    std::cout << "Initializing fields" << std::endl;
    // First we define what fields we're using.
    // This parameter is a c-style array but only works with c++ at the moment
    //(the interface relies on templates for safety and array type deduction).
    VertexBufferHandle all_fields[] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                       VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ}; //,  VTXBUF_ENTROPY};

    // Miikka's note: this would be a good quality of life feature
    // VertexBufferHandle all_fields = ALL_VERTEX_BUFFERS;
    // or
    // VertexBufferHandle all_fields = ALL_FIELDS;

    std::cout << "Generating graph" << std::endl;
    // Build a task graph consisting of:
    // - a halo exchange with periodic boundconds for all fields
    // - a calculation of the solve kernel touching all fields
    //
    // This function call generates tasks for each subregions in the domain
    // and figures out the dependencies between the tasks.
    AcTaskGraph* hc_graph = acGridBuildTaskGraph(
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
         acCompute(KERNEL_solve, all_fields)});

    // We can build multiple TaskGraphs, the MPI requests will not collide
    // because MPI tag space has been partitioned into ranges that each HaloExchange step uses.
    /*
    AcTaskGraph* shock_graph = acGridBuildTaskGraph({
        acHaloExchange(all_fields),
        acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, all_fields),
        acCompute(KERNEL_shock1, all_fields),
        acCompute(KERNEL_shock2, shock_fields),
        acCompute(KERNEL_solve, all_fields)
    });
    */

    std::cout << "Setting time delta" << std::endl;
    // Set the time delta
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON);
    acGridSynchronizeStream(STREAM_DEFAULT);

    std::cout << "Executing taskgraph Halo->Compute for 3 iterations" << std::endl;
    // Execute the task graph for three iterations.
    acGridExecuteTaskGraph(hc_graph, 3);

    // Execute the task graph for three iterations.
    // acGridExecuteTaskGraph(shock_graph, 3);
    // End example

    std::cout << "Destroying grid" << std::endl;
    acGridDestroyTaskGraph(hc_graph);
    // acGridDestroyTaskGraph(shock_graph);
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
