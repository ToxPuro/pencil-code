/*
    Copyright (C) 2020-2021, Johannes Pekkilä, Miikka Väisälä, Oskar Lappi

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
#if AC_MPI_ENABLED
/**
 * Quick overview of the MPI implementation:
 *
 * The halo is partitioned into segments, each segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * struct PackedData is used for packing and unpacking. Holds the actual data in
 *                   the halo partition (wrapped by HaloMessage)
 * struct Grid contains information about the local GPU device, decomposition,
 *             the total mesh dimensions, default tasks, and MPI requests
 * struct AcTaskGraph contains *Tasks*, encapsulated pieces of work that depend on each other.
 *                  Users can create their own AcTaskGraphs, but the struct implementation is
 *                  hidden from them.

 * Basic steps:
 *   1) Distribute the mesh among ranks
 *   2) Integrate & communicate
 *     - start inner integration and at the same time, pack halo data and send it to neighbors
 *     - as halo data is received and unpacked, integrate segments whose dependencies are ready
 *     - tasks in the task graph are run for three iterations. They are started early as possible
 *   3) Gather the mesh to rank 0 for postprocessing
 *
 * This file contains the grid interface, with algorithms and high level functionality
 * The nitty gritty of the MPI communication and the Task interface is defined in task.h/task.cc
 */

#include "astaroth.h"
#include "astaroth_utils.h"
#include "astaroth_debug.h"
#include "task.h"

#include <assert.h>
#include <algorithm>
#include <cstring> //memcpy
#include <mpi.h>
#include <queue>
#include <vector>
#include "decomposition.h" //getPid3D, morton3D
#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"

#ifdef USE_PERFSTUBS
#define PERFSTUBS_USE_TIMER
#include "perfstubs_api/timer.h"
#endif

/* Internal interface to grid (a global variable)  */
typedef struct Grid {
    Device device;
    AcMesh submesh; // Submesh in host memory. Used as scratch space.
    uint3_64 decomposition;
    bool initialized;
    int3 nn;
    std::shared_ptr<AcTaskGraph> default_tasks;
    size_t mpi_tag_space_count;
} Grid;

static Grid grid = {};

static constexpr int astaroth_comm_split_key = 666;

// In case some old programs still  use MPI_Init or MPI_Init_thread, we don't want to break them
static MPI_Comm astaroth_comm = MPI_COMM_WORLD;
static int3 domain_decomposition;
AcResult
ac_MPI_Init()
{
    if (MPI_Init(NULL, NULL)) {
        return AC_FAILURE;
    }

    // Get rank for new communicator
    int rank = -1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        return AC_FAILURE;
    }

    // Split MPI_COMM_WORLD
    if (MPI_Comm_split(MPI_COMM_WORLD, astaroth_comm_split_key, rank, &astaroth_comm) !=
        MPI_SUCCESS) {
        return AC_FAILURE;
    }
    return AC_SUCCESS;
}

AcResult
ac_MPI_Init_thread(int thread_level)
{
    int thread_support_level = -1;
    int result               = MPI_Init_thread(NULL, NULL, thread_level, &thread_support_level);
    if (thread_support_level < thread_level || result != MPI_SUCCESS) {
        fprintf(stderr, "Thread level %d not supported by the MPI implementation\n", thread_level);
        return AC_FAILURE;
    }

    // Get rank for new communicator
    int rank = -1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        return AC_FAILURE;
    }

    // Split MPI_COMM_WORLD
    if (MPI_Comm_split(MPI_COMM_WORLD, astaroth_comm_split_key, rank, &astaroth_comm) !=
        MPI_SUCCESS) {
        return AC_FAILURE;
    }
    return AC_SUCCESS;
}

void
ac_MPI_Finalize()
{
    if (astaroth_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&astaroth_comm);
        astaroth_comm = MPI_COMM_NULL;
    }
    MPI_Finalize();
}

MPI_Comm
acGridMPIComm()
{
    return astaroth_comm;
}

AcResult
acGridSynchronizeStream(const Stream stream)
{
    ERRCHK(grid.initialized);
    acDeviceSynchronizeStream(grid.device, stream);
    MPI_Barrier(astaroth_comm);
    return AC_SUCCESS;
}

AcResult
acGridRandomize(void)
{
    ERRCHK(grid.initialized);

    const Stream stream = STREAM_DEFAULT;

    AcMesh host;
    acHostMeshCreate(grid.submesh.info, &host);
    acHostMeshRandomize(&host);
    acDeviceLoadMesh(grid.device, stream, host);
    acDeviceSynchronizeStream(grid.device, stream);
    acHostMeshDestroy(&host);

    acGridPeriodicBoundconds(stream);
    acGridSynchronizeStream(stream);

    return AC_SUCCESS;
}

Device
acGridGetDevice()
{
    return grid.device;
}

AcMeshInfo
acGridDecomposeMeshInfo(const AcMeshInfo global_config)
{
    AcMeshInfo submesh_config = global_config;

    int nprocs, pid;
    MPI_Comm_size(astaroth_comm, &nprocs);
    MPI_Comm_rank(astaroth_comm, &pid);

    // const uint3_64 decomp = decompose(nprocs);
    const uint3_64 decomp = (uint3_64){domain_decomposition.x,domain_decomposition.y,domain_decomposition.z};
    const int3 pid3d      = getPid3D(pid, decomp);

    ERRCHK_ALWAYS(submesh_config.int_params[AC_nx] % decomp.x == 0);
    ERRCHK_ALWAYS(submesh_config.int_params[AC_ny] % decomp.y == 0);
    ERRCHK_ALWAYS(submesh_config.int_params[AC_nz] % decomp.z == 0);

    const int submesh_nx = submesh_config.int_params[AC_nx] / decomp.x;
    const int submesh_ny = submesh_config.int_params[AC_ny] / decomp.y;
    const int submesh_nz = submesh_config.int_params[AC_nz] / decomp.z;

    submesh_config.int_params[AC_nx]               = submesh_nx;
    submesh_config.int_params[AC_ny]               = submesh_ny;
    submesh_config.int_params[AC_nz]               = submesh_nz;
    submesh_config.int3_params[AC_global_grid_n]   = (int3){global_config.int_params[AC_nx],
                                                            global_config.int_params[AC_ny],
                                                            global_config.int_params[AC_nz]};
    submesh_config.int3_params[AC_multigpu_offset] = pid3d *
                                                     (int3){submesh_nx, submesh_ny, submesh_nz};
    int3 gpu_offset = pid3d * (int3){submesh_nx, submesh_ny, submesh_nz};
    printf("GPU offset:pid:%d: %d,%d,%d", pid, gpu_offset.x, gpu_offset.y, gpu_offset.z);
    acHostUpdateBuiltinParams(&submesh_config);
    return submesh_config;
}
AcResult
acGridSetDomainDecomposition(const int3 decomposition){
    domain_decomposition = decomposition;
}
AcResult
acGridGetVBAPointers(AcReal *vbapointer[2]){
    return acDeviceGetVBApointers(grid.device, vbapointer);
}
AcResult
acGridInit(const AcMeshInfo info)
{
    ERRCHK(!grid.initialized);

    // Check that MPI is initialized
    int nprocs, pid;
    MPI_Comm_size(astaroth_comm, &nprocs);
    MPI_Comm_rank(astaroth_comm, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Check that device allocation is valid
    int device_count = -1;
    cudaGetDeviceCount(&device_count);
    if (device_count > nprocs) {
        fprintf(stderr,
                "Invalid device-task allocation: Must allocate one MPI task per GPU but got %d "
                "devices per node and only %d task(s).",
                device_count, nprocs);
        ERRCHK_ALWAYS(device_count <= nprocs);
    }
    MPI_Barrier(acGridMPIComm());

    // Decompose
    
    //const uint3_64 decomp = decompose(nprocs);
    const uint3_64 decomp = (uint3_64){domain_decomposition.x,domain_decomposition.y,domain_decomposition.z};
    printf("Decomposition: %d,%d,%d\n", decomp.x,decomp.y,decomp.z);

    // Check that the decomposition is valid
    const int3 nn       = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const bool nx_valid = nn.x % decomp.x == 0;
    const bool ny_valid = nn.y % decomp.y == 0;
    const bool nz_valid = nn.z % decomp.z == 0;
    if (!nx_valid || !ny_valid || !nz_valid) {
        WARNING("Mesh dimensions must be divisible by the decomposition\n");
        fprintf(stderr, "Decomposition: (%lu, %lu, %lu)\n", decomp.x, decomp.y, decomp.z);
        fprintf(stderr, "Mesh dimensions: (%d, %d, %d)\n", nn.x, nn.y, nn.z);
        fprintf(stderr, "Divisible: (%d, %d, %d)\n", nx_valid, ny_valid, nz_valid);
    }
    if (nn.x < STENCIL_WIDTH)
        fprintf(stderr, "nn.x %d too small, must be >= %d (stencil width)\n", nn.x, STENCIL_WIDTH);
    if (nn.y < STENCIL_HEIGHT)
        fprintf(stderr, "nn.y %d too small, must be >= %d (stencil height)\n", nn.y,
                STENCIL_HEIGHT);
    if (nn.z < STENCIL_DEPTH)
        fprintf(stderr, "nn.z %d too small, must be >= %d (stencil depth)\n", nn.z, STENCIL_DEPTH);

    MPI_Barrier(astaroth_comm);

#if AC_VERBOSE
    const int3 pid3d = getPid3D(pid, decomp);
    printf("Processor %s. Process %d of %d: (%d, %d, %d)\n", processor_name, pid, nprocs, pid3d.x,
           pid3d.y, pid3d.z);
    printf("Decomposition: %lu, %lu, %lu\n", decomp.x, decomp.y, decomp.z);
    printf("Mesh size: %d, %d, %d\n", info.int_params[AC_nx], info.int_params[AC_ny],
           info.int_params[AC_nz]);
    fflush(stdout);
    MPI_Barrier(astaroth_comm);
#endif

    // Check that mixed precision is correctly configured, AcRealPacked == AC_REAL_MPI_TYPE
    // CAN BE REMOVED IF MIXED PRECISION IS SUPPORTED AS A PREPROCESSOR FLAG
    int mpi_type_size;
    MPI_Type_size(AC_REAL_MPI_TYPE, &mpi_type_size);
    ERRCHK_ALWAYS(sizeof(AcRealPacked) == mpi_type_size);

    // Decompose config (divide dimensions by decomposition)
    AcMeshInfo submesh_info = acGridDecomposeMeshInfo(info);

    // GPU alloc
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    acLogFromRootProc(pid, "acGridInit: Calling acDeviceCreate\n");
    Device device;
    printf("Device creation: %d,%d\n", pid, pid % devices_per_node);
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
    acLogFromRootProc(pid, "acGridInit: Returned from acDeviceCreate\n");

    // CPU alloc
    acLogFromRootProc(pid, "acGridInit: Allocating CPU mesh\n");
    AcMesh submesh;
    acHostMeshCreate(submesh_info, &submesh);
    acLogFromRootProc(pid, "acGridInit: Done allocating CPU mesh\n");

    // Setup the global grid structure
    grid.device        = device;
    grid.submesh       = submesh;
    grid.decomposition = decomp;

    // Configure
    grid.nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };
    printf("Grid nn:%d: %d,%d,%d\n", pid,grid.nn.x,grid.nn.y,grid.nn.z);

    grid.mpi_tag_space_count = 0;

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }

    acLogFromRootProc(pid, "acGridInit: Creating default task graph\n");
    AcTaskDefinition default_ops[] = {
        acHaloExchange(all_fields),
        acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,all_fields),
        acCompute(KERNEL_singlepass_solve, all_fields, 0, 0),
        
        acHaloExchange(all_fields),
        acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,all_fields),
        acCompute(KERNEL_singlepass_solve, all_fields, 1, 1),
        
        acHaloExchange(all_fields),
        acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,all_fields),
        acCompute(KERNEL_singlepass_solve, all_fields, 2, 2)

    };
//     AcTaskDefinition default_ops[] = {acHaloExchange(all_fields),
//                                       acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,
//                                                           all_fields),
// #ifdef AC_INTEGRATION_ENABLED
// #ifdef AC_SINGLEPASS_INTEGRATION
//                                       acCompute(KERNEL_singlepass_solve, all_fields, 0, 0)
// #else
//                                       acCompute(KERNEL_twopass_solve_intermediate, all_fields, 0),
//                                       acCompute(KERNEL_twopass_solve_final, all_fields, 1)
// #endif
// #endif // AC_INTEGRATION_ENABLED
//     };

    // Random number generator
    // const auto rr            = (int3){STENCIL_WIDTH, STENCIL_HEIGHT, STENCIL_DEPTH};
    // const auto local_m       = acConstructInt3Param(AC_mx, AC_my, AC_mz, submesh_info);
    // const auto global_m      = submesh_info.int3_params[AC_global_grid_n] + 2 * rr;
    // const auto global_offset = submesh_info.int3_params[AC_multigpu_offset];
    // acRandInit(1234UL, to_volume(local_m), to_volume(global_m), to_volume(global_offset));
    const Volume local_m = to_volume(acConstructInt3Param(AC_mx, AC_my, AC_mz, submesh_info));
    printf("Grid mm:%d: %d,%d,%d\n", pid,local_m.x,local_m.y,local_m.z);
    const size_t count   = local_m.x * local_m.y * local_m.z;
    acRandInitAlt(1234UL, count, pid);

    grid.initialized   = true;
    grid.default_tasks = std::shared_ptr<AcTaskGraph>(acGridBuildTaskGraph(default_ops));
    acLogFromRootProc(pid, "acGridInit: Done creating default task graph\n");

    acVerboseLogFromRootProc(pid, "acGridInit: Synchronizing streams\n");
    acGridSynchronizeStream(STREAM_ALL);
    acVerboseLogFromRootProc(pid, "acGridInit: Done synchronizing streams\n");
    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    // Random number generator
    acRandQuit();

    grid.default_tasks = nullptr;

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    acHostMeshDestroy(&grid.submesh);
    acDeviceDestroy(grid.device);

    return AC_SUCCESS;
}

AcResult
acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal buffer       = value;
    MPI_Bcast(&buffer, 1, AC_REAL_MPI_TYPE, root_proc, astaroth_comm);

    return acDeviceLoadScalarUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadVectorUniform(const Stream stream, const AcReal3Param param, const AcReal3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal3 buffer      = value;
    MPI_Bcast(&buffer, 3, AC_REAL_MPI_TYPE, root_proc, astaroth_comm);

    return acDeviceLoadVectorUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadIntUniform(const Stream stream, const AcIntParam param, const int value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    int buffer          = value;
    MPI_Bcast(&buffer, 1, MPI_INT, root_proc, astaroth_comm);

    return acDeviceLoadIntUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadInt3Uniform(const Stream stream, const AcInt3Param param, const int3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    int3 buffer         = value;
    MPI_Bcast(&buffer, 3, MPI_INT, root_proc, astaroth_comm);

    return acDeviceLoadInt3Uniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadMeshWorking(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 monolithic_mm     = info.int3_params[AC_global_grid_n] + 2 * rr;
    const int3 monolithic_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 monolithic_offset = rr;

    MPI_Datatype monolithic_subarray;
    const int monolithic_mm_arr[]     = {monolithic_mm.z, monolithic_mm.y, monolithic_mm.x};
    const int monolithic_nn_arr[]     = {monolithic_nn.z, monolithic_nn.y, monolithic_nn.x};
    const int monolithic_offset_arr[] = {monolithic_offset.z, monolithic_offset.y,
                                         monolithic_offset.x};
    MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
    MPI_Type_commit(&monolithic_subarray);

    const int3 distributed_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    const int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 distributed_offset = rr;

    MPI_Datatype distributed_subarray;
    const int distributed_mm_arr[]     = {distributed_mm.z, distributed_mm.y, distributed_mm.x};
    const int distributed_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
    const int distributed_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                          distributed_offset.x};
    MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
    MPI_Type_commit(&distributed_subarray);

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    MPI_Request recv_reqs[NUM_VTXBUF_HANDLES];
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        MPI_Irecv(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &recv_reqs[vtxbuf]);
        if (pid == 0) {
            for (int tgt = 0; tgt < nprocs; ++tgt) {
                const int3 tgt_pid3d = getPid3D(tgt, grid.decomposition);
                const size_t idx     = acVertexBufferIdx(tgt_pid3d.x * distributed_nn.x, //
                                                         tgt_pid3d.y * distributed_nn.y, //
                                                         tgt_pid3d.z * distributed_nn.z, //
                                                         host_mesh.info);
                MPI_Send(&host_mesh.vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, tgt, vtxbuf,
                         acGridMPIComm());
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, recv_reqs, MPI_STATUSES_IGNORE);
    /*
        Strategy:
            1) Select a subarray from the input mesh
            2) Select a subarray from the output mesh
            3) Scatter

        Notes:
            1) Check that subarray divisible by number of procs (required in init iirc)
    MPI_Datatype input_subarray_resized;
    MPI_Type_create_resized(input_subarray, 0, sizeof(AcReal), &input_subarray_resized);
    MPI_Type_commit(&input_subarray_resized);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = host_mesh.vertex_buffer[vtxbuf];
        AcReal* dst       = grid.submesh.vertex_buffer[vtxbuf];
        //MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());

        int nprocs;
        MPI_Comm_size(acGridMPIComm(), &nprocs);
        const uint3_64 p = morton3D(nprocs - 1) + (uint3_64){1, 1, 1};
        int counts[nprocs];
        int displacements[nprocs];
        for (int i = 0; i < nprocs; ++i) {
            counts[i]    = 1;

            const uint3_64 block = morton3D(i);
            const size_t block_offset = block.x * output_nn.x + block.y * output_nn.y * output_nn.x
    * p.x + block.z * output_nn.z * output_nn.x * output_nn.y; displacements[i] = block_offset;
        }

        //MPI_Scatterv(src, counts, displacements, input_subarray, dst, 1, output_subarray, 0,
        //             acGridMPIComm());
        MPI_Scatterv(src, counts, displacements, input_subarray_resized, dst, output_nn.z *
    output_nn.y * output_nn.x, AC_REAL_MPI_TYPE, 0, acGridMPIComm());

    }*/

    MPI_Type_free(&monolithic_subarray);
    MPI_Type_free(&distributed_subarray);
    return acDeviceLoadMesh(grid.device, stream, grid.submesh);
}

/*
// has some illegal memory access issue (create_subarray overwrites block value and loop fails)
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(device, stream, &grid.submesh);
    acDeviceSynchronizeStream(device, stream);

    int pid, nprocs;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    MPI_Comm_size(acGridMPIComm(), &nprocs);

    const int3 pid3d   = getPid3D(pid, grid.decomposition);
    const uint3_64 min = (uint3_64){0, 0, 0};
    const uint3_64 max = morton3D(nprocs - 1); // inclusive

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 monolithic_mm  = info.int3_params[AC_global_grid_n] + 2 * rr;
    const int3 distributed_mm = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    for (int block = 0; block < nprocs; ++block) {
        ERRCHK_ALWAYS(block < nprocs);
        int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        int3 distributed_offset = rr;
        ERRCHK_ALWAYS(block < nprocs);
        if (pid3d.x == min.x) {
            distributed_nn.x += rr.x;
            distributed_offset.x = 0;
        }
        if (pid3d.x == max.x) {
            distributed_nn.x += rr.x;
        }
        if (pid3d.y == min.y) {
            distributed_nn.y += rr.y;
            distributed_offset.y = 0;
        }
        if (pid3d.y == max.y) {
            distributed_nn.y += rr.y;
        }
        if (pid3d.z == min.z) {
            distributed_nn.z += rr.z;
            distributed_offset.z = 0;
        }
        if (pid3d.z == max.z) {
            distributed_nn.z += rr.z;
        }
        // fprintf(stderr, "proc %d, pid %d %d %d, box size %d %d %d\n", pid, pid3d.x, pid3d.y,
        // pid3d.z,
        //         distributed_nn.x, distributed_nn.y, distributed_nn.z);
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Datatype monolithic_subarray;
        const int monolithic_mm_arr[]     = {monolithic_mm.z, monolithic_mm.y, monolithic_mm.x};
        const int monolithic_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
        const int monolithic_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                             distributed_offset.x};
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                                 MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Type_commit(&monolithic_subarray);
        ERRCHK_ALWAYS(block < nprocs);

        // const int3 distributed_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        // const int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 distributed_offset = rr;
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Datatype distributed_subarray;
        const int distributed_mm_arr[]     = {distributed_mm.z, distributed_mm.y, distributed_mm.x};
        const int distributed_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
        const int distributed_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                             distributed_offset.x};
        MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                                 MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
        MPI_Type_commit(&distributed_subarray);

        ERRCHK_ALWAYS(block < nprocs);
        MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            ERRCHK_ALWAYS(block < nprocs);
            // send to 0
            if (pid == block)
                MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                          acGridMPIComm(), &send_reqs[vtxbuf]);
            ERRCHK_ALWAYS(block < nprocs);
            if (pid == 0) {
                // recv from block
                ERRCHK_ALWAYS(block < nprocs);
                const int3 block_pid3d = getPid3D(block, grid.decomposition);
                const int3 nn          = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
                const size_t idx       = acVertexBufferIdx(block_pid3d.x * nn.x, //
                                                           block_pid3d.y * nn.y, //
                                                           block_pid3d.z * nn.z, //
                                                           host_mesh->info);
                MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, block,
                         vtxbuf, acGridMPIComm(), MPI_STATUS_IGNORE);
            }
        }
        if (pid == block)
            MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);
        // Free
        MPI_Type_free(&monolithic_subarray);
        MPI_Type_free(&distributed_subarray);
        // TODO
        // for each block
        //      declare the mapping
        //      all send
        //      if pid == 0
        //          recv
        //
        // could possibly do with scatter/gather but not that important and
        // diminishing returns/no-point finetuning because this is just a
        // simple debug function anyways.
        // More important to focus on getting science/meaningful work done!
    }
}
*/

static void
to_mpi_array_order_c(const int3 v, int arr[3])
{
    arr[0] = v.z;
    arr[1] = v.y;
    arr[2] = v.x;
}

/*
static void
print_mpi_array(const char* str, const int arr[3])
{
    printf("%s: (%d, %d, %d)\n", str, arr[2], arr[1], arr[0]);
}
*/

static void
get_subarray(const int pid, //
             int monolithic_mm_arr[3], int monolithic_nn_arr[3],
             int monolithic_offset_arr[3], //
             int distributed_mm_arr[3], int distributed_nn_arr[3], int distributed_offset_arr[3])
{
    int nprocs;
    MPI_Comm_size(acGridMPIComm(), &nprocs);

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    const int3 pid3d = getPid3D(pid, grid.decomposition);
    const int3 rr    = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };

    const int3 min = (int3){0, 0, 0};
    const int3 max = getPid3D(nprocs - 1, grid.decomposition); // inclusive

    const int3 base_distributed_nn = acConstructInt3Param(AC_nx, AC_ny, AC_nz,
                                                          device->local_config);
    int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, device->local_config);
    int3 distributed_offset = rr;

    if (pid3d.x == min.x) {
        distributed_offset.x -= rr.x;
        distributed_nn.x += rr.x;
    }
    if (pid3d.x == max.x) {
        distributed_nn.x += rr.x;
    }
    if (pid3d.y == min.y) {
        distributed_offset.y -= rr.y;
        distributed_nn.y += rr.y;
    }
    if (pid3d.y == max.y) {
        distributed_nn.y += rr.y;
    }
    if (pid3d.z == min.z) {
        distributed_offset.z -= rr.z;
        distributed_nn.z += rr.z;
    }
    if (pid3d.z == max.z) {
        distributed_nn.z += rr.z;
    }

    // Monolithic
    to_mpi_array_order_c(info.int3_params[AC_global_grid_n] + 2 * rr, monolithic_mm_arr);
    to_mpi_array_order_c(distributed_nn, monolithic_nn_arr);
    to_mpi_array_order_c(pid3d * base_distributed_nn + distributed_offset, monolithic_offset_arr);

    // Distributed
    to_mpi_array_order_c(acConstructInt3Param(AC_mx, AC_my, AC_mz, info), distributed_mm_arr);
    to_mpi_array_order_c(distributed_nn, distributed_nn_arr);
    to_mpi_array_order_c(distributed_offset, distributed_offset_arr);

    /*
    printf("------\n");
    printf("pid %d\n", pid);
    print_mpi_array("monol mm", monolithic_mm_arr);
    print_mpi_array("monol nn", monolithic_nn_arr);
    print_mpi_array("monol os", monolithic_offset_arr);

    print_mpi_array("distr mm", distributed_mm_arr);
    print_mpi_array("distr nn", distributed_nn_arr);
    print_mpi_array("distr os", distributed_offset_arr);
    printf("------\n");
    */
}

// With ghost zone
AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    int pid, nprocs;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    MPI_Comm_size(acGridMPIComm(), &nprocs);

    // Datatype:
    // 1) All processes: Local subarray (sending)
    //  1.1) function that takes the pid and outputs the local subarray
    // 2) Root process:  Global array (receiving)
    // 3) Root process:  Local subarrays for all procs (same as used for sending)

    // Receive the local subarray
    MPI_Request recv_reqs[NUM_VTXBUF_HANDLES];
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
        int distributed_mm[3], distributed_nn[3], distributed_offset[3];
        get_subarray(pid, monolithic_mm, monolithic_nn, monolithic_offset, //
                     distributed_mm, distributed_nn, distributed_offset);

        MPI_Datatype distributed_subarray;
        MPI_Type_create_subarray(3, distributed_mm, distributed_nn, distributed_offset, MPI_ORDER_C,
                                 AC_REAL_MPI_TYPE, &distributed_subarray);
        MPI_Type_commit(&distributed_subarray);

        MPI_Irecv(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &recv_reqs[vtxbuf]);

        MPI_Type_free(&distributed_subarray);
    }

    if (pid == 0) {
        for (int tgt = 0; tgt < nprocs; ++tgt) {
            for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
                int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
                int distributed_mm[3], distributed_nn[3], distributed_offset[3];
                get_subarray(tgt, monolithic_mm, monolithic_nn, monolithic_offset, //
                             distributed_mm, distributed_nn, distributed_offset);

                MPI_Datatype monolithic_subarray;
                MPI_Type_create_subarray(3, monolithic_mm, monolithic_nn, monolithic_offset,
                                         MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
                MPI_Type_commit(&monolithic_subarray);

                MPI_Send(host_mesh.vertex_buffer[vtxbuf], 1, monolithic_subarray, tgt, vtxbuf,
                         acGridMPIComm());

                MPI_Type_free(&monolithic_subarray);
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, recv_reqs, MPI_STATUSES_IGNORE);

    // TODO: Should apply halo exchange here without touching the ghost zones, how?
    // Currently the users need to update halos after each load, which is error-prone
    // acDeviceLoadMesh(grid.device, stream, grid.submesh);
    // return acGridPeriodicBoundconds(STREAM_DEFAULT);
    int mx = grid.nn.x+6;
    int my = grid.nn.y+6;
    int mz = grid.nn.z+6;
    printf("CPU full check before Astaroth uux :%d: %f\n",pid, grid.submesh.vertex_buffer[VTXBUF_UUX][9+mx*9+my*mx*9]);
    printf("CPU full check before Astaroth uuy :%d: %f\n",pid, grid.submesh.vertex_buffer[VTXBUF_UUY][9+mx*9+my*mx*9]);
    printf("CPU full check before Astaroth uuz :%d: %f\n",pid, grid.submesh.vertex_buffer[VTXBUF_UUZ][9+mx*9+my*mx*9]);

    printf(">>>>>>>>>>>>>>>>>>>>>>>>Host mesh<<<<<<<<<<<<<<<<<<<<");
    printf("CPU full check before Astaroth uux :%d: %f\n",pid, host_mesh.vertex_buffer[VTXBUF_UUX][9+mx*9+my*mx*9]);
    printf("CPU full check before Astaroth uuy :%d: %f\n",pid, host_mesh.vertex_buffer[VTXBUF_UUY][9+mx*9+my*mx*9]);
    printf("CPU full check before Astaroth uuz :%d: %f\n",pid, host_mesh.vertex_buffer[VTXBUF_UUZ][9+mx*9+my*mx*9]);
    return acDeviceLoadMesh(grid.device, stream, host_mesh);
}

// Working with ghost zone
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(grid.device, stream, host_mesh);
    acDeviceSynchronizeStream(grid.device, stream);

    // int pid, nprocs;
    // MPI_Comm_rank(acGridMPIComm(), &pid);
    // MPI_Comm_size(acGridMPIComm(), &nprocs);

    // // Datatype:
    // // 1) All processes: Local subarray (sending)
    // //  1.1) function that takes the pid and outputs the local subarray
    // // 2) Root process:  Global array (receiving)
    // // 3) Root process:  Local subarrays for all procs (same as used for sending)

    // // Send the local subarray
    // MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
    // for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    //     int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
    //     int distributed_mm[3], distributed_nn[3], distributed_offset[3];
    //     get_subarray(pid, monolithic_mm, monolithic_nn, monolithic_offset, //
    //                  distributed_mm, distributed_nn, distributed_offset);

    //     MPI_Datatype distributed_subarray;
    //     MPI_Type_create_subarray(3, distributed_mm, distributed_nn, distributed_offset, MPI_ORDER_C,
    //                              AC_REAL_MPI_TYPE, &distributed_subarray);
    //     MPI_Type_commit(&distributed_subarray);

    //     MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
    //               acGridMPIComm(), &send_reqs[vtxbuf]);

    //     MPI_Type_free(&distributed_subarray);
    // }

    // if (pid == 0) {
    //     for (int src = 0; src < nprocs; ++src) {
    //         for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    //             int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
    //             int distributed_mm[3], distributed_nn[3], distributed_offset[3];
    //             get_subarray(src, monolithic_mm, monolithic_nn, monolithic_offset, //
    //                          distributed_mm, distributed_nn, distributed_offset);

    //             MPI_Datatype monolithic_subarray;
    //             MPI_Type_create_subarray(3, monolithic_mm, monolithic_nn, monolithic_offset,
    //                                      MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
    //             MPI_Type_commit(&monolithic_subarray);

    //             MPI_Recv(host_mesh->vertex_buffer[vtxbuf], 1, monolithic_subarray, src, vtxbuf,
    //                      acGridMPIComm(), MPI_STATUS_IGNORE);

    //             MPI_Type_free(&monolithic_subarray);
    //         }
    //     }
    // }
    // MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);

    return AC_SUCCESS;
}

AcResult
acGridStoreMeshWorking(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acDeviceSynchronizeStream(grid.device, stream);

    /*
    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 input_nn     = info.int3_params[AC_global_grid_n]; // Without halo
    const int3 input_mm     = input_nn + 2 * rr;
    const int3 input_offset = rr; //  + info.int3_params[AC_multigpu_offset];

    MPI_Datatype input_subarray;
    const int input_mm_arr[]     = {input_mm.z, input_mm.y, input_mm.x};
    const int input_nn_arr[]     = {input_nn.z, input_nn.y, input_nn.x};
    const int input_offset_arr[] = {input_offset.z, input_offset.y, input_offset.x};
    MPI_Type_create_subarray(3, input_mm_arr, input_nn_arr, input_offset_arr, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &input_subarray);
    MPI_Type_commit(&input_subarray);

    const int3 output_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 output_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    const int3 output_offset = rr;

    MPI_Datatype output_subarray;
    const int output_mm_arr[]     = {output_mm.z, output_mm.y, output_mm.x};
    const int output_nn_arr[]     = {output_nn.z, output_nn.y, output_nn.x};
    const int output_offset_arr[] = {output_offset.z, output_offset.y, output_offset.x};
    MPI_Type_create_subarray(3, output_mm_arr, output_nn_arr, output_offset_arr, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &output_subarray);
    MPI_Type_commit(&output_subarray);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = grid.submesh.vertex_buffer[vtxbuf];
        AcReal* dst       = host_mesh->vertex_buffer[vtxbuf];
        MPI_Gather(src, 1, output_subarray, dst, 1, input_subarray, 0, acGridMPIComm());
        // MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());
    }

    MPI_Type_free(&input_subarray);
    MPI_Type_free(&output_subarray);
    */

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 monolithic_mm     = info.int3_params[AC_global_grid_n] + 2 * rr;
    const int3 monolithic_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 monolithic_offset = rr;

    MPI_Datatype monolithic_subarray;
    const int monolithic_mm_arr[]     = {monolithic_mm.z, monolithic_mm.y, monolithic_mm.x};
    const int monolithic_nn_arr[]     = {monolithic_nn.z, monolithic_nn.y, monolithic_nn.x};
    const int monolithic_offset_arr[] = {monolithic_offset.z, monolithic_offset.y,
                                         monolithic_offset.x};
    MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
    MPI_Type_commit(&monolithic_subarray);

    const int3 distributed_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    const int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 distributed_offset = rr;

    MPI_Datatype distributed_subarray;
    const int distributed_mm_arr[]     = {distributed_mm.z, distributed_mm.y, distributed_mm.x};
    const int distributed_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
    const int distributed_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                          distributed_offset.x};
    MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
    MPI_Type_commit(&distributed_subarray);

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &send_reqs[vtxbuf]);
        if (pid == 0) {
            for (int tgt = 0; tgt < nprocs; ++tgt) {
                const int3 tgt_pid3d = getPid3D(tgt, grid.decomposition);
                const size_t idx     = acVertexBufferIdx(tgt_pid3d.x * distributed_nn.x, //
                                                         tgt_pid3d.y * distributed_nn.y, //
                                                         tgt_pid3d.z * distributed_nn.z, //
                                                         host_mesh->info);
                MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, tgt,
                         vtxbuf, acGridMPIComm(), MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);
    /*
        Strategy:
            1) Select a subarray from the input mesh
            2) Select a subarray from the output mesh
            3) Scatter

        Notes:
            1) Check that subarray divisible by number of procs (required in init iirc)
    MPI_Datatype input_subarray_resized;
    MPI_Type_create_resized(input_subarray, 0, sizeof(AcReal), &input_subarray_resized);
    MPI_Type_commit(&input_subarray_resized);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = host_mesh.vertex_buffer[vtxbuf];
        AcReal* dst       = grid.submesh.vertex_buffer[vtxbuf];
        //MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());

        int nprocs;
        MPI_Comm_size(acGridMPIComm(), &nprocs);
        const uint3_64 p = morton3D(nprocs - 1) + (uint3_64){1, 1, 1};
        int counts[nprocs];
        int displacements[nprocs];
        for (int i = 0; i < nprocs; ++i) {
            counts[i]    = 1;

            const uint3_64 block = morton3D(i);
            const size_t block_offset = block.x * output_nn.x + block.y * output_nn.y * output_nn.x
    * p.x + block.z * output_nn.z * output_nn.x * output_nn.y; displacements[i] = block_offset;
        }

        //MPI_Scatterv(src, counts, displacements, input_subarray, dst, 1, output_subarray, 0,
        //             acGridMPIComm());
        MPI_Scatterv(src, counts, displacements, input_subarray_resized, dst, output_nn.z *
    output_nn.y * output_nn.x, AC_REAL_MPI_TYPE, 0, acGridMPIComm());

    }*/

    MPI_Type_free(&monolithic_subarray);
    MPI_Type_free(&distributed_subarray);

    return AC_SUCCESS;
}

AcResult
acGridLoadMeshOld(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acGridDiskAccessSync(); // Note: syncs all streams

#if AC_VERBOSE
    printf("Distributing mesh...\n");
    fflush(stdout);
#endif

    int pid, nprocs;
    MPI_Comm_rank(astaroth_comm, &pid);
    MPI_Comm_size(astaroth_comm, &nprocs);

    ERRCHK_ALWAYS(&grid.submesh);

    // Submesh nn
    const int3 nn = (int3){
        grid.submesh.info.int_params[AC_nx],
        grid.submesh.info.int_params[AC_ny],
        grid.submesh.info.int_params[AC_nz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
                for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                    const int i       = NGHOST;
                    const int count   = nn.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, host_mesh.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    memcpy(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], //
                           &host_mesh.vertex_buffer[vtxbuf][src_idx],    //
                           count * sizeof(host_mesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
            for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                const int i     = NGHOST;
                const int count = nn.x;

                if (pid != 0) {
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    // Recv
                    MPI_Status status;
                    MPI_Recv(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], count, AC_REAL_MPI_TYPE,
                             0, 0, astaroth_comm, &status);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, grid.decomposition);
                        const int src_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                                 j + tgt_pid3d.y * nn.y, //
                                                                 k + tgt_pid3d.z * nn.z, //
                                                                 host_mesh.info);

                        // Send
                        MPI_Send(&host_mesh.vertex_buffer[vtxbuf][src_idx], count, AC_REAL_MPI_TYPE,
                                 tgt_pid, 0, astaroth_comm);
                    }
                }
            }
        }
    }

    acDeviceLoadMesh(grid.device, stream, grid.submesh);
    return AC_SUCCESS;
}

// TODO: do with packed data
AcResult
acGridStoreMeshAA(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acGridDiskAccessSync(); // Note: syncs all streams

    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acGridSynchronizeStream(stream);

#if AC_VERBOSE
    printf("Gathering mesh...\n");
    fflush(stdout);
#endif

    int pid, nprocs;
    MPI_Comm_rank(astaroth_comm, &pid);
    MPI_Comm_size(astaroth_comm, &nprocs);

    if (pid == 0)
        ERRCHK_ALWAYS(host_mesh);

    // Submesh nn
    const int3 nn = (int3){
        grid.submesh.info.int_params[AC_nx],
        grid.submesh.info.int_params[AC_ny],
        grid.submesh.info.int_params[AC_nz],
    };

    // Submesh mm
    const int3 mm = (int3){
        grid.submesh.info.int_params[AC_mx],
        grid.submesh.info.int_params[AC_my],
        grid.submesh.info.int_params[AC_mz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = 0; k < mm.z; ++k) {
                for (int j = 0; j < mm.y; ++j) {
                    const int i       = 0;
                    const int count   = mm.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, host_mesh->info);
                    memcpy(&host_mesh->vertex_buffer[vtxbuf][dst_idx],   //
                           &grid.submesh.vertex_buffer[vtxbuf][src_idx], //
                           count * sizeof(grid.submesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = 0; k < mm.z; ++k) {
            for (int j = 0; j < mm.y; ++j) {
                const int i     = 0;
                const int count = mm.x;

                if (pid == 0) {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, grid.decomposition);
                        const int dst_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                                 j + tgt_pid3d.y * nn.y, //
                                                                 k + tgt_pid3d.z * nn.z, //
                                                                 host_mesh->info);

                        // Recv
                        MPI_Status status;
                        MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][dst_idx], count,
                                 AC_REAL_MPI_TYPE, tgt_pid, 0, astaroth_comm, &status);
                    }
                }
                else {
                    // Send
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    MPI_Send(&grid.submesh.vertex_buffer[vtxbuf][src_idx], count, AC_REAL_MPI_TYPE,
                             0, 0, astaroth_comm);
                }
            }
        }
    }
    MPI_Barrier(astaroth_comm);

    return AC_SUCCESS;
}

AcTaskGraph*
acGridGetDefaultTaskGraph()
{
    ERRCHK(grid.initialized);
    return grid.default_tasks.get();
}

static void
check_ops(const AcTaskDefinition ops[], const size_t n_ops)
{
    if (n_ops == 0) {
        ERROR("\nIncorrect task graph {}:\n - Task graph is empty.\n")
    }

    bool found_halo_exchange        = false;
    unsigned int boundaries_defined = 0x00;
    // bool found_compute              = false;

    bool boundary_condition_before_halo_exchange = false;
    bool compute_before_halo_exchange            = false;
    bool compute_before_boundary_condition       = false;

    bool error   = false;
    bool warning = false;

    std::string task_graph_repr = "{";

    for (size_t i = 0; i < n_ops; i++) {
        AcTaskDefinition op = ops[i];
        switch (op.task_type) {
        case TASKTYPE_HALOEXCHANGE:
            found_halo_exchange = true;
            task_graph_repr += "HaloExchange,";
            break;
        case TASKTYPE_BOUNDCOND:
        case TASKTYPE_SPECIAL_MHD_BOUNDCOND:
            if (!found_halo_exchange) {
                boundary_condition_before_halo_exchange = true;
                error                                   = true;
            }
            boundaries_defined |= (unsigned int)op.boundary;
            task_graph_repr += "BoundCond,";
            break;
        case TASKTYPE_COMPUTE:
            if (!found_halo_exchange) {
                compute_before_halo_exchange = true;
                warning                      = true;
            }
            if (boundaries_defined != BOUNDARY_XYZ) {
                compute_before_boundary_condition = true;
                warning                           = true;
            }
            // found_compute = true;
            task_graph_repr += "Compute,";
            break;
        }
    }

    task_graph_repr += "}";

    std::string msg = "";

    if (!found_halo_exchange) {
        msg += " - No halo exchange defined in task graph.\n";
        error = true;
    }

    if (boundaries_defined != BOUNDARY_XYZ) {
        error = true;
    }
    if ((boundaries_defined & BOUNDARY_X_TOP) != BOUNDARY_X_TOP) {
        msg += " - Boundary conditions not defined for top X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_X_BOT) != BOUNDARY_X_BOT) {
        msg += " - Boundary conditions not defined for bottom X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_TOP) != BOUNDARY_Y_TOP) {
        msg += " - Boundary conditions not defined for top Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_BOT) != BOUNDARY_Y_BOT) {
        msg += " - Boundary conditions not defined for bottom Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_TOP) != BOUNDARY_Z_TOP) {
        msg += " - Boundary conditions not defined for top Z boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_BOT) != BOUNDARY_Z_BOT) {
        msg += " - Boundary conditions not defined for bottom Z boundary.\n";
    }

    // This warning is probably unnecessary
    /*
    if (!found_compute) {
        //msg += " - No compute kernel defined in task graph.\n";
        //warning = true;
    }
    */

    if (found_halo_exchange && boundary_condition_before_halo_exchange) {
        msg += " - Boundary condition before halo exchange. Halo exchange must come first.\n";
    }
    if (boundaries_defined == BOUNDARY_XYZ && compute_before_boundary_condition) {
        msg += " - Compute ordered before boundary conditions. Boundary conditions must usually be "
               "resolved before running kernels.\n";
    }
    if (found_halo_exchange && compute_before_halo_exchange) {
        msg += " - Compute ordered before halo exchange. Halo exchange must usually be performed "
               "before running kernels.\n";
    }

    // if (error) {
    //     ERROR(("\nIncorrect task graph " + task_graph_repr + ":\n" + msg).c_str())
    // }
    if (warning) {
        WARNING(("\nUnusual task graph " + task_graph_repr + ":\n" + msg).c_str())
    }
}
void
testNewRegionConstruction(){
    int3 nn = {128,128,128};
    printf("Starting new region construction test\n");
    Field fields[] = {VTXBUF_LNRHO};
    for(int x_id=-1;x_id<=1;x_id++){
        for(int y_id=-1;y_id<=1;y_id++){
            for(int z_id=-1;z_id<=1;z_id++){
                int3 old_id = {x_id,y_id,z_id};
                RegionId new_id = {0,{x_id,y_id,z_id}};
                Region test_region = Region(RegionFamily::Compute_input,old_id,nn,fields,1);
                Region new_test_region = Region(RegionFamily::Compute_input,new_id,nn,fields,1);
                assert(test_region.position == new_test_region.position);
                assert(test_region.dims == new_test_region.dims);
                test_region =Region(RegionFamily::Compute_output,old_id,nn,fields,1);
                new_test_region = Region(RegionFamily::Compute_output,new_id,nn,fields,1);
                assert(test_region.position == new_test_region.position);
                assert(test_region.dims == new_test_region.dims);
            }
        }
    }
    int3 correct_position = {6,6,6};
    int3 correct_dims= {3,3,3};
    RegionId new_id = {1,{-1,-1,-1}};
    Region new_test_region = Region(RegionFamily::Compute_output,new_id,nn,fields,1);
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    new_test_region = Region(RegionFamily::Compute_input,new_id,nn,fields,1);
    correct_position = {3,3,3};
    correct_dims = {9,9,9};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    new_id = {0,{0,0,0}};
    new_test_region = Region(RegionFamily::Compute_output,new_id,nn,fields,1);
    correct_position = {6,6,6};
    correct_dims = {122,122,122};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    new_test_region = Region(RegionFamily::Compute_input,new_id,nn,fields,1);
    correct_position = {3,3,3};
    correct_dims = {128,128,128};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    //Test new inner core core the most important one
    new_id = {1,{0,0,0}};
    new_test_region = Region(RegionFamily::Compute_output,new_id,nn,fields,1);
    correct_position = {9,9,9};
    correct_dims = {116,116,116};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    new_test_region = Region(RegionFamily::Compute_input,new_id,nn,fields,1);
    correct_position = {6,6,6};
    correct_dims = {122,122,122};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    //Test the right now i.e. x_id = 1 z_id=1
    correct_position = {125,6,125};
    correct_dims= {3,3,3};
    new_id = {1,{1,-1,1}};
    new_test_region = Region(RegionFamily::Compute_output,new_id,nn,fields,1);
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    new_test_region = Region(RegionFamily::Compute_input,new_id,nn,fields,1);
    correct_position = {122,3,122};
    correct_dims = {9,9,9};
    assert(new_test_region.position == correct_position);
    assert(new_test_region.dims == correct_dims);
    std::vector<Region> regions;
    std::vector<int3> ids;
    int shell_num_iter = 1;
    for(int shell_num = 0; shell_num<=shell_num_iter;shell_num++){
        for(int x_id=-1;x_id<=1;x_id++){
            for(int y_id=-1;y_id<=1;y_id++){
                for(int z_id=-1;z_id<=1;z_id++){
                    // we don't want to add the current shell grid but the
                     // outer regions of it and the inner region is the whole next shell grid
                    if(x_id != 0 || y_id != 0 || z_id != 0){
                            regions.push_back(Region(RegionFamily::Compute_output,{shell_num,{x_id,y_id,z_id}},nn,fields,1));
                            ids.push_back({x_id,y_id,z_id});
                        }                   
                    }
                }
            }
        }
    regions.push_back(Region(RegionFamily::Compute_output,{shell_num_iter,{0,0,0}},nn,fields,1));
    ids.push_back({0,0,0});
    for(int i=0;i<regions.size();i++){
        for(int j=i+1;j<regions.size();j++){
            bool overlap = regions[i].overlaps(&regions[j]);
            if(overlap){
                Region first = regions[i];
                Region second = regions[j];
                printf("Overlap found!\nFirst: position: %d,%d,%d\ndims: %d,%d,%d\nSecond position %d,%d,%d\ndims: %d,%d,%d\n",first.position.x,first.position.y,first.position.z,first.dims.x,first.dims.y,first.dims.z,second.position.x,second.position.y,second.position.z,second.dims.x,second.dims.y,second.dims.z);
                printf("%d,%d\n",i,j);
                printf("%d,%d,%d\n%d,%d,%d\n",ids[i].x,ids[i].y,ids[i].z,ids[j].x,ids[j].y,ids[j].z);
                fflush(stdout);
                assert(false);
            }
        }
    }
    printf("Region test passed :)\n");
}
AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition ops[], const size_t n_ops)
{
    // ERRCHK(grid.initialized);

    int rank;
    MPI_Comm_rank(astaroth_comm, &rank);

    check_ops(ops, n_ops);
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Allocating task graph\n");

    testNewRegionConstruction();
    AcTaskGraph* graph = new AcTaskGraph();

    //Calculate storage need for compute region since they can have different levels of decomposition
    size_t required_storage_for_comp_regions = 0;
    for (size_t i = 0; i < n_ops; i++) {
        auto op = ops[i];
        switch (op.task_type) {
        case TASKTYPE_COMPUTE: {    
            required_storage_for_comp_regions += 26*(op.shell_num+1)+1;
            break;
            }
        default: {
            break;
            }
        }
    }
    printf("Storage required for comp_regions: %d\n", required_storage_for_comp_regions);
    graph->periodic_boundaries = BOUNDARY_NONE;

    graph->halo_tasks.reserve(n_ops * Region::n_halo_regions);
    graph->all_tasks.reserve(max(n_ops* Region::n_halo_regions, required_storage_for_comp_regions));

    // Create tasks for each operation & store indices to ranges of tasks belonging to operations
    std::vector<size_t> op_indices;
    op_indices.reserve(n_ops);

    int3 nn         = grid.nn;
    uint3_64 decomp = grid.decomposition;
    int3 pid3d      = getPid3D(rank, grid.decomposition);
    Device device   = grid.device;

    auto boundary_normal = [&decomp, &pid3d](int tag) -> int3 {
        int3 neighbor = pid3d + Region::tag_to_id(tag);
        if (neighbor.z == -1) {
            return int3{0, 0, -1};
        }
        else if (neighbor.z == (int)decomp.z) {
            return int3{0, 0, 1};
        }
        else if (neighbor.y == -1) {
            return int3{0, -1, 0};
        }
        else if (neighbor.y == (int)decomp.y) {
            return int3{0, 1, 0};
        }
        else if (neighbor.x == -1) {
            return int3{-1, 0, 0};
        }
        else if (neighbor.x == (int)decomp.x) {
            return int3{1, 0, 0};
        }
        else {
            // Something went wrong, this tag does not identify a boundary region.
            return int3{0, 0, 0};
        }
        // return int3{(neighbor.x == -1) ? -1 : (neighbor.x == (int)decomp.x ? 1 : 0),
        //             (neighbor.y == -1) ? -1 : (neighbor.y == (int)decomp.y ? 1 : 0),
        //             (neighbor.z == -1) ? -1 : (neighbor.z == (int)decomp.z ? 1 : 0)};
    };

    // The tasks start at different offsets from the beginning of the iteration
    // this array of bools keep track of that state
    std::array<bool, NUM_VTXBUF_HANDLES> swap_offset{};

    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Creating tasks: %lu ops\n", n_ops);

    for (size_t i = 0; i < n_ops; i++) {
        acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Creating tasks for op %lu\n", i);
        auto op = ops[i];
        op_indices.push_back(graph->all_tasks.size());

        if (op.task_type == TASKTYPE_BOUNDCOND && op.bound_cond == BOUNDCOND_PERIODIC) {
            graph->periodic_boundaries = (AcBoundary)(graph->periodic_boundaries | op.boundary);
        }
        switch (op.task_type) {

        case TASKTYPE_COMPUTE: {
            for(int shell_num = 0; shell_num<=op.shell_num;shell_num++){
                for(int x_id=-1;x_id<=1;x_id++){
                    for(int y_id=-1;y_id<=1;y_id++){
                        for(int z_id=-1;z_id<=1;z_id++){
                            // we don't want to add the current shell grid but the
                            // outer regions of it and the inner region is the whole next shell grid
                            if(x_id != 0 || y_id != 0 || z_id != 0){
                                auto task = std::make_shared<ComputeTask>(op, i, RegionId{shell_num, int3{x_id,y_id,z_id}}, nn, device, swap_offset);
                                graph->all_tasks.push_back(task);
                            }                   
                        }
                    }
                }
            }
            auto inner_task = std::make_shared<ComputeTask>(op, i, RegionId{op.shell_num, int3{0,0,0}}, nn, device, swap_offset);
            graph->all_tasks.push_back(inner_task);
            Kernel kernel = kernels[(int)op.kernel];
            // for (int tag = Region::min_comp_tag; tag < Region::max_comp_tag; tag++) {
            //     auto task = std::make_shared<ComputeTask>(op, i, tag, nn, device, swap_offset);
            //     graph->all_tasks.push_back(task);
            // }
            for (size_t buf = 0; buf < op.num_fields_out; buf++) {
                swap_offset[op.fields_out[buf]] = !swap_offset[op.fields_out[buf]];
            }
            break;
        }

        case TASKTYPE_HALOEXCHANGE: {
            acVerboseLogFromRootProc(rank, "Creating halo exchange tasks\n");
            int tag0 = grid.mpi_tag_space_count * Region::max_halo_tag;
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (!Region::is_on_boundary(decomp, rank, tag, BOUNDARY_XYZ)) {
                    auto task = std::make_shared<HaloExchangeTask>(op, i, tag0, tag, nn, decomp,
                                                                   device, swap_offset);
                    graph->halo_tasks.push_back(task);
                    graph->all_tasks.push_back(task);
                }
            }
            acVerboseLogFromRootProc(rank, "Halo exchange tasks created\n");
            grid.mpi_tag_space_count++;
            break;
        }

        case TASKTYPE_BOUNDCOND: {
            acVerboseLogFromRootProc(rank, "Creating Boundcond tasks\n");
            AcBoundcond bc = op.bound_cond;
            int tag0       = grid.mpi_tag_space_count * Region::max_halo_tag;
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (Region::is_on_boundary(decomp, rank, tag, op.boundary)) {
                    if (bc == BOUNDCOND_PERIODIC) {
                        acVerboseLogFromRootProc(rank, "Creating periodic bc task with tag%d\n",
                                                 tag);
                        auto task = std::make_shared<HaloExchangeTask>(op, i, tag0, tag, nn, decomp,
                                                                       device, swap_offset);
                        acVerboseLogFromRootProc(rank,
                                                 "Done creating periodic bc task with tag%d\n",
                                                 tag);

                        graph->halo_tasks.push_back(task);
                        graph->all_tasks.push_back(task);
                    }
                    else {
                        acVerboseLogFromRootProc(rank, "Creating generic bc with tag%d\n", tag);
                        auto task = std::make_shared<BoundaryConditionTask>(op,
                                                                            boundary_normal(tag), i,
                                                                            tag, nn, device,
                                                                            swap_offset);
                        graph->all_tasks.push_back(task);
                    }
                }
            }
            acVerboseLogFromRootProc(rank, "Boundcond tasks created\n");
            grid.mpi_tag_space_count++;
            break;
        }

        case TASKTYPE_SPECIAL_MHD_BOUNDCOND: {
#ifdef AC_INTEGRATION_ENABLED
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (Region::is_on_boundary(decomp, rank, tag, op.boundary)) {
                    auto task = std::make_shared<SpecialMHDBoundaryConditionTask>(op,
                                                                                  boundary_normal(
                                                                                      tag),
                                                                                  i, tag, nn,
                                                                                  device,
                                                                                  swap_offset);
                    graph->all_tasks.push_back(task);
                }
            }
#endif
            break;
        }
        }
    }
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done creating tasks\n");

    op_indices.push_back(graph->all_tasks.size());
    graph->vtxbuf_swaps = swap_offset;

    graph->halo_tasks.shrink_to_fit();
    graph->all_tasks.shrink_to_fit();

    // In order to reduce redundant dependencies, we keep track of which tasks are connected
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Calculating dependencies\n");

    const size_t n_tasks               = graph->all_tasks.size();
    const size_t adjacancy_matrix_size = n_tasks * n_tasks;
    bool adjacent[adjacancy_matrix_size];
    memset(adjacent, 0, adjacancy_matrix_size * sizeof(adjacent[0]));
    for (size_t i = 0; i < adjacancy_matrix_size; ++i) { // Belt & suspenders safety
        ERRCHK_ALWAYS(adjacent[i] == false);
    }

    //...and check if there is already a forward path that connects two tasks
    auto forward_search = [&adjacent, &op_indices, n_tasks,
                           n_ops](size_t preq, size_t dept, size_t preq_op, size_t path_len) {
        bool visited[n_tasks];
        memset(visited, 0, n_tasks * sizeof(visited[0]));
        for (size_t i = 0; i < n_tasks; ++i) { // Belt & suspenders safety
            ERRCHK_ALWAYS(visited[i] == false);
        }

        size_t start_op = (preq_op + 1) % n_ops;

        struct walk_node {
            size_t node;
            size_t op_offset;
        };
        std::queue<walk_node> walk;
        walk.push({preq, 0});

        while (!walk.empty()) {
            auto curr = walk.front();
            walk.pop();
            if (adjacent[curr.node * n_tasks + dept]) {
                return true;
            }
            for (size_t op_offset = curr.op_offset; op_offset < path_len; op_offset++) {
                size_t op = (start_op + op_offset) % n_ops;
                for (size_t neighbor = op_indices[op]; neighbor != op_indices[op + 1]; neighbor++) {
                    if (!visited[neighbor] && adjacent[curr.node * n_tasks + neighbor]) {
                        walk.push({neighbor, op_offset});
                        visited[neighbor] = true;
                    }
                }
            }
        }
        return false;
    };

    // We walk through all tasks, and compare tasks from pairs of operations at
    // a time. Pairs are considered in order of increasing distance between the
    // operations in the pair. The final set of pairs that are considered are
    // self-equal pairs, since the operations form a cycle when iterated over
    for (size_t op_offset = 0; op_offset < n_ops; op_offset++) {
        for (size_t dept_op = 0; dept_op < n_ops; dept_op++) {
            size_t preq_op = (n_ops + dept_op - op_offset - 1) % n_ops;
            for (auto i = op_indices[preq_op]; i != op_indices[preq_op + 1]; i++) {
                auto preq_task = graph->all_tasks[i];
                if (preq_task->active) {
                    for (auto j = op_indices[dept_op]; j != op_indices[dept_op + 1]; j++) {
                        auto dept_task = graph->all_tasks[j];
                        // Task A depends on task B if the output region of A overlaps with the
                        // input region of B.
                        if (dept_task->active &&
                            (preq_task->output_region.overlaps(&(dept_task->input_region)) ||
                             preq_task->output_region.overlaps(&(dept_task->output_region)))) {
                            // iteration offset of 0 -> dependency in the same iteration
                            // iteration offset of 1 -> dependency from preq_task in iteration k to
                            // dept_task in iteration k+1
                            if (!forward_search(i, j, preq_op, op_offset)) {
                                preq_task->registerDependent(dept_task, preq_op < dept_op ? 0 : 1);
                                adjacent[i * n_tasks + j] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done calculating dependencies\n");

    // Finally sort according to a priority. Larger volumes first and comm before comp
    auto sort_lambda = [](std::shared_ptr<Task> t1, std::shared_ptr<Task> t2) {
        auto comp1 = t1->task_type == TASKTYPE_COMPUTE;
        auto comp2 = t2->task_type == TASKTYPE_COMPUTE;

        auto vol1 = t1->output_region.volume;
        auto vol2 = t2->output_region.volume;
        auto dim1 = t1->output_region.dims;
        auto dim2 = t2->output_region.dims;

        return vol1 > vol2 ||
               (vol1 == vol2 && ((!comp1 && comp2) || dim1.x < dim2.x || dim1.z > dim2.z));
    };
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Sorting tasks by priority\n");

    std::sort(graph->halo_tasks.begin(), graph->halo_tasks.end(), sort_lambda);
    std::sort(graph->all_tasks.begin(), graph->all_tasks.end(), sort_lambda);
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done sorting tasks by priority\n");
    return graph;
}

AcResult
acGridDestroyTaskGraph(AcTaskGraph* graph)
{
    graph->all_tasks.clear();
    graph->comp_tasks.clear();
    graph->halo_tasks.clear();
    delete graph;
    return AC_SUCCESS;
}

AcResult
acGridExecuteTaskGraph(AcTaskGraph* graph, size_t n_iterations)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acDeviceSynchronizeStream(grid.device, stream);
    cudaSetDevice(grid.device->id);

    if (graph->trace_file.enabled) {
        timer_reset(&(graph->trace_file.timer));
    }

    for (auto& task : graph->all_tasks) {
        if (task->active) {
            task->syncVBA();
            task->setIterationParams(0, n_iterations);
        }
    }

    bool ready;
    do {
        ready = true;
        for (auto& task : graph->all_tasks) {
            if (task->active) {
                task->update(graph->vtxbuf_swaps, &(graph->trace_file));
                ready &= task->isFinished();
            }
        }
    } while (!ready);

    if (n_iterations % 2 != 0) {
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; i++) {
            if (graph->vtxbuf_swaps[i]) {
                acDeviceSwapBuffer(grid.device, (VertexBufferHandle)i);
            }
        }
    }
    return AC_SUCCESS;
}

#ifdef AC_INTEGRATION_ENABLED
AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridLoadScalarUniform(stream, AC_dt, dt);
    acDeviceSynchronizeStream(grid.device, stream);
    return acGridExecuteTaskGraph(grid.default_tasks.get(), 1);
}
#endif // AC_INTEGRATION_ENABLED

AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    // Active halo exchange tasks use send() instead of exchange() because there is an active eager
    // receive that needs to be used. A new eager receive is posted after the exchange.
    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->syncVBA();
        halo_task->pack();
        if (halo_task->active) {
            halo_task->send();
        }
        else {
            halo_task->exchange();
        }
    }

    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->wait_recv();
        halo_task->unpack();
        halo_task->sync();
        if (halo_task->active) {
            halo_task->receive();
        }
    }

    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->wait_send();
    }
    return AC_SUCCESS;
}

static AcResult
distributedScalarReduction(const AcReal local_result, const ReductionType rtype, AcReal* result)
{
    MPI_Op op;
    if (rtype == RTYPE_MAX || rtype == RTYPE_ALFVEN_MAX) {
        op = MPI_MAX;
    }
    else if (rtype == RTYPE_MIN || rtype == RTYPE_ALFVEN_MIN) {
        op = MPI_MIN;
    }
    else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_SUM ||
             rtype == RTYPE_ALFVEN_RMS) {
        op = MPI_SUM;
    }
    else {
        ERROR("Unrecognised rtype");
    }

    int rank;
    MPI_Comm_rank(astaroth_comm, &rank);

    AcReal mpi_res;
    MPI_Allreduce(&local_result, &mpi_res, 1, AC_REAL_MPI_TYPE, op, astaroth_comm);

    if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_ALFVEN_RMS) {
        const AcReal inv_n = AcReal(1.) / (grid.nn.x * grid.decomposition.x * grid.nn.y *
                                           grid.decomposition.y * grid.nn.z * grid.decomposition.z);
        mpi_res            = sqrt(inv_n * mpi_res);
    }
    *result = mpi_res;
    return AC_SUCCESS;
}

AcResult
acGridReduceScal(const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceScalNotAveraged(device, stream, rtype, vtxbuf_handle, &local_result);

    return distributedScalarReduction(local_result, rtype, result);
}

AcResult
acGridReduceVec(const Stream stream, const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceVecNotAveraged(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, &local_result);

    return distributedScalarReduction(local_result, rtype, result);
}

AcResult
acGridReduceVecScal(const Stream stream, const ReductionType rtype,
                    const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                    const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                    AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceVecScalNotAveraged(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                     &local_result);

    return distributedScalarReduction(local_result, rtype, result);
}

/** */
AcResult
acGridLaunchKernel(const Stream stream, const Kernel kernel, const int3 start, const int3 end)
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(stream);
    return acDeviceLaunchKernel(grid.device, stream, kernel, start, end);
}

AcResult
acGridSwapBuffers(void)
{
    ERRCHK(grid.initialized);
    return acDeviceSwapBuffers(grid.device);
}

/** */
AcResult
acGridLoadStencil(const Stream stream, const Stencil stencil,
                  const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(stream);
    return acDeviceLoadStencil(grid.device, stream, stencil, data);
}

/** */
AcResult
acGridStoreStencil(const Stream stream, const Stencil stencil,
                   AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(stream);
    return acDeviceStoreStencil(grid.device, stream, stencil, data);
}

/** */
AcResult
acGridLoadStencils(const Stream stream,
                   const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);
    ERRCHK((int)AC_SUCCESS == 0);
    ERRCHK((int)AC_FAILURE == 1);
    acGridSynchronizeStream(stream);

    int retval = 0;
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        retval |= acGridLoadStencil(stream, (Stencil)i, data[i]);

    return (AcResult)retval;
}

/** */
AcResult
acGridStoreStencils(const Stream stream,
                    AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);
    ERRCHK((int)AC_SUCCESS == 0);
    ERRCHK((int)AC_FAILURE == 1);
    acGridSynchronizeStream(stream);

    int retval = 0;
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        retval |= acGridStoreStencil(stream, (Stencil)i, data[i]);

    return (AcResult)retval;
}

/*
static AcResult
volume_copy_to_from_host(const VertexBufferHandle vtxbuf, const AccessType type)
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(STREAM_ALL); // Possibly unnecessary

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    if (type == ACCESS_WRITE) {
        const AcReal* in      = device->vba.in[vtxbuf];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[vtxbuf];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        // ---------------------------------------
        // Buffer through CPU
        cudaSetDevice(device->id);
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
        // ----------------------------------------
    }

    if (type == ACCESS_READ) {
        AcReal* in           = device->vba.out[vtxbuf];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

        AcReal* out           = device->vba.in[vtxbuf];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

        // ---------------------------------------
        // Buffer through CPU
        cudaSetDevice(device->id);
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(in, grid.submesh.vertex_buffer[vtxbuf], count, cudaMemcpyHostToDevice);
        // ----------------------------------------

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);

        // Apply boundconds and sync
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    }
    acGridSynchronizeStream(STREAM_ALL); // Possibly unnecessary
    return AC_SUCCESS;
}

static AcResult
access_vtxbuf_on_disk(const VertexBufferHandle vtxbuf, const char* path, const AccessType type)
{
    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;
    const int3 nn         = info.int3_params[AC_global_grid_n];
    const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 offset     = info.int3_params[AC_multigpu_offset]; // Without halo

    MPI_Datatype subarray;
    const int arr_nn[]     = {nn.z, nn.y, nn.x};
    const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
    const int arr_offset[] = {offset.z, offset.y, offset.x};
    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
    return AC_SUCCESS;
}
*/

/*

    write:
        sync transfer to host
        async write to disk
    read:
        async read from disk
        sync transfer to device

    sync:
        complete write or read locally  (future.get() and status complete)
        complete write or read globally (MPI_Barrier)

    static:
        future
        status

    static std::future<void> future;
    static AccessType access_type;
    static bool complete = true;
*/

/*
#include <chrono>
#include <future>

static std::future<void> future;
static AccessType access_type = ACCESS_WRITE;
static bool complete          = true;

AcResult
acGridDiskAccessSyncOld(void)
{
    ERRCHK(grid.initialized);

    // Sync and mark as completed
    if (future.valid())
        future.get();

    if (access_type == ACCESS_READ)
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
            volume_copy_to_from_host((VertexBufferHandle)i, ACCESS_READ);

    acGridSynchronizeStream(STREAM_ALL);
    access_type = ACCESS_WRITE;
    complete    = true;
    return AC_SUCCESS;
}

static void
write_async(void)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i); // Note: could use vtxbuf_names[i]
        access_vtxbuf_on_disk((VertexBufferHandle)i, file, ACCESS_WRITE);
    }
}

static void
read_async(void)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i); // Note: could use vtxbuf_names[i]
        access_vtxbuf_on_disk((VertexBufferHandle)i, file, ACCESS_READ);
    }
}

AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK_ALWAYS(grid.initialized);
    WARNING("\n------------------------\n"
            "acGridDiskAccessLaunch does not work concurrently with acGridIntegrate due to an\n"
            "unknown issue (invalid CUDA context, double free, or invalid memory access). Suspect\n"
            "some complex interaction with the underlying MPI library and the asynchronous task\n"
            "system. `acGridAccessMeshOnDiskSynchronous` has been tested to work on multiple\n"
            "processes. It is recommended to use that instead in production."
            "\n------------------------\n");

    acGridDiskAccessSync();
    ERRCHK_ALWAYS(!future.valid());

    ERRCHK_ALWAYS(complete);
    complete    = false;
    access_type = type;

    if (type == ACCESS_WRITE) {
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
            volume_copy_to_from_host((VertexBufferHandle)i, ACCESS_WRITE);

        future = std::async(std::launch::async, write_async);
    }
    else if (type == ACCESS_READ) {
        future = std::async(std::launch::async, read_async);
    }
    else {
        ERROR("Unknown access type in acGridDiskAccessLaunch");
        return AC_FAILURE;
    }
    return AC_SUCCESS;
}*/

#define USE_CPP_THREADS (1)
#if USE_CPP_THREADS
#include <thread>
#include <vector>

static std::vector<std::thread> threads;
static bool running = false;

AcResult
acGridDiskAccessSync(void)
{
    ERRCHK(grid.initialized);

    for (auto& thread : threads)
        if (thread.joinable())
            thread.join();

    threads.clear();

    acGridSynchronizeStream(STREAM_ALL);
    running = false;
    return AC_SUCCESS;
}

AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(type == ACCESS_WRITE);
    ERRCHK_ALWAYS(!running)
    running = true;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        const Device device = grid.device;
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const AcMeshInfo info = device->local_config;
        // const int3 nn         = info.int3_params[AC_global_grid_n];
        // const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 offset     = info.int3_params[AC_multigpu_offset]; // Without halo
        AcReal* host_buffer = grid.submesh.vertex_buffer[i];

        const AcReal* in      = device->vba.in[i];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[i];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        const auto write_async = [](const int device_id, const int i, const AcMeshInfo info,
                                    const AcReal* host_buffer) {
#if USE_PERFSTUBS
	    PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridDiskAccessLaunch::write_async");
#endif
            cudaSetDevice(device_id);

            char path[4096] = "";
            sprintf(path, "%s.out", vtxbuf_names[i]);

            const int3 offset = info.int3_params[AC_multigpu_offset]; // Without halo
#if USE_DISTRIBUTED_IO
#define USE_POSIX_IO (0)

#if USE_POSIX_IO
            char outfile[4096] = "";
            snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);

            FILE* fp = fopen(outfile, "w");
            ERRCHK_ALWAYS(fp);

            const size_t count         = acVertexBufferCompdomainSize(info);
            const size_t count_written = fwrite(host_buffer, sizeof(AcReal), count, fp);
            ERRCHK_ALWAYS(count_written == count);

            fclose(fp);
#else // Use MPI IO
            MPI_File file;
            int mode           = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            char outfile[4096] = "";
            snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);
#if AC_VERBOSE
            fprintf(stderr, "Writing %s\n", outfile);
#endif
            int retval = MPI_File_open(MPI_COMM_SELF, outfile, mode, MPI_INFO_NULL, &file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            MPI_Status status;
            const size_t count = acVertexBufferCompdomainSize(info);
            retval = MPI_File_write(file, host_buffer, count, AC_REAL_MPI_TYPE, &status);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            retval = MPI_File_close(&file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
#endif
#else
            MPI_Datatype subarray;
            const int3 nn          = info.int3_params[AC_global_grid_n];
            const int3 nn_sub      = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
            const int arr_nn[]     = {nn.z, nn.y, nn.x};
            const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
            const int arr_offset[] = {offset.z, offset.y, offset.x};

            // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i,
            // nn_sub.x %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
            //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y,
            //         offset.x);

            MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C,
                                     AC_REAL_MPI_TYPE, &subarray);
            MPI_Type_commit(&subarray);

            MPI_File file;

#if AC_VERBOSE
            fprintf(stderr, "Writing %s\n", path);
#endif

            int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) ==
                          MPI_SUCCESS); // ISSUE TODO: fails with multiple threads

            ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native",
                                            MPI_INFO_NULL) == MPI_SUCCESS);

            MPI_Status status;

            const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
            ERRCHK_ALWAYS(MPI_File_write_all(file, host_buffer, nelems, AC_REAL_MPI_TYPE,
                                             &status) == MPI_SUCCESS);

            ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

            MPI_Type_free(&subarray);
#endif
#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

        threads.push_back(std::thread(write_async, device->id, i, info, host_buffer));
        // write_async();
    }

    return AC_SUCCESS;
}

AcResult
acGridWriteMeshToDiskLaunch(const char* dir, const char* label)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running)
    running = true;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        const Device device = grid.device;
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const AcMeshInfo info = device->local_config;
        // const int3 nn         = info.int3_params[AC_global_grid_n];
        // const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 offset     = info.int3_params[AC_multigpu_offset]; // Without halo
        AcReal* host_buffer = grid.submesh.vertex_buffer[i];

        const AcReal* in      = device->vba.in[i];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[i];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        const int3 offset = info.int3_params[AC_multigpu_offset]; // Without halo
        char filepath[4096];
#if USE_DISTRIBUTED_IO
        sprintf(filepath, "%s/%s-segment-%d-%d-%d-%s.mesh", dir, vtxbuf_names[i], offset.x,
                offset.y, offset.z, label);
#else
        sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[i], label);
#endif

        const auto write_async = [filepath, offset](const AcMeshInfo info,
                                                    const AcReal* host_buffer) {

#if USE_PERFSTUBS
	    PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridWriteMeshToDiskLaunch::write_async");
#endif


#if USE_DISTRIBUTED_IO
            (void)offset; // Unused
#define USE_POSIX_IO (0)
#if USE_POSIX_IO
            FILE* fp = fopen(outfile, "w");
            ERRCHK_ALWAYS(fp);

            const size_t count         = acVertexBufferCompdomainSize(info);
            const size_t count_written = fwrite(host_buffer, sizeof(AcReal), count, fp);
            ERRCHK_ALWAYS(count_written == count);

            fclose(fp);
#else // Use MPI IO
            MPI_File file;
            int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            // fprintf(stderr, "Writing %s\n", filepath);
            int retval = MPI_File_open(MPI_COMM_SELF, filepath, mode, MPI_INFO_NULL, &file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            MPI_Status status;
            const size_t count = acVertexBufferCompdomainSize(info);
            retval = MPI_File_write(file, host_buffer, count, AC_REAL_MPI_TYPE, &status);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            retval = MPI_File_close(&file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
#endif
#undef USE_POSIX_IO
#else
            ERROR("Collective mesh writing not working with async IO");
            MPI_Datatype subarray;
            const int3 nn          = info.int3_params[AC_global_grid_n];
            const int3 nn_sub      = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
            const int arr_nn[]     = {nn.z, nn.y, nn.x};
            const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
            const int arr_offset[] = {offset.z, offset.y, offset.x};

            // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i,
            // nn_sub.x %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
            //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y,
            //         offset.x);

            MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C,
                                     AC_REAL_MPI_TYPE, &subarray);
            MPI_Type_commit(&subarray);

            MPI_File file;
            // fprintf(stderr, "Writing %s\n", filepath);

            int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                          MPI_SUCCESS); // ISSUE TODO: fails with multiple threads

            ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native",
                                            MPI_INFO_NULL) == MPI_SUCCESS);

            MPI_Status status;

            const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
            ERRCHK_ALWAYS(MPI_File_write_all(file, host_buffer, nelems, AC_REAL_MPI_TYPE,
                                             &status) == MPI_SUCCESS);

            ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

            MPI_Type_free(&subarray);
#endif
#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

        // write_async(info, host_buffer); // Synchronous, non-threaded
        threads.push_back(std::thread(write_async, info, host_buffer)); // Async, threaded
    }

    return AC_SUCCESS;
}

AcResult
acGridWriteSlicesToDiskLaunch(const char* dir, const char* label)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running);
    running = true;

    const Device device       = grid.device;
    const AcMeshInfo info     = device->local_config;
    const int3 local_nn       = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn      = info.int3_params[AC_global_grid_n];
    const int3 global_offset  = info.int3_params[AC_multigpu_offset];
    const int3 global_pos_min = global_offset;
    // const int3 global_pos_max = global_pos_min + local_nn;

    const int global_z = global_nn.z / 2;
    const int local_z  = global_z - global_pos_min.z;
    const int color    = local_z >= 0 && local_z < local_nn.z ? 0 : MPI_UNDEFINED;

    for (int field = 0; field < NUM_FIELDS; ++field) {

        acDeviceSynchronizeStream(device, STREAM_ALL);

        const int3 slice_volume = (int3){
            info.int_params[AC_nx],
            info.int_params[AC_ny],
            1,
        };
        const int3 slice_offset = (int3){0, 0, local_z};

        const AcReal* in     = device->vba.in[field];
        const int3 in_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info) +
                               slice_offset;
        const int3 in_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

        AcReal* out           = device->vba.out[field];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = slice_volume;

        if (color != MPI_UNDEFINED)
            acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                               out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        AcReal* host_buffer = grid.submesh.vertex_buffer[field];
        const size_t count  = slice_volume.x * slice_volume.y * slice_volume.z;
        const size_t bytes  = sizeof(host_buffer[0]) * count;
        if (color != MPI_UNDEFINED)
            cudaMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char filepath[4096];
#if USE_DISTRIBUTED_IO
        sprintf(filepath, "%s/%s-segment-at_%d_%d_%d-dims_%d_%d-%s.slice", dir, vtxbuf_names[field],
                global_pos_min.x, global_pos_min.y, global_z, local_nn.x, local_nn.y, label);
#else
        sprintf(filepath, "%s/%s-dims_%d_%d-%s.slice", dir, vtxbuf_names[field], global_nn.x,
                global_nn.y, label);
#endif

        int pid;
        MPI_Comm_rank(astaroth_comm, &pid);
        // if (color != MPI_UNDEFINED)
        //     fprintf(stderr, "Writing field %d, proc %d, to %s\n", field, pid, filepath);

        acGridSynchronizeStream(STREAM_ALL);
        const auto write_async = [filepath, global_nn, global_pos_min, slice_volume,
                                  color](const AcReal* host_buffer, const size_t count,
                                         const int device_id) {
#if USE_PERFSTUBS
	    PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridWriteMeshToDiskLaunch::write_async");
#endif

            cudaSetDevice(device_id);
            // Write to file

#if USE_DISTRIBUTED_IO
            (void)global_nn;      // Unused
            (void)global_pos_min; // Unused
            (void)slice_volume;   // Unused
#define USE_POSIX_IO (0)
#if USE_POSIX_IO
            if (color != MPI_UNDEFINED) {
                FILE* fp = fopen(filepath, "w");
                ERRCHK_ALWAYS(fp);

                const size_t count_written = fwrite(host_buffer, sizeof(AcReal), count, fp);
                ERRCHK_ALWAYS(count_written == count);

                fclose(fp);
            }
#else // Use MPI IO
            if (color != MPI_UNDEFINED) {
                MPI_File file;
                int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
                // fprintf(stderr, "Writing %s\n", filepath);
                int retval = MPI_File_open(MPI_COMM_SELF, filepath, mode, MPI_INFO_NULL, &file);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write(file, host_buffer, count, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&file);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);
            }
#endif
#undef USE_POSIX_IO
#else
            ERROR("Collective slice writing not working with async IO");
            // Possible MPI bug: need to cudaSetDevice or otherwise invalid context
            // But also causes a deadlock for some reason
            MPI_Comm slice_communicator;
            MPI_Comm_split(astaroth_comm, color, 0, &slice_communicator);
            if (color != MPI_UNDEFINED) {
                const int3 nn     = (int3){global_nn.x, global_nn.y, 1};
                const int3 nn_sub = slice_volume;

                const int nn_[]     = {nn.z, nn.y, nn.x};
                const int nn_sub_[] = {nn_sub.z, nn_sub.y, nn_sub.x};
                const int offset_[] = {
                    0,
                    global_pos_min.y,
                    global_pos_min.x,
                };
                MPI_Datatype subdomain;
                MPI_Type_create_subarray(3, nn_, nn_sub_, offset_, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                         &subdomain);
                MPI_Type_commit(&subdomain);

                // printf("Writing %s\n", filepath);

                MPI_File fp;
                int retval = MPI_File_open(slice_communicator, filepath,
                                           MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_set_view(fp, 0, AC_REAL_MPI_TYPE, subdomain, "native",
                                           MPI_INFO_NULL);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write_all(fp, host_buffer, count, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Type_free(&subdomain);

                MPI_Comm_free(&slice_communicator);
            }
#endif
	    
#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

        // write_async(host_buffer, count, device->id); // Synchronous, non-threaded
        threads.push_back(
            std::thread(write_async, host_buffer, count, device->id)); // Async, threaded
    }
    return AC_SUCCESS;
}

AcResult
acGridWriteSlicesToDiskCollectiveSynchronous(const char* dir, const char* label)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running);
    running = true;

    const Device device       = grid.device;
    const AcMeshInfo info     = device->local_config;
    const int3 local_nn       = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn      = info.int3_params[AC_global_grid_n];
    const int3 global_offset  = info.int3_params[AC_multigpu_offset];
    const int3 global_pos_min = global_offset;
    // const int3 global_pos_max = global_pos_min + local_nn;

    const int global_z = global_nn.z / 2;
    const int local_z  = global_z - global_pos_min.z;
    const int color    = local_z >= 0 && local_z < local_nn.z ? 0 : MPI_UNDEFINED;

    for (int field = 0; field < NUM_FIELDS; ++field) {

        acDeviceSynchronizeStream(device, STREAM_ALL);

        const int3 slice_volume = (int3){
            info.int_params[AC_nx],
            info.int_params[AC_ny],
            1,
        };
        const int3 slice_offset = (int3){0, 0, local_z};

        const AcReal* in     = device->vba.in[field];
        const int3 in_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info) +
                               slice_offset;
        const int3 in_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

        AcReal* out           = device->vba.out[field];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = slice_volume;

        if (color != MPI_UNDEFINED)
            acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                               out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        AcReal* host_buffer = grid.submesh.vertex_buffer[field];
        const size_t count  = slice_volume.x * slice_volume.y * slice_volume.z;
        const size_t bytes  = sizeof(host_buffer[0]) * count;
        if (color != MPI_UNDEFINED)
            cudaMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char filepath[4096];
        sprintf(filepath, "%s/%s-dims_%d_%d-%s.slice", dir, vtxbuf_names[field], global_nn.x,
                global_nn.y, label);

        int pid;
        MPI_Comm_rank(astaroth_comm, &pid);
        // if (color != MPI_UNDEFINED)
        //     fprintf(stderr, "Writing field %d, proc %d, to %s\n", field, pid, filepath);

        acGridSynchronizeStream(STREAM_ALL);
        const auto write_sync = [filepath, global_nn, global_pos_min, slice_volume,
                                  color](const AcReal* host_buffer, const size_t count,
                                         const int device_id) {
            cudaSetDevice(device_id);
            // Write to file

            // Possible MPI bug: need to cudaSetDevice or otherwise invalid context
            // But also causes a deadlock for some reason
            MPI_Comm slice_communicator;
            MPI_Comm_split(astaroth_comm, color, 0, &slice_communicator);
            if (color != MPI_UNDEFINED) {
                const int3 nn     = (int3){global_nn.x, global_nn.y, 1};
                const int3 nn_sub = slice_volume;

                const int nn_[]     = {nn.z, nn.y, nn.x};
                const int nn_sub_[] = {nn_sub.z, nn_sub.y, nn_sub.x};
                const int offset_[] = {
                    0,
                    global_pos_min.y,
                    global_pos_min.x,
                };
                MPI_Datatype subdomain;
                MPI_Type_create_subarray(3, nn_, nn_sub_, offset_, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                         &subdomain);
                MPI_Type_commit(&subdomain);

                // printf("Writing %s\n", filepath);

                MPI_File fp;
                int retval = MPI_File_open(slice_communicator, filepath,
                                           MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_set_view(fp, 0, AC_REAL_MPI_TYPE, subdomain, "native",
                                           MPI_INFO_NULL);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write_all(fp, host_buffer, count, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Type_free(&subdomain);

                MPI_Comm_free(&slice_communicator);
            }
        };

        write_sync(host_buffer, count, device->id); // Synchronous, non-threaded
        // threads.push_back(std::move(std::thread(write_sync, host_buffer, count, device->id)));
        // // Async, threaded
    }
    return AC_SUCCESS;
}
#else

static MPI_File files[NUM_VTXBUF_HANDLES];
static MPI_Request reqs[NUM_VTXBUF_HANDLES];
static bool req_running[NUM_VTXBUF_HANDLES];

AcResult
acGridDiskAccessSync(void)
{
    ERRCHK(grid.initialized);

    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (req_running[i]) {
            MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
            const int retval = MPI_File_close(&files[i]);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
            req_running[i] = false;
        }
    }
    MPI_Barrier(astaroth_comm);
    return AC_SUCCESS;
}

AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(type == ACCESS_WRITE);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        ERRCHK_ALWAYS(!reqs[i]);

        const Device device = grid.device;
        cudaSetDevice(device->id);
        cudaDeviceSynchronize();

        const AcMeshInfo info = device->local_config;
        AcReal* host_buffer   = grid.submesh.vertex_buffer[i];

        const AcReal* in      = device->vba.in[i];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[i];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char path[4096] = "";
        sprintf(path, "%s.out", vtxbuf_names[i]);

        const int3 offset  = info.int3_params[AC_multigpu_offset]; // Without halo
#if USE_DISTRIBUTED_IO
        int mode           = MPI_MODE_CREATE | MPI_MODE_WRONLY;
        char outfile[4096] = "";
        snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);

#if AC_VERBOSE
        fprintf(stderr, "Writing %s\n", outfile);
#endif

        int retval = MPI_File_open(MPI_COMM_SELF, outfile, mode, MPI_INFO_NULL, &files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        const size_t count = acVertexBufferCompdomainSize(info);
        retval = MPI_File_iwrite(files[i], host_buffer, count, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        req_running[i] = true;
#else
        MPI_Datatype subarray;
        const int3 nn          = info.int3_params[AC_global_grid_n];
        const int3 nn_sub      = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        const int arr_nn[]     = {nn.z, nn.y, nn.x};
        const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
        const int arr_offset[] = {offset.z, offset.y, offset.x};

        MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                 &subarray);
        MPI_Type_commit(&subarray);

#if AC_VERBOSE
        fprintf(stderr, "Writing %s\n", path);
#endif

        int flags  = MPI_MODE_CREATE | MPI_MODE_WRONLY;
        int retval = MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_set_view(files[i], 0, AC_REAL_MPI_TYPE, subarray, "native",
                                   MPI_INFO_NULL);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
#if 0   // Does not work
        retval = MPI_File_iwrite_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);
        MPI_Type_free(&subarray);
        req_running[i] = true;
#elif 0 // Does not work either, even though otherwise identical to the blocking version below
        // (except iwrite + wait)
        ERRCHK_ALWAYS(&files[i]);
        ERRCHK_ALWAYS(&reqs[i]);
        ERRCHK_ALWAYS(host_buffer);
        ERRCHK_ALWAYS(subarray);
        retval = MPI_File_iwrite_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_close(&files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Type_free(&subarray);
        req_running[i] = false;
#else   // Blocking, this works
        WARNING("Called collective non-blocking MPI_File_write_all, but currently blocks\n");
        MPI_Status status;
        retval = MPI_File_write_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &status);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_close(&files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Type_free(&subarray);
        req_running[i] = false;
#endif
#endif
    }

    return AC_SUCCESS;
}
#endif

AcResult
acGridAccessMeshOnDiskSynchronous(const VertexBufferHandle vtxbuf, const char* dir,
                                  const char* label, const AccessType type)
{
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;
    // const int3 nn         = info.int3_params[AC_global_grid_n];
    const int3 nn_sub = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 offset = info.int3_params[AC_multigpu_offset]; // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
#if USE_DISTRIBUTED_IO
    sprintf(filepath, "%s/%s-segment-%d-%d-%d-%s.mesh", dir, vtxbuf_names[vtxbuf], offset.x,
            offset.y, offset.z, label);
#else
    sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[vtxbuf], label);
#endif

#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
        const AcReal* in      = device->vba.in[vtxbuf];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[vtxbuf];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#ifndef NDEBUG
    if (type == ACCESS_READ) {
        const int3 nn              = info.int3_params[AC_global_grid_n];
        const size_t expected_size = sizeof(AcReal) * nn.x * nn.y * nn.z;
        FILE* fp                   = fopen(filepath, "r");
        ERRCHK_ALWAYS(fp);
        fseek(fp, 0L, SEEK_END);
        const size_t measured_size = ftell(fp);
        fclose(fp);
        if (expected_size != measured_size) {
            fprintf(stderr,
                    "Expected size did not match measured size (%lu vs %lu), factor of %g "
                    "difference\n",
                    expected_size, measured_size, (double)expected_size / measured_size);
            fprintf(stderr, "Note that old data files must be removed when switching to a smaller "
                            "mesh size, otherwise the file on disk will be too large (the above "
                            "factor < 1)\n");
            ERRCHK_ALWAYS(expected_size == measured_size);
        }
    }
#endif // NDEBUG

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    AcReal* arr = device->vba.out[vtxbuf];
#endif

#if USE_DISTRIBUTED_IO
    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;

    FILE* fp;
    if (type == ACCESS_READ)
        fp = fopen(filepath, "r");
    else
        fp = fopen(filepath, "w");
    ERRCHK_ALWAYS(fp);

    if (type == ACCESS_READ)
        fread(arr, sizeof(AcReal), nelems, fp);
    else
        fwrite(arr, sizeof(AcReal), nelems, fp);
    fclose(fp);
#else // Collective IO
    MPI_Datatype subarray;
    const int arr_nn[]     = {nn.z, nn.y, nn.x};
    const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
    const int arr_offset[] = {offset.z, offset.y, offset.x};

    // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i, nn_sub.x
    // %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
    //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y, offset.x);

    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                  MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
#endif

#ifndef NDEBUG
    if (type == ACCESS_WRITE) {
        const int3 nn              = info.int3_params[AC_global_grid_n];
        const size_t expected_size = sizeof(AcReal) * nn.x * nn.y * nn.z;
        FILE* fp                   = fopen(filepath, "r");
        ERRCHK_ALWAYS(fp);
        fseek(fp, 0L, SEEK_END);
        const size_t measured_size = ftell(fp);
        fclose(fp);
        if (expected_size != measured_size) {
            fprintf(stderr,
                    "Expected size did not match measured size (%lu vs %lu), factor of %g "
                    "difference\n",
                    expected_size, measured_size, (double)expected_size / measured_size);
            fprintf(stderr, "Note that old data files must be removed when switching to a smaller "
                            "mesh size, otherwise the file on disk will be too large (the above "
                            "factor < 1)\n");
            ERRCHK_ALWAYS(expected_size == measured_size);
        }
    }
#endif // NDEBUG

    if (type == ACCESS_READ) {
        AcReal* in           = device->vba.out[vtxbuf];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

        AcReal* out           = device->vba.in[vtxbuf];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        cudaDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END
    }
    return AC_SUCCESS;
}

AcResult
acGridAccessMeshOnDiskSynchronousDistributed(const VertexBufferHandle vtxbuf, const char* dir,
                                             const char* label, const AccessType type)
{
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;
    // const int3 nn         = info.int3_params[AC_global_grid_n];
    const int3 nn_sub = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 offset = info.int3_params[AC_multigpu_offset]; // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
    sprintf(filepath, "%s/%s-segment-%d-%d-%d-%s.mesh", dir, vtxbuf_names[vtxbuf], offset.x,
            offset.y, offset.z, label);
#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
        const AcReal* in      = device->vba.in[vtxbuf];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[vtxbuf];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    AcReal* arr = device->vba.out[vtxbuf];
#endif

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;

    FILE* fp;
    if (type == ACCESS_READ)
        fp = fopen(filepath, "r");
    else
        fp = fopen(filepath, "w");
    ERRCHK_ALWAYS(fp);

    if (type == ACCESS_READ)
        fread(arr, sizeof(AcReal), nelems, fp);
    else
        fwrite(arr, sizeof(AcReal), nelems, fp);
    fclose(fp);

    if (type == ACCESS_READ) {
        AcReal* in           = device->vba.out[vtxbuf];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

        AcReal* out           = device->vba.in[vtxbuf];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        cudaDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END
    }
    return AC_SUCCESS;
}

AcResult
acGridAccessMeshOnDiskSynchronousCollective(const VertexBufferHandle vtxbuf, const char* dir,
                                            const char* label, const AccessType type)
{
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;
    const int3 nn         = info.int3_params[AC_global_grid_n];
    const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 offset     = info.int3_params[AC_multigpu_offset]; // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
    sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[vtxbuf], label);
#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
        const AcReal* in      = device->vba.in[vtxbuf];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.out[vtxbuf];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    AcReal* arr = device->vba.out[vtxbuf];
#endif

    MPI_Datatype subarray;
    const int arr_nn[]     = {nn.z, nn.y, nn.x};
    const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
    const int arr_offset[] = {offset.z, offset.y, offset.x};

    // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i, nn_sub.x
    // %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
    //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y, offset.x);

    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                  MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);

    if (type == ACCESS_READ) {
        AcReal* in           = device->vba.out[vtxbuf];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

        AcReal* out           = device->vba.in[vtxbuf];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        cudaDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END
    }
    return AC_SUCCESS;
}

AcMeshInfo
acGridGetLocalMeshInfo(void)
{
    return grid.device->local_config;
}

AcResult
acGridReadVarfileToMesh(const char* file, const Field fields[], const size_t num_fields,
                        const int3 nn, const int3 rr)
{
    // Ensure the library state is ready
    ERRCHK_ALWAYS(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    // Derive the input mesh dimensions
    const int3 mm = (int3){
        nn.x + 2 * rr.x,
        nn.y + 2 * rr.y,
        nn.z + 2 * rr.z,
    };
    const size_t field_offset = (size_t)mm.x * (size_t)mm.y * (size_t)mm.z;

    // Set the helper variables
    const Device device         = grid.device;
    const AcMeshInfo info       = device->local_config;
    const int3 subdomain_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 subdomain_offset = info.int3_params[AC_multigpu_offset]; // Without halo
    int retval;

    // Load the fields to host memory
    MPI_Datatype subdomain;
    const int domain_mm_[]        = {mm.z, mm.y, mm.x};
    const int subdomain_nn_[]     = {subdomain_nn.z, subdomain_nn.y, subdomain_nn.x};
    const int subdomain_offset_[] = {
        rr.z + subdomain_offset.z,
        rr.y + subdomain_offset.y,
        rr.x + subdomain_offset.x,
    }; // Offset the ghost zone

    MPI_Type_create_subarray(3, domain_mm_, subdomain_nn_, subdomain_offset_, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &subdomain);
    MPI_Type_commit(&subdomain);

    MPI_File fp;
    retval = MPI_File_open(astaroth_comm, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
    ERRCHK_ALWAYS(retval == MPI_SUCCESS);

    for (size_t i = 0; i < num_fields; ++i) {
        const Field field = fields[i];

        // Load from file to host memory
        AcReal* host_buffer       = grid.submesh.vertex_buffer[field];
        const size_t displacement = i * field_offset * sizeof(AcReal); // Bytes

        retval = MPI_File_set_view(fp, displacement, AC_REAL_MPI_TYPE, subdomain, "native",
                                   MPI_INFO_NULL);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Status status;
        const size_t count = acVertexBufferCompdomainSize(info);
        // retval             = MPI_File_read_all(fp, host_buffer, count, AC_REAL_MPI_TYPE,
        // &status);
        retval = MPI_File_read(fp, host_buffer, count, AC_REAL_MPI_TYPE, &status); // workaround
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        /*
        for (size_t kk = 0; kk < subdomain_nn.z; ++kk) {
            for (size_t jj = 0; jj < subdomain_nn.y; ++jj) {
                for (size_t ii = 0; ii < subdomain_nn.x; ++ii) {
                    const size_t idx = ii + jj * subdomain_nn.x + kk * subdomain_nn.x *
        subdomain_nn.y; host_buffer[idx] = (ii+subdomain_offset.x) + (jj+subdomain_offset.y);
                }
            }
        }
        */

        // Load from host memory to device memory
        AcReal* in           = device->vba.out[field];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = subdomain_nn;

        AcReal* out           = device->vba.in[field];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info);
        cudaMemcpy(in, host_buffer, bytes, cudaMemcpyHostToDevice);
        retval = acDeviceVolumeCopy(device, field, in, in_offset, in_volume, out, out_offset,
                                    out_volume);
        ERRCHK_ALWAYS(retval == AC_SUCCESS);
    }
    acGridSynchronizeStream(STREAM_ALL);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);

    return AC_SUCCESS;
}

bool
acGridTaskGraphHasPeriodicBoundcondsX(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_X) != 0;
}

bool
acGridTaskGraphHasPeriodicBoundcondsY(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_Y) != 0;
}

bool
acGridTaskGraphHasPeriodicBoundcondsZ(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_Z) != 0;
}

/*
AcResult
acGridLoadFieldFromFile(const char* path, const VertexBufferHandle vtxbuf)
{
    ERRCHK(grid.initialized);

    acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;
    const int3 global_nn  = info.int3_params[AC_global_grid_n];
    const int3 global_mm  = (int3){
        2 * STENCIL_ORDER + global_nn.x,
        2 * STENCIL_ORDER + global_nn.y,
        2 * STENCIL_ORDER + global_nn.z,
    };
    const int3 local_nn         = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn_offset = info.int3_params[AC_multigpu_offset];

    MPI_Datatype subarray;
    const int mm[]     = {global_mm.z, global_mm.y, global_mm.x};
    const int nn_sub[] = {local_nn.z, local_nn.y, local_nn.x};
    const int offset[] = {
        STENCIL_ORDER + global_nn_offset.z,
        STENCIL_ORDER + global_nn_offset.y,
        STENCIL_ORDER + global_nn_offset.x,
    };
    MPI_Type_create_subarray(3, mm, nn_sub, offset, MPI_ORDER_C, AC_REAL_MPI_TYPE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;
    const int flags = MPI_MODE_RDONLY;
    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    AcReal* arr        = grid.submesh.vertex_buffer[vtxbuf];
    const size_t count = nn_sub[0] * nn_sub[1] * nn_sub[2];
    ERRCHK_ALWAYS(MPI_File_read_all(file, arr, count, AC_REAL_MPI_TYPE, &status) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);

    AcReal* in         = device->vba.out[vtxbuf]; // Note swapped order (vba.out)
    const size_t bytes = sizeof(in[0]) * count;
    cudaMemcpy(in, arr, bytes, cudaMemcpyHostToDevice);

    const int3 in_offset = (int3){0, 0, 0};
    const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

    AcReal* out           = device->vba.in[vtxbuf]; // Note swapped order (vba.in)
    const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                       out_volume);

    // Update halos
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    return AC_SUCCESS;
}

AcResult
acGridStoreFieldToFile(const char* path, const VertexBufferHandle vtxbuf)
{
    ERRCHK(grid.initialized);

    const Device device   = grid.device;
    const AcMeshInfo info = device->local_config;

    AcReal* in           = device->vba.in[vtxbuf];
    const int3 in_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 in_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    AcReal* out           = device->vba.out[vtxbuf];
    const int3 out_offset = (int3){0, 0, 0};
    const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

    acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                       out_volume);

    AcReal* arr        = grid.submesh.vertex_buffer[vtxbuf];
    const size_t bytes = sizeof(in[0]) * out_volume.x * out_volume.y * out_volume.z;
    cudaMemcpy(in, arr, bytes, cudaMemcpyHostToDevice);

    acGridDiskAccessSync();

    const int3 global_nn = info.int3_params[AC_global_grid_n];
    const int3 global_mm = (int3){
        2 * STENCIL_ORDER + global_nn.x,
        2 * STENCIL_ORDER + global_nn.y,
        2 * STENCIL_ORDER + global_nn.z,
    };
    const int3 local_nn         = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn_offset = info.int3_params[AC_multigpu_offset];

    MPI_Datatype subarray;
    const int mm[]     = {global_mm.z, global_mm.y, global_mm.x};
    const int nn_sub[] = {local_nn.z, local_nn.y, local_nn.x};
    const int offset[] = {
        STENCIL_ORDER + global_nn_offset.z,
        STENCIL_ORDER + global_nn_offset.y,
        STENCIL_ORDER + global_nn_offset.x,
    };
    MPI_Type_create_subarray(3, mm, nn_sub, offset, MPI_ORDER_C, AC_REAL_MPI_TYPE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;
    const int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t count = nn_sub[0] * nn_sub[1] * nn_sub[2];
    ERRCHK_ALWAYS(MPI_File_write_all(file, arr, count, AC_REAL_MPI_TYPE, &status) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
    return AC_SUCCESS;
}
*/

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridGeneralBoundconds(const Device device, const Stream stream)
{
    // Non-periodic Boundary conditions
    // Check the position in MPI frame
    int nprocs, pid;
    MPI_Comm_size(astaroth_comm, &nprocs);
    MPI_Comm_rank(astaroth_comm, &pid);
    const uint3_64 decomposition = decompose(nprocs);
    const int3 pid3d             = getPid3D(pid, decomposition);

    // Set outer boudaries after substep computation.
    const int3 m1 = (int3){0, 0, 0};
    const int3 m2 = grid.nn;
    const int3 pid3d = getPid3D(pid, decomposition);
    // If we are are a boundary element
    int3 bindex = (int3){0, 0, 0};

    // Check if there are active boundary condition edges.
    // 0 is no boundary, 1 both edges, 2 is top edge, 3 bottom edge
    if      ((pid3d.x == 0) && (pid3d.x == decomposition.x - 1)) { bindex.x = 1; }
    else if  (pid3d.x == 0)                                      { bindex.x = 2; }
    else if                    (pid3d.x == decomposition.x - 1)  { bindex.x = 3; }

    if      ((pid3d.y == 0) && (pid3d.y == decomposition.y - 1)) { bindex.y = 1; }
    else if  (pid3d.y == 0)                                      { bindex.y = 2; }
    else if                    (pid3d.y == decomposition.y - 1)  { bindex.y = 3; }

    if      ((pid3d.z == 0) && (pid3d.z == decomposition.z - 1)) { bindex.z = 1; }
    else if  (pid3d.z == 0)                                      { bindex.z = 2; }
    else if                    (pid3d.z == decomposition.z - 1)  { bindex.z = 3; }


    if (bindex.x != 1) && (bindex.y != 1) && (bindex.z != 1) {
        acDeviceGeneralBoundconds(device, stream, m1, m2, bindex);
    }
    acGridSynchronizeStream(stream);

    return AC_SUCCESS;
}
*/

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const Device device = grid.device;
    const int3 nn       = grid.nn;
#if MPI_INCL_CORNERS
    CommData corner_data = grid.corner_data; // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
    CommData edgex_data  = grid.edgex_data;
    CommData edgey_data  = grid.edgey_data;
    CommData edgez_data  = grid.edgez_data;
    CommData sidexy_data = grid.sidexy_data;
    CommData sidexz_data = grid.sidexz_data;
    CommData sideyz_data = grid.sideyz_data;

    acGridLoadScalarUniform(stream, AC_dt, dt);
    acDeviceSynchronizeStream(device, stream);


// Corners
#if MPI_INCL_CORNERS
    // Do not rm: required for corners
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){0, 0, NGHOST + nn.z},

        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };
#endif // MPI_INCL_CORNERS

    // Edges X
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges Y
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };

    // Edges Z
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    // Sides XY
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };

    // Sides XZ
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };

    // Sides YZ
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };

    for (int isubstep = 0; isubstep < 3; ++isubstep) {

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acPackCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                    // MPI_INCL_CORNERS
        acPackCommData(device, edgex_b0s, &edgex_data);
        acPackCommData(device, edgey_b0s, &edgey_data);
        acPackCommData(device, edgez_b0s, &edgez_data);
        acPackCommData(device, sidexy_b0s, &sidexy_data);
        acPackCommData(device, sidexz_b0s, &sidexz_data);
        acPackCommData(device, sideyz_b0s, &sideyz_data);
#endif

#if MPI_COMM_ENABLED
        MPI_Barrier(astaroth_comm);

#if MPI_GPUDIRECT_DISABLED
#if MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &corner_data); // Do not rm: required for corners
#endif                                                  // MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &edgex_data);
        acTransferCommDataToHost(device, &edgey_data);
        acTransferCommDataToHost(device, &edgez_data);
        acTransferCommDataToHost(device, &sidexy_data);
        acTransferCommDataToHost(device, &sidexz_data);
        acTransferCommDataToHost(device, &sideyz_data);
#endif
#if MPI_INCL_CORNERS
        acTransferCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                        // MPI_INCL_CORNERS
        acTransferCommData(device, edgex_b0s, &edgex_data);
        acTransferCommData(device, edgey_b0s, &edgey_data);
        acTransferCommData(device, edgez_b0s, &edgez_data);
        acTransferCommData(device, sidexy_b0s, &sidexy_data);
        acTransferCommData(device, sidexz_b0s, &sidexz_data);
        acTransferCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        //////////// INNER INTEGRATION //////////////
        {
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = nn;
            acKernelIntegrateSubstep(device->streams[STREAM_16], isubstep, m1, m2, device->vba);
        }
////////////////////////////////////////////
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acTransferCommDataWait(corner_data); // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
        acTransferCommDataWait(edgex_data);
        acTransferCommDataWait(edgey_data);
        acTransferCommDataWait(edgez_data);
        acTransferCommDataWait(sidexy_data);
        acTransferCommDataWait(sidexz_data);
        acTransferCommDataWait(sideyz_data);

#if MPI_INCL_CORNERS
        acUnpinCommData(device, &corner_data); // Do not rm: required for corners
#endif                                         // MPI_INCL_CORNERS
        acUnpinCommData(device, &edgex_data);
        acUnpinCommData(device, &edgey_data);
        acUnpinCommData(device, &edgez_data);
        acUnpinCommData(device, &sidexy_data);
        acUnpinCommData(device, &sidexz_data);
        acUnpinCommData(device, &sideyz_data);

#if MPI_INCL_CORNERS
        acUnpackCommData(device, corner_b0s, &corner_data);
#endif // MPI_INCL_CORNERS
        acUnpackCommData(device, edgex_b0s, &edgex_data);
        acUnpackCommData(device, edgey_b0s, &edgey_data);
        acUnpackCommData(device, edgez_b0s, &edgez_data);
        acUnpackCommData(device, sidexy_b0s, &sidexy_data);
        acUnpackCommData(device, sidexz_b0s, &sidexz_data);
        acUnpackCommData(device, sideyz_b0s, &sideyz_data);
//////////// OUTER INTEGRATION //////////////

// Wait for unpacking
#if MPI_INCL_CORNERS
        acSyncCommData(corner_data); // Do not rm: required for corners
#endif                               // MPI_INCL_CORNERS
        acSyncCommData(edgex_data);
        acSyncCommData(edgey_data);
        acSyncCommData(edgez_data);
        acSyncCommData(sidexy_data);
        acSyncCommData(sidexz_data);
        acSyncCommData(sideyz_data);
#endif // MPI_COMM_ENABLED

        // Invoke outer edge boundary conditions.
        acGridGeneralBoundconds(device, stream)

#if MPI_COMPUTE_ENABLED
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_0], isubstep, m1, m2, device->vba);
        }
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nn.z};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_1], isubstep, m1, m2, device->vba);
        }
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_2], isubstep, m1, m2, device->vba);
        }
        { // Top
            const int3 m1 = (int3){NGHOST, nn.y, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_3], isubstep, m1, m2, device->vba);
        }
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_4], isubstep, m1, m2, device->vba);
        }
        { // Right
            const int3 m1 = (int3){nn.x, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_5], isubstep, m1, m2, device->vba);
        }
#endif // MPI_COMPUTE_ENABLED
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
        ////////////////////////////////////////////

    }

    return AC_SUCCESS;
}
*/

#endif // AC_MPI_ENABLED
