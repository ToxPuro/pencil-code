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
#pragma once

#include "../acc-runtime/api/acc_runtime.h"
//#include "acc_runtime.h"
#if AC_MPI_ENABLED
#include <mpi.h>
#endif

#define NGHOST (STENCIL_ORDER / 2) // Astaroth 2.0 backwards compatibility

typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;

#define STREAM_0 (0)
#define STREAM_1 (1)
#define STREAM_2 (2)
#define STREAM_3 (3)
#define STREAM_4 (4)
#define STREAM_5 (5)
#define STREAM_6 (6)
#define STREAM_7 (7)
#define STREAM_8 (8)
#define STREAM_9 (9)
#define STREAM_10 (10)
#define STREAM_11 (11)
#define STREAM_12 (12)
#define STREAM_13 (13)
#define STREAM_14 (14)
#define STREAM_15 (15)
#define STREAM_16 (16)
#define STREAM_17 (17)
#define STREAM_18 (18)
#define STREAM_19 (19)
#define STREAM_20 (20)
#define STREAM_21 (21)
#define STREAM_22 (22)
#define STREAM_23 (23)
#define STREAM_24 (24)
#define STREAM_25 (25)
#define STREAM_26 (26)
#define STREAM_27 (27)
#define STREAM_28 (28)
#define STREAM_29 (29)
#define STREAM_30 (30)
#define STREAM_31 (31)
#define NUM_STREAMS (32)
#define STREAM_DEFAULT (STREAM_0)
#define STREAM_ALL (NUM_STREAMS)
typedef int Stream;

// For plate buffers.
enum {AC_H2D, AC_D2H};    // pack/unpack direction
typedef enum {AC_XY, AC_XZ, AC_YZ, AC_FRONT, AC_BACK, NUM_PLATE_BUFFERS} PlateType;

#define AC_FOR_RTYPES(FUNC)                                                                        \
    FUNC(RTYPE_MAX)                                                                                \
    FUNC(RTYPE_MIN)                                                                                \
    FUNC(RTYPE_SUM)                                                                                \
    FUNC(RTYPE_RMS)                                                                                \
    FUNC(RTYPE_RMS_EXP)                                                                            \
    FUNC(RTYPE_ALFVEN_MAX)                                                                         \
    FUNC(RTYPE_ALFVEN_MIN)                                                                         \
    FUNC(RTYPE_ALFVEN_RMS)

#define RTYPE_ISNAN (RTYPE_SUM)

#define AC_FOR_BCTYPES(FUNC)                                                                       \
    FUNC(BOUNDCOND_PERIODIC)                                                                       \
    FUNC(BOUNDCOND_SYMMETRIC)                                                                      \
    FUNC(BOUNDCOND_ANTISYMMETRIC)                                                                  \
    FUNC(BOUNDCOND_A2)                                                                             \
    FUNC(BOUNDCOND_PRESCRIBED_DERIVATIVE)

#ifdef AC_INTEGRATION_ENABLED

#define AC_FOR_SPECIAL_MHD_BCTYPES(FUNC)                                                           \
    FUNC(SPECIAL_MHD_BOUNDCOND_ENTROPY_CONSTANT_TEMPERATURE)                                       \
    FUNC(SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION)                                        \
    FUNC(SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX)                                       \
    FUNC(SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_NORMAL_AND_TURBULENT_HEAT_FLUX)

#endif

#define AC_FOR_INIT_TYPES(FUNC)                                                                    \
    FUNC(INIT_TYPE_RANDOM)                                                                         \
    FUNC(INIT_TYPE_AA_RANDOM)                                                                      \
    FUNC(INIT_TYPE_XWAVE)                                                                          \
    FUNC(INIT_TYPE_GAUSSIAN_RADIAL_EXPL)                                                           \
    FUNC(INIT_TYPE_ABC_FLOW)                                                                       \
    FUNC(INIT_TYPE_SIMPLE_CORE)                                                                    \
    FUNC(INIT_TYPE_KICKBALL)                                                                       \
    FUNC(INIT_TYPE_VEDGE)                                                                          \
    FUNC(INIT_TYPE_VEDGEX)                                                                         \
    FUNC(INIT_TYPE_RAYLEIGH_TAYLOR)                                                                \
    FUNC(INIT_TYPE_RAYLEIGH_BENARD)

#define AC_GEN_ID(X) X,

// Naming the associated number of the boundary condition types
typedef enum {
    AC_FOR_BCTYPES(AC_GEN_ID) //
    NUM_BCTYPES,
} AcBoundcond;
/*
typedef enum {
    AC_FOR_SCALARARRAY_HANDLES(AC_GEN_ID) //
    NUM_SCALARARRAY_HANDLES
} ScalarArrayHandle;
*/
#ifdef AC_INTEGRATION_ENABLED
typedef enum {
    AC_FOR_SPECIAL_MHD_BCTYPES(AC_GEN_ID) //
    NUM_SPECIAL_MHD_BCTYPES,
} AcSpecialMHDBoundcond;
#endif

typedef enum {
    AC_FOR_RTYPES(AC_GEN_ID) //
    NUM_RTYPES
} ReductionType;

typedef enum {
    AC_FOR_INIT_TYPES(AC_GEN_ID) //
    NUM_INIT_TYPES
} InitType;

#undef AC_GEN_ID

#define _UNUSED __attribute__((unused)) // Does not give a warning if unused
#define AC_GEN_STR(X) #X,
static const char* bctype_names[] _UNUSED       = {AC_FOR_BCTYPES(AC_GEN_STR) "-end-"};
static const char* rtype_names[] _UNUSED        = {AC_FOR_RTYPES(AC_GEN_STR) "-end-"};
static const char* initcondtype_names[] _UNUSED = {AC_FOR_INIT_TYPES(AC_GEN_STR) "-end-"};

#ifdef AC_INTEGRATION_ENABLED
static const char* special_bctype_names[] _UNUSED = {
    AC_FOR_SPECIAL_MHD_BCTYPES(AC_GEN_STR) "-end-"};
#endif

#undef AC_GEN_STR
#undef _UNUSED

typedef struct node_s* Node;
typedef struct device_s* Device;

typedef struct {
    int3 m;
    int3 n;
} GridDims;

typedef struct {
    int num_devices;
    Device* devices;

    GridDims grid;
    GridDims subgrid;
} DeviceConfiguration;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    return as_size_t(info.int_params[AC_mx]) * as_size_t(info.int_params[AC_my]) *
           as_size_t(info.int_params[AC_mz]);
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferSize(info);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info)
{
    return as_size_t(info.int_params[AC_nx]) * as_size_t(info.int_params[AC_ny]) *
           as_size_t(info.int_params[AC_nz]);
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}

static inline int3
acConstructInt3Param(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                     const AcMeshInfo info)
{
    return (int3){
        info.int_params[a],
        info.int_params[b],
        info.int_params[c],
    };
}

typedef struct {
    int3 n0, n1;
    int3 m0, m1;
} AcMeshDims;

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
    const int3 n0 = (int3){
        info.int_params[AC_nx_min],
        info.int_params[AC_ny_min],
        info.int_params[AC_nz_min],
    };
    const int3 n1 = (int3){
        info.int_params[AC_nx_max],
        info.int_params[AC_ny_max],
        info.int_params[AC_nz_max],
    };
    const int3 m0 = (int3){0, 0, 0};
    const int3 m1 = (int3){
        info.int_params[AC_mx],
        info.int_params[AC_my],
        info.int_params[AC_mz],
    };

    return (AcMeshDims){
        .n0 = n0,
        .n1 = n1,
        .m0 = m0,
        .m1 = m1,
    };
}

AcMeshInfo acGridDecomposeMeshInfo(const AcMeshInfo global_config);

AcMeshInfo acGridGetLocalMeshInfo(void);

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    return as_size_t(i) +                          //
           as_size_t(j) * info.int_params[AC_mx] + //
           as_size_t(k) * info.int_params[AC_mx] * info.int_params[AC_my];
}

static inline int3
acVertexBufferSpatialIdx(const size_t i, const AcMeshInfo info)
{
    const int3 mm = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    return (int3){
        (int)i % mm.x,
        ((int)i % (mm.x * mm.y)) / mm.x,
        (int)i / (mm.x * mm.y),
    };
}

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("[%s]: (%d, %d, %d)\n", int3param_names[i], config.int3_params[i].x,
               config.int3_params[i].y, config.int3_params[i].z);
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("[%s]: %g\n", realparam_names[i], (double)(config.real_params[i]));
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("[%s]: (%g, %g, %g)\n", real3param_names[i], (double)(config.real3_params[i].x),
               (double)(config.real3_params[i].y), (double)(config.real3_params[i].z));
}

/** Prints a list of boundary condition types */
static inline void
acQueryBCtypes(void)
{
    for (int i = 0; i < NUM_BCTYPES; ++i)
        printf("%s (%d)\n", bctype_names[i], i);
}

/** Prints a list of initial condition condition types */
static inline void
acQueryInitcondtypes(void)
{
    for (int i = 0; i < NUM_INIT_TYPES; ++i)
        printf("%s (%d)\n", initcondtype_names[i], i);
}

/** Prints a list of reduction types */
static inline void
acQueryRtypes(void)
{
    for (int i = 0; i < NUM_RTYPES; ++i)
        printf("%s (%d)\n", rtype_names[i], i);
}

/** Prints a list of int parameters */
static inline void
acQueryIntparams(void)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("%s (%d)\n", intparam_names[i], i);
}

/** Prints a list of int3 parameters */
static inline void
acQueryInt3params(void)
{
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("%s (%d)\n", int3param_names[i], i);
}

/** Prints a list of real parameters */
static inline void
acQueryRealparams(void)
{
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("%s (%d)\n", realparam_names[i], i);
}

/** Prints a list of real3 parameters */
static inline void
acQueryReal3params(void)
{
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("%s (%d)\n", real3param_names[i], i);
}

/** Prints a list of Scalar array handles */
/*
static inline void
acQueryScalarrays(void)
{
    for (int i = 0; i < NUM_REAL_ARRS_1D; ++i)
        printf("%s (%d)\n", realarr1D_names[i], i);
}
*/

/** Prints a list of vertex buffer handles */
static inline void
acQueryVtxbufs(void)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        printf("%s (%d)\n", vtxbuf_names[i], i);
}

/** Prints a list of kernels */
static inline void
acQueryKernels(void)
{
    for (int i = 0; i < NUM_KERNELS; ++i)
        printf("%s (%d)\n", kernel_names[i], i);
}

static inline void
acPrintIntParam(const AcIntParam a, const AcMeshInfo info)
{
    printf("%s: %d\n", intparam_names[a], info.int_params[a]);
}

static inline void
acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c, const AcMeshInfo info)
{
    acPrintIntParam(a, info);
    acPrintIntParam(b, info);
    acPrintIntParam(c, info);
}

static inline void
acPrintInt3Param(const AcInt3Param a, const AcMeshInfo info)
{
    const int3 vec = info.int3_params[a];
    printf("{%s: (%d, %d, %d)}\n", int3param_names[a], vec.x, vec.y, vec.z);
}

/*
 * =============================================================================
 * Legacy interface
 * =============================================================================
 */
/** Allocates all memory and initializes the devices visible to the caller. Should be
 * called before any other function in this interface. */
AcResult acInit(const AcMeshInfo mesh_info,int rank);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
 * AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Synchronizes a specific stream. All streams are synchronized if STREAM_ALL is passed as a
 * parameter*/
AcResult acSynchronizeStream(const Stream stream);

/** */
AcResult acSynchronizeMesh(void);

/** Loads a constant to the memories of the devices visible to the caller */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

/** Loads a constant to the memories of the devices visible to the caller */
AcResult acLoadVectorConstant(const AcReal3Param param, const AcReal3 value);

/** Loads an AcMesh to the devices visible to the caller */
AcResult acLoad(const AcMesh host_mesh);

/** Sets the whole mesh to some value */
AcResult acSetVertexBuffer(const VertexBufferHandle handle, const AcReal value);

/** Stores the AcMesh distributed among the devices visible to the caller back to the host*/
AcResult acStore(AcMesh* host_mesh);

// Loads a YZ-plate 
AcResult acLoadYZPlate(const int3 start, const int3 end, AcMesh* host_mesh, AcReal *yzPlateBuffer);
 
/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data. */
AcResult acIntegrate(const AcReal dt);

/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data.
 * Has customizable boundary conditions. */
AcResult acIntegrateGBC(const AcMeshInfo config, const AcReal dt);

/** Applies periodic boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStep(void);

/** Applies general outer boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStepGBC(const AcMeshInfo config);

/** Does a scalar reduction with the data stored in some vertex buffer */
AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle);

/** Does a vector reduction with vertex buffers where the vector components are (a, b, c) */
AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

/** Does a reduction for an operation which requires a vector and a scalar with vertex buffers
 *  * where the vector components are (a, b, c) and scalr is (d) */
AcReal acReduceVecScal(const ReductionType rtype, const VertexBufferHandle a,
                       const VertexBufferHandle b, const VertexBufferHandle c,
                       const VertexBufferHandle d);

/** Stores a subset of the mesh stored across the devices visible to the caller back to host memory.
 */
AcResult acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh);

/** Will potentially be deprecated in later versions. Added only to fix backwards compatibility with
 * PC for now.*/
AcResult acIntegrateStep(const int isubstep, const AcReal dt);
AcResult acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start,
                                   const int3 end);
AcResult acSynchronize(void);
AcResult acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices);

int acGetNumDevicesPerNode(void);

/** Returns the number of fields (vertexbuffer handles). */
size_t acGetNumFields(void);

/** Gets the field handle corresponding to a null-terminated `str` and stores the result in
 * `handle`.
 *
 * Returns AC_SUCCESS on success.
 * Returns AC_FAILURE if the field was not found and sets `handle` to SIZE_MAX.
 *
 * Example usage:
 * ```C
 * size_t handle;
 * AcResult res = acGetFieldHandle("VTXBUF_LNRHO", &handle);
 * if (res != AC_SUCCESS)
 *  fprintf(stderr, "Handle not found\n");
 * ```
 *  */
AcResult acGetFieldHandle(const char* field, size_t* handle);

/** */
Node acGetNode(void);

/*
 * =============================================================================
 * Grid interface
 * =============================================================================
 */

/**
Calls MPI_Init and creates a separate communicator for Astaroth procs with MPI_Comm_split, color =
666 Any program running in the same MPI process space must also call MPI_Comm_split with some color
!= 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI and creating a communicator.

Returns AC_FAILURE otherwise.
 */
AcResult ac_MPI_Init();

/**
Calls MPI_Init_thread with the provided thread_level and creates a separate communicator for
Astaroth procs with MPI_Comm_split, color = 666 Any program running in the same MPI process space
must also call MPI_Comm_split with some color != 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI with the requested thread level and creating a
communicator.

Returns AC_FAILURE otherwise.
 */
AcResult ac_MPI_Init_thread(int thread_level);

/**
Destroys the communicator and calls MPI_Finalize

/**
Returns the MPI communicator used by all Astaroth processes.

If MPI was initialized with MPI_Init* instead of ac_MPI_Init, this will return MPI_COMM_WORLD
 */
MPI_Comm acGridMPIComm();

/**
Initializes all available devices.

Must compile and run the code with MPI.

Must allocate exactly one process per GPU. And the same number of processes
per node as there are GPUs on that node.

Devices in the grid are configured based on the contents of AcMesh.
 */
AcResult acGridInit(const AcMeshInfo info);

/**
Resets all devices on the current grid.
 */
AcResult acGridQuit(void);

/** Get the local device */
Device acGridGetDevice(void);

/** Get Vertexbuffer pointers on grid device*/
AcResult acGridGetVBApointers(AcReal *vbapointer[2]);

/** Randomizes the local mesh */
AcResult acGridRandomize(void);

/** */
AcResult acGridSynchronizeStream(const Stream stream);

/** */
AcResult acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value);

/** */
AcResult acGridLoadVectorUniform(const Stream stream, const AcReal3Param param,
                                 const AcReal3 value);

/** */
AcResult acGridLoadIntUniform(const Stream stream, const AcIntParam param, const int value);

/** */
AcResult acGridLoadInt3Uniform(const Stream stream, const AcInt3Param param, const int3 value);

/** */
AcResult acGridLoadMesh(const Stream stream, const AcMesh host_mesh);

/** */
AcResult acGridStoreMesh(const Stream stream, AcMesh* host_mesh);

/** */
AcResult acGridIntegrate(const Stream stream, const AcReal dt);

AcResult acGridSwapBuffers(void);

/** */
/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)

AcResult acGridIntegrateNonperiodic(const Stream stream, const AcReal dt);
*/

/** */
AcResult acGridPeriodicBoundconds(const Stream stream);

/** */
AcResult acGridGeneralBoundconds(const Device device, const Stream stream);

/** */
AcResult acGridReduceScal(const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** */
AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);

/** */
AcResult acGridReduceVecScal(const Stream stream, const ReductionType rtype,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result);

typedef enum {
    ACCESS_READ,
    ACCESS_WRITE,
} AccessType;

AcResult acGridAccessMeshOnDiskSynchronous(const VertexBufferHandle field, const char* dir,
                                           const char* label, const AccessType type);

AcResult acGridDiskAccessLaunch(const AccessType type);

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
AcResult acGridWriteSlicesToDiskLaunch(const char* dir, const char* label);

/* Synchronous */
AcResult acGridWriteSlicesToDiskCollectiveSynchronous(const char* dir, const char* label);

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
AcResult acGridWriteMeshToDiskLaunch(const char* dir, const char* label);

AcResult acGridDiskAccessSync(void);

AcResult acGridReadVarfileToMesh(const char* file, const Field fields[], const size_t num_fields,
                                 const int3 nn, const int3 rr);

/* Quick hack for the hero run, will be removed in future builds */
AcResult acGridAccessMeshOnDiskSynchronousDistributed(const VertexBufferHandle vtxbuf,
                                                      const char* dir, const char* label,
                                                      const AccessType type);

/* Quick hack for the hero run, will be removed in future builds */
AcResult acGridAccessMeshOnDiskSynchronousCollective(const VertexBufferHandle vtxbuf,
                                                     const char* dir, const char* label,
                                                     const AccessType type);

// Bugged
// AcResult acGridLoadFieldFromFile(const char* path, const VertexBufferHandle field);

// Bugged
// AcResult acGridStoreFieldToFile(const char* path, const VertexBufferHandle field);

/*
 * =============================================================================
 * Task interface (part of the grid interface)
 * =============================================================================
 */

/** */
typedef enum AcTaskType {
    TASKTYPE_COMPUTE,
    TASKTYPE_HALOEXCHANGE,
    TASKTYPE_BOUNDCOND,
    TASKTYPE_SPECIAL_MHD_BOUNDCOND
} AcTaskType;

typedef enum AcBoundary {
    BOUNDARY_NONE  = 0,
    BOUNDARY_X_TOP = 0x01,
    BOUNDARY_X_BOT = 0x02,
    BOUNDARY_X     = BOUNDARY_X_TOP | BOUNDARY_X_BOT,
    BOUNDARY_Y_TOP = 0x04,
    BOUNDARY_Y_BOT = 0x08,
    BOUNDARY_Y     = BOUNDARY_Y_TOP | BOUNDARY_Y_BOT,
    BOUNDARY_Z_TOP = 0x10,
    BOUNDARY_Z_BOT = 0x20,
    BOUNDARY_Z     = BOUNDARY_Z_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_XY    = BOUNDARY_X | BOUNDARY_Y,
    BOUNDARY_XZ    = BOUNDARY_X | BOUNDARY_Z,
    BOUNDARY_YZ    = BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_XYZ   = BOUNDARY_X | BOUNDARY_Y | BOUNDARY_Z
} AcBoundary;

/** TaskDefinition is a datatype containing information necessary to generate a set of tasks for
 * some operation.*/
typedef struct AcTaskDefinition {
    AcTaskType task_type;
    union {
        AcKernel kernel;
        AcBoundcond bound_cond;
#ifdef AC_INTEGRATION_ENABLED
        AcSpecialMHDBoundcond special_mhd_bound_cond;
#endif
    };
    AcBoundary boundary;

    Field* fields_in;
    size_t num_fields_in;

    Field* fields_out;
    size_t num_fields_out;

    AcRealParam* parameters;
    size_t num_parameters;
    int shell_num;
    int step_number;
} AcTaskDefinition;

/** TaskGraph is an opaque datatype containing information necessary to execute a set of
 * operations.*/
typedef struct AcTaskGraph AcTaskGraph;

/** */
AcTaskDefinition acCompute(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out, int shell_num, int step_number);

/** */
AcTaskDefinition acHaloExchange(Field fields[], const size_t num_fields);

/** */
AcTaskDefinition acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                                     Field fields[], const size_t num_fields,
                                     AcRealParam parameters[], const size_t num_parameters);

#ifdef AC_INTEGRATION_ENABLED
/** SpecialMHDBoundaryConditions are tied to some specific DSL implementation (At the moment, the
   MHD implementation). They launch specially written CUDA kernels that implement the specific
   boundary condition procedure They are a stop-gap temporary solution. The sensible solution is to
   replace them with a task type that runs a boundary condition procedure written in the Astaroth
   DSL.
*/
// AcTaskDefinition acSpecialMHDBoundaryCondition(const AcBoundary boundary,
//                                                const AcSpecialMHDBoundcond bound_cond,
//                                                AcRealParam parameters[],
//                                                AcRealParam parameters[],
//                                                AcRealParam parameters[],
//                                                const size_t num_parameters);
#endif

/** */
AcTaskGraph* acGridGetDefaultTaskGraph();

/** */
bool acGridTaskGraphHasPeriodicBoundcondsX(AcTaskGraph* graph);

/** */
bool acGridTaskGraphHasPeriodicBoundcondsY(AcTaskGraph* graph);

/** */
bool acGridTaskGraphHasPeriodicBoundcondsZ(AcTaskGraph* graph);

/** */
AcTaskGraph* acGridBuildTaskGraph(const AcTaskDefinition ops[], const size_t n_ops);

/** */
AcResult acGridDestroyTaskGraph(AcTaskGraph* graph);

/** */
AcResult acGridExecuteTaskGraph(AcTaskGraph* graph, const size_t n_iterations);

/** */
AcResult acGridLaunchKernel(const Stream stream, const Kernel kernel, const int3 start,
                            const int3 end);

/** */
AcResult acGridLoadStencil(const Stream stream, const Stencil stencil,
                           const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

/** */
AcResult acGridStoreStencil(const Stream stream, const Stencil stencil,
                            AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

/** */
AcResult
acGridLoadStencils(const Stream stream,
                   const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

/** */
AcResult
acGridStoreStencils(const Stream stream,
                    AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);


/*
 * =============================================================================
 * Node interface
 * =============================================================================
 */
/**
Initializes all devices on the current node.

Devices on the node are configured based on the contents of AcMesh.

@return Exit status. Places the newly created handle in the output parameter.
@see AcMeshInfo


Usage example:
@code
AcMeshInfo info;
acLoadConfig(AC_DEFAULT_CONFIG, &info);

Node node;
acNodeCreate(0, info, &node);
acNodeDestroy(node);
@endcode
 */
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node, int rank);

/**
Resets all devices on the current node.

@see acNodeCreate()
 */
AcResult acNodeDestroy(Node node);

/**
Prints information about the devices available on the current node.

Requires that Node has been initialized with
@See acNodeCreate().
*/
AcResult acNodePrintInfo(const Node node);

/**



@see DeviceConfiguration
*/
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);

/** */
AcResult acNodeAutoOptimize(const Node node);

/** */
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);

/** Deprecated ? */
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle); // Not in Device

/** */
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream); // Not in Device

/** */
AcResult acNodeSwapBuffers(const Node node);

/** */
AcResult acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value);

/** Deprecated ? Might be useful though if the user wants to load only one vtxbuf. But in this case
 * the user should supply a AcReal* instead of vtxbuf_handle */
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);
/** */
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);
/** Deprecated ? */
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acNodeSetVertexBuffer(const Node node, const Stream stream,
                               const VertexBufferHandle handle, const AcReal value);

/** Deprecated ? */
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);

/** */
AcResult acNodeIntegrate(const Node node, const AcReal dt);

/** */
AcResult acNodeIntegrateGBC(const Node node, const AcMeshInfo config, const AcReal dt);

/** */
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);

/** */
AcResult acNodeGeneralBoundcondStep(const Node node, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle,
                                    const AcMeshInfo config);

/** */
AcResult acNodeGeneralBoundconds(const Node node, const Stream stream, const AcMeshInfo config);

/** */
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
/** */
AcResult acNodeReduceVecScal(const Node node, const Stream stream_type, const ReductionType rtype,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result);
/** */
AcResult acNodeLoadPlate(const Node node, const Stream stream, const int3 start, const int3 end, 
                         AcMesh* host_mesh, AcReal* plateBuffer, PlateType plate);
/** */
AcResult acNodeStorePlate(const Node node, const Stream stream, const int3 start, const int3 end,
                          AcMesh* host_mesh, AcReal* plateBuffer, PlateType plate);
/** */
AcResult acNodeStoreIXYPlate(const Node node, const Stream stream, const int3 start, const int3 end, 
                             AcMesh* host_mesh, PlateType plate);
/** */
AcResult acNodeLoadPlateXcomp(const Node node, const Stream stream, const int3 start, const int3 end, 
                              AcMesh* host_mesh, AcReal* plateBuffer, PlateType plate);

/** */
AcResult acNodeGetVBApointers(Node* node_handle, AcReal *vbapointer[2]);

/*
 * =============================================================================
 * Device interface
 * =============================================================================
 */
/** */
AcResult acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device);

/** */
AcResult acDeviceDestroy(Device device);

/** */
AcResult acDevicePrintInfo(const Device device);

/** */
// AcResult acDeviceAutoOptimize(const Device device);

/** */
AcResult acDeviceSynchronizeStream(const Device device, const Stream stream);

/** */
AcResult acDeviceSwapBuffer(const Device device, const VertexBufferHandle handle);

/** */
AcResult acDeviceSwapBuffers(const Device device);

/** */
AcResult acDeviceLoadScalarUniform(const Device device, const Stream stream,
                                   const AcRealParam param, const AcReal value);

/** */
AcResult acDeviceLoadVectorUniform(const Device device, const Stream stream,
                                   const AcReal3Param param, const AcReal3 value);

/** */
AcResult acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                                const int value);

/** */
AcResult acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                                 const int3 value);

/** */
AcResult acDeviceStoreScalarUniform(const Device device, const Stream stream,
                                    const AcRealParam param, AcReal* value);

/** */
AcResult acDeviceStoreVectorUniform(const Device device, const Stream stream,
                                    const AcReal3Param param, AcReal3* value);

/** */
AcResult acDeviceStoreIntUniform(const Device device, const Stream stream, const AcIntParam param,
                                 int* value);

/** */
AcResult acDeviceStoreInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                                  int3* value);

/** */
/*
AcResult acDeviceLoadScalarArray(const Device device, const Stream stream,
                                 const ScalarArrayHandle handle, const size_t start,
                                 const AcReal* data, const size_t num);
                                 */

/** */
AcResult acDeviceLoadMeshInfo(const Device device, const AcMeshInfo device_config);

/** */
AcResult acDeviceLoadDefaultUniforms(const Device device);

/** */
AcResult acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices);

/** Deprecated */
AcResult acDeviceLoadMeshWithOffset(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices);

/** */
AcResult acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acDeviceSetVertexBuffer(const Device device, const Stream stream,
                                 const VertexBufferHandle handle, const AcReal value);

/** */
AcResult acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                             const VertexBufferHandle vtxbuf_handle, const int3 src,
                                             const int3 dst, const int num_vertices,
                                             AcMesh* host_mesh);

/** Deprecated */
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device);

/** Deprecated */
AcResult acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device);

/** */
AcResult acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device);

/** */
AcResult acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device);

/** */
AcResult acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                                  const int3 start, const int3 end, const AcReal dt);
/** */
AcResult acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end);

/** */
AcResult acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                                    const int3 end);

/** */
AcResult acDeviceGeneralBoundcondStep(const Device device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, const int3 start,
                                      const int3 end, const AcMeshInfo config, const int3 bindex);

/** */
AcResult acDeviceGeneralBoundconds(const Device device, const Stream stream, const int3 start,
                                   const int3 end, const AcMeshInfo config, const int3 bindex);

/** */
AcResult acDeviceReduceScalNotAveraged(const Device device, const Stream stream,
                                       const ReductionType rtype,
                                       const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** */
AcResult acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** */
AcResult acDeviceReduceVecNotAveraged(const Device device, const Stream stream_type,
                                      const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                                      const VertexBufferHandle vtxbuf1,
                                      const VertexBufferHandle vtxbuf2, AcReal* result);

/** */
AcResult acDeviceReduceVec(const Device device, const Stream stream_type, const ReductionType rtype,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result);

/** */
AcResult acDeviceReduceVecScalNotAveraged(const Device device, const Stream stream_type,
                                          const ReductionType rtype,
                                          const VertexBufferHandle vtxbuf0,
                                          const VertexBufferHandle vtxbuf1,
                                          const VertexBufferHandle vtxbuf2,
                                          const VertexBufferHandle vtxbuf3, AcReal* result);

/** */
AcResult acDeviceReduceVecScal(const Device device, const Stream stream_type,
                               const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                               const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2,
                               const VertexBufferHandle vtxbuf3, AcReal* result);
/** */
AcResult acDeviceRunMPITest(void);

/** */
AcResult acDeviceLaunchKernel(const Device device, const Stream stream, const Kernel kernel,
                              const int3 start, const int3 end);

/** */
AcResult acDeviceLoadStencil(const Device device, const Stream stream, const Stencil stencil,
                             const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

/** */
AcResult acDeviceStoreStencil(const Device device, const Stream stream, const Stencil stencil,
                              AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

/** */
AcResult acDeviceVolumeCopy(const Device device, const Stream stream,
                            const AcReal* in, const int3 in_offset, const int3 in_volume,
                            AcReal* out, const int3 out_offset, const int3 out_volume);

/** */
AcResult acDeviceLoadPlateBuffer(const Device device, int3 start, int3 end, const Stream stream,
                                 AcReal* buffer, PlateType plate);

/** */
AcResult acDeviceStorePlateBuffer(const Device device, int3 start, int3 end, const Stream stream, 
                                  AcReal* buffer, PlateType plate);

/** */
AcResult acDeviceStoreIXYPlate(const Device device, int3 start, int3 end, int src_offset, const Stream stream, 
                               AcMesh *host_mesh);

/** */
AcResult acDeviceGetVBApointers(Device device, AcReal *vbapointer[2]);

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
/** Updates the built-in parameters based on nx, ny and nz */
AcResult acHostUpdateBuiltinParams(AcMeshInfo* config);

/** Creates a mesh stored in host memory */
AcResult acHostMeshCreate(const AcMeshInfo mesh_info, AcMesh* mesh);

/** Randomizes a host mesh */
AcResult acHostMeshRandomize(AcMesh* mesh);

/** Destroys a mesh stored in host memory */
AcResult acHostMeshDestroy(AcMesh* mesh);

/** Sets the dimensions of the computational domain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */
AcResult acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
/** Backwards compatible interface, input fields = output fields*/
template <size_t num_fields>
AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields)[num_fields], int shell_num = 0, int step_number = -1)
{
    return acCompute(kernel, fields, num_fields, fields, num_fields, shell_num, step_number);
}

template <size_t num_fields_in, size_t num_fields_out>
AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out], int shell_num = 0, int step_number = -1)
{
    return acCompute(kernel, fields_in, num_fields_in, fields_out, num_fields_out, shell_num, step_number);
}

/** */
template <size_t num_fields>
AcTaskDefinition
acHaloExchange(Field (&fields)[num_fields])
{
    return acHaloExchange(fields, num_fields);
}

/** */
template <size_t num_fields>
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                    Field (&fields)[num_fields])
{
    return acBoundaryCondition(boundary, bound_cond, fields, num_fields, nullptr, 0);
}

/** */
template <size_t num_fields, size_t num_parameters>
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                    Field (&fields)[num_fields], AcRealParam (&parameters)[num_parameters])
{
    return acBoundaryCondition(boundary, bound_cond, fields, num_fields, parameters,
                               num_parameters);
}

#ifdef AC_INTEGRATION_ENABLED
/** */
AcTaskDefinition acSpecialMHDBoundaryCondition(const AcBoundary boundary,
                                               const AcSpecialMHDBoundcond bound_cond);

/** */
template <size_t num_parameters>
AcTaskDefinition
acSpecialMHDBoundaryCondition(const AcBoundary boundary, const AcSpecialMHDBoundcond bound_cond,
                              AcRealParam (&parameters)[num_parameters])
{
    return acSpecialMHDBoundaryCondition(boundary, bound_cond, parameters, num_parameters);
}

#endif

/** */
template <size_t n_ops>
AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition (&ops)[n_ops])
{
    return acGridBuildTaskGraph(ops, n_ops);
}

/** */
AcResult acGridSetDomainDecomposition(const int3 decomposition);
#endif
