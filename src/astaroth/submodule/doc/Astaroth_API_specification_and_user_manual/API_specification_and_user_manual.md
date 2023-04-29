# Astaroth Specification and User Manual

Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

	   Astaroth is free software: you can redistribute it and/or modify
	   it under the terms of the GNU General Public License as published by
	   the Free Software Foundation, either version 3 of the License, or
	   (at your option) any later version.
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


# Introduction and Background

Astaroth is a collection of tools for utilizing multiple graphics processing units (GPUs)
efficiently in three-dimensional stencil computations. This document specifies the Astaroth
application-programming interface (API) and domain-specific language (DSL).

Astaroth has been designed for the demands in computational sciences, where large stencils are
often used to attain sufficient accuracy. The majority of previous work focuses on stencil
computations with low-order stencils for which several efficient algorithms have been proposed,
whereas work on high-order stencils is more limited. In addition, in computational physics multiple
fields interact with each other, such as the velocity and magnetic fields of electrically
conducting fluids. Such computations are especially challenging to solve efficiently because of the
problem's relatively low operational intensity and the small caches provided by GPUs. Efficient
methods for computations with several coupled fields and large stencils have not been addressed
sufficiently in prior work.

With Astaroth, we have taken inspiration of image processing and graphics pipelines which rely on
holding intermediate data in caches for the duration of computations, and extended the idea to work
efficiently also with large three-dimensional stencils and an arbitrary number of coupled fields.
As programming GPUs efficiently is relatively verbose and requires deep knowledge of the underlying
hardware and execution model, we have created a high-level domain-specific language for expressing
a wide range of tasks in computational sciences and provide a source-to-source compiler for
translating stencil problems expressed in our language into efficient CUDA kernels.

The kernels generated from the Astaroth DSL are embedded in the Astaroth Core library, which is
usable via the Astaroth API. While the Astaroth library is written in C++/CUDA, the API conforms to
the C99 standard.


## Publications

The foundational work was done in (Väisälä, Pekkilä, 2017) and the library, API and DSL described
in this document were introduced in (Pekkilä, 2019). We kindly wish the users of Astaroth to cite
to these publications in their work.

> [Väisälä, M., Pekkilä, J., Käpylä, M., Rheinhardt, M., Shang, H., & Krasnopolsky, R. “Interaction of Large- and Small-Scale Dynamos in Isotropic Turbulent Flows from GPU-Accelerated Simulations.” The Astrophysical Journal, vol. 907, no. 2, Feb. 2021, p. 83.](https://doi.org/10.3847/1538-4357/abceca)

> [Pekkilä, J. Astaroth: A Library for Stencil Computations on Graphics Processing Units. Master's thesis, Aalto University School of Science, Espoo, Finland, 2019.](http://urn.fi/URN:NBN:fi:aalto-201906233993)

> [Väisälä, M. Magnetic Phenomena of the Interstellar Medium in Theory and Observation. PhD thesis, University of Helsinki, Finland, 2017.](http://urn.fi/URN:ISBN:978-951-51-2778-5)

> [Pekkilä, J., Väisälä, M., Käpylä, M., Käpylä, P. J., and Anjum, O. “Methods for compressible fluid simulation on GPUs using high-order finite differences, ”Computer Physics Communications, vol. 217, pp. 11–22, Aug. 2017.](https://doi.org/10.1016/j.cpc.2017.03.011)



# Astaroth API

The Astroth application-programming interface (API) provides the means for controlling execution of
user-defined and built-in functions on multiple graphics processing units. Functions in the API are
prefixed with lower case `ac`, while structures and data types are prefixed with capitalized
`Ac`. Compile-time constants, such as definitions and enumerations, have the prefix `AC_`.
All of the API functions return an AcResult value indicating either success or failure. The return
codes are
```C
typedef enum {
    AC_SUCCESS = 0,
    AC_FAILURE = 1
} AcResult;
```

The API is divided into layers which differ in the level of control provided over the execution.
There are two primary layers:

* Device layer
    * Provides control over a single GPU.
    * Functions start with `acDevice`.
    * All functions are asynchronous and managed using Streams.

* Grid layer
    * Provides control over all devices on multiple node. Uses MPI.
    * Functions start with `acGrid`.
    * `MPI_Init()` must be called before calling any acGrid functions.
    * Streams are used to control concurrency the same way as on the acDevice layer.

Finally, two additional layers are provided for convenience and backwards compatibility.

* Node layer (backwards compatibility)
    * Provides control over multiple devices in a single node.
    * Functions start with `acNode`.
    * All functions are asynchronous and executed concurrently on all devices in the node.
    * Subsequent functions called in the same stream (see Section #Streams and synchronization) are guaranteed to be synchronous.
    * For machines without CUDA-aware MPI support
    * New applications should opt for the Grid layer instead if possible

* Astaroth layer (deprecated)
    * Very high-level interace to Astaroth without concurrency control
    * Functions start with `ac` only, f.ex. acInit().
    * Provided for backwards compatibility.
    * Essentially a wrapper for the Node layer.
    * All functions are guaranteed to be synchronous.
    * At the moment, `samples/standalone/` is using the Astaroth/Node layer, but it will be phased out with `samples/standalone_mpi/` during 2021.  

There are also several helper functions defined in `include/astaroth.h`, which can be used for, say, determining the size or performing index calculations within the simulation domain.


## List of Astaroth API Functions

Here's a non-exhaustive list of astaroth API functions. For more info and an up-to-date list, see headers `include/astaroth.h` and `include/astaroth_utils.h`.

### Initialization, Quitting and Helper Functions

Device layer.
```C
AcResult acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device);
AcResult acDeviceDestroy(Device device);
AcResult acDevicePrintInfo(const Device device);
AcResult acDeviceAutoOptimize(const Device device);
```

Node layer.
```C
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);
AcResult acNodeDestroy(Node node);
AcResult acNodePrintInfo(const Node node);
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);
AcResult acNodeAutoOptimize(const Node node);
```

Grid layer.
```C
AcResult acGridInit(const AcMeshInfo info);
AcResult acGridQuit(void);
```

General helper functions.
```C
size_t acVertexBufferSize(const AcMeshInfo info);
size_t acVertexBufferSizeBytes(const AcMeshInfo info);
size_t acVertexBufferCompdomainSize(const AcMeshInfo info);
size_t acVertexBufferCompdomainSizeBytes(const AcMeshInfo info);
size_t acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info);
```

### Loading and Storing

Loading meshes and vertex buffers to device memory.
```C
AcResult acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh);
AcResult acDeviceLoadMeshWithOffset(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices);
AcResult acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle);
AcResult acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices);

AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);
AcResult acGridLoadMesh(const Stream stream, const AcMesh host_mesh);
```

Storing meshes and vertex buffer to host memory.
```C
AcResult acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh);
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);
AcResult acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);

AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);
AcResult acGridStoreMesh(const Stream stream, AcMesh* host_mesh);
```

Transferring data between devices
```C
AcResult acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device);
AcResult acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device);
AcResult acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device);
AcResult acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device);
```

Loading uniforms (device constants)
```C
AcResult acDeviceLoadScalarUniform(const Device device, const Stream stream,
                                    const AcRealParam param, const AcReal value);
AcResult acDeviceLoadVectorUniform(const Device device, const Stream stream,
                                    const AcReal3Param param, const AcReal3 value);
AcResult acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                                 const int value);
AcResult acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                                  const int3 value);
AcResult acDeviceLoadScalarArray(const Device device, const Stream stream,
                                 const ScalarArrayHandle handle, const AcReal* data,
                                 const size_t num);
AcResult acDeviceLoadMeshInfo(const Device device, const Stream stream,
                              const AcMeshInfo device_config);
```

### Integration, Reductions and Boundary Conditions

The library provides the following functions for integration, reductions and computing periodic
boundary conditions.
```C
AcResult acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                                  const int3 start, const int3 end, const AcReal dt);
AcResult acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end);
AcResult acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                                    const int3 end);
AcResult acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result);
AcResult acDeviceReduceVec(const Device device, const Stream stream_type, const ReductionType rtype,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result);

AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);
AcResult acNodeIntegrate(const Node node, const AcReal dt);
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
AcResult acGridIntegrate(const Stream stream, const AcReal dt);
AcResult acGridPeriodicBoundconds(const Stream stream);
AcResult acGridReduceScal(const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
```

Finally, kernels generated from DSL sources can be launched manually by using
the acc-runtime API function
```C
AcResult acLaunchKernel(Kernel func, const cudaStream_t stream,
                        const int3 start, const int3 end,
                        VertexBufferArray vba);
```
For example, a kernel named `kernel` defined in the DSL code can be launched by calling
`acLaunchKernel(dsl_kernel, ...)`.

## Stream Synchronization

All library functions that take a `Stream` as a parameter are asynchronous. When calling these
functions, control returns immediately back to the host even if the called device function has not
yet completed. Therefore special care must be taken in order to ensure proper synchronization.

Synchronization is done using `Stream` primitives, defined as
```C
typedef enum {STREAM_0, ..., STREAM_15, NUM_STREAMS};
#define STREAM_DEFAULT (STREAM_0)
#define STREAM_ALL (NUM_STREAMS)
```

> **Note:** There is guaranteed to be at least 16 distinct streams.

Functions queued in the same stream will be executed sequentially. If two or more consequent
functions are queued in different streams, then these functions may execute in parallel. For
additional control over streams, there is a barrier synchronization function which blocks execution
until all functions in the specified streams have completed. The Astaroth API provides barrier
synchronization with functions `acDeviceSynchronize` and `acNodeSynchronize`. All streams can be
synchronized at once by passing the alias `STREAM_ALL` to the synchronization function.

Usage of streams is demonstrated with the following example.
```C
funcA(STREAM_0);
funcB(STREAM_0);                      // Blocks until funcA has completed
funcC(STREAM_1);                      // May execute in parallel with funcB
barrierSynchronizeStream(STREAM_ALL); // Blocks until functions in all streams have completed
funcD(STREAM_2);                      // Is started when command returns from synchronizeStream()
```

Astaroth API provides the following functions for barrier synchronization.
```C
AcResult acSynchronize(void);
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);
AcResult acDeviceSynchronizeStream(const Device device, const Stream stream);
AcResult acGridSynchronizeStream(const Stream stream);
```

## Data Synchronization

Stream synchronization works in the same fashion on node and device layers. However on the node
layer, one has to take in account that a portion of the mesh is shared between devices and that the
data is always up to date.

In stencil computations, the mesh is surrounded by a halo, where data is only used for updating grid
points near the boundaries of the simulation domain. A portion of this halo is shared by neighboring
devices. As there is no way of knowing when the user has completed operations on the mesh, the data
communication between neighboring devices must be explicitly triggered. For this purpose, we provide
the functions
```C
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream);
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle);

```

> **Note:** Local halos must be up to date before synchronizing the data. Local halos are the grid points outside the computational domain which are used only by a single device. The mesh is distributed to multiple devices by blocking along the z axis. If there are *n* devices and the z-dimension of the computational domain is *nz*, then each device is assigned *nz / n* two-dimensional planes. For example with two devices, the data block that has to be up to date ranges from *(0, 0, nz)* to *(mx, my, nz + 2 * NGHOST)*.

## Input and Output Buffers

The mesh is duplicated to input and output buffers for performance reasons. The input buffers are
read-only in user-specified compute kernels, which allows us to read them via the texture cache
optimized for spatially local memory accesses. The results of compute kernels are written into the
output buffers.

Since we allow the user to operate on subsets of the computational domain in user-specified
kernels, we have no way of knowing when the output buffers are complete and can be swapped.
Therefore the user must explicitly state when the input and output buffer should be swapped. This
is done via the API calls
```C
AcResult acDeviceSwapBuffer(const Device device, const VertexBufferHandle handle);
AcResult acDeviceSwapBuffers(const Device device);
AcResult acNodeSwapBuffers(const Node node);
```
> **Note:** All functions provided with the API operate on input buffers and ensure that the complete result is available in the input buffer when the function has completed. User-specified kernels are exceptions and write the result to output buffers. Therefore buffers have to be swapped only after calling user-specified kernels.

## Devices

`Device` is a handle to some single device and is used in device layer functions to specify which
device should execute the function. A `Device` is created and destroyed with the following
interface functions.
```C
AcResult acDeviceCreate(const int device_id, const AcMeshInfo device_config, Device* device);
AcResult acDeviceDestroy(Device device);
```

## Nodes

`Node` is a handle to some compute node which consists of multiple devices. The `Node` handle is
used to specify which node the node layer functions should operate in. A node is created and
destroyed with the following interface functions.
```C
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);
AcResult acNodeDestroy(Node node);
```

The function acNodeCreate calls acDeviceCreate for all devices that are visible from the current
process. After a node has been created, the devices in it can be retrieved with the function
```C
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);
```
where DeviceConfiguration is defined as
```C
typedef struct {
    int num_devices;
    Device* devices; // Todo make this predefined s.t. the user/us do not have to alloc/free

    Grid grid;
    Grid subgrid;
} DeviceConfiguration;
```

See Section **Decomposition** for discussion about `Grid`.

## Meshes

Meshes are the primary structures for passing information to the library and kernels. The definition
of a `Mesh` is declared as
```C
typedef struct {
    int     int_params[NUM_INT_PARAMS];
    int3    int3_params[NUM_INT3_PARAMS];
    AcReal  real_params[NUM_REAL_PARAMS];
    AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

typedef struct {
    AcReal*    vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;
```

`AcMeshInfo` holds all the parameters required to execute DSL kernels. The following built-in parameters are available:

| Type | Built-in parameter | Description | Required |
| ---- | --------- | ----------- | -------- |
| int | AC_nx | Number of cells in the computational domain (x-axis) | YES |
| int | AC_ny | Number of cells in the computational domain (y-axis) | YES |
| int | AC_nz | Number of cells in the computational domain (z-axis) | YES |
| int | AC_bc_type_[**bot**⎮**top**]_[**x**⎮**y**⎮**z**] | Boundary conditions used in each ghost zone segment. `top` and `bot` indicate the segments at high and low indices on the selected axis, respectively. | NO |
| real | AC_dt | The timestep length when using the built-in RK3 integrator | NO |


Before passing `AcMeshInfo` into API functions, at least `AC_nx, AC_ny, AC_nz`
must be set, others are optional. Astaroth will warn about possibly uninitialized
parameters when running the code. Initialization is demonstrated in the following
example:
```C
#include "astaroth.h"

AcMeshInfo info;
info.int_params[AC_nx] = 128;
info.int_params[AC_ny] = 64;
info.int_params[AC_nz] = 32;

Device device;
acDeviceCreate(0, info, &device);
```

`AcMesh` is used to load and store data from host to device memory, and vice
versa. Before calling f.ex. `acDeviceLoadMesh()`, the vertex buffers stored in
`AcMesh` must point to valid arrays containing `mx * my * mz` elements, which
can be queried with
```C
#include "astaroth.h"

AcMeshInfo info;
info.int_params[AC_nx] = 128;
info.int_params[AC_ny] = 64;
info.int_params[AC_nz] = 32;
acHostUpdateBuiltinParams(&info); // Recalculates mx, my, and mz

// Compact
int3 mm = acConstructInt3Param(AC_mx, AC_my, AC_mz);

// Or more verbose
mm.x = info.int_params[AC_mx];
mm.y = info.int_params[AC_my];
mm.z = info.int_params[AC_mz];

int number_of_elements = mm.x * mm.y * mm.z;
```

The data must be stored in a row-wise scanline pattern, where the one-dimensional
index of a spatial index `(i, j, k)` is `i + j * mm.x + k * mm.x * mm.y`.



# Astaroth Domain-Specific Language 2.0 (**DEPRECATED**. See acc-runtime/README.md for the Astaroth 3.0 DSL syntax)

We designed the Astaroth Domain-specific Language (DSL) for expressing stencil computations in a
high-level language that can be translated into efficient GPU kernels. The benefits of creating a
DSL are two-fold. First, scientists using the language can focus on developing solvers and
mathematical models using an easy-to-use language, while still achieving performance close to
handwritten code. Second, procedures written in the DSL are decoupled from implementation, which
allows us to extend the DSL compiler, say, to generate optimized code for several hardware
generations without the users having to modify existing DSL sources.

## Overview

The syntax of the Astaroth DSL is an extended subset of C-like languages. The programming model is
based on stream processing, or dataflow programming, where a chain of functions are executed on
streams of data. A kernel is a small GPU program, which defines the operations performed on a
number of data streams. In our case, each data stream corresponds to a single vertex in the mesh,
similar to how vertex shaders operate in graphics shading languages.

With Astaroth DSL, we have borrowed the idea of graphics and image processing pipelines, and used
it for performing three-dimensional stencil computations cache efficiently. The Astaroth DSL is
comprises of three closely related languages, which correspond to distinct stages in the stencil
pipeline shown in the following figure.

![Figure: Stencil pipeline.](../../Astaroth_API_specification_and_user_manual/stencilpipeline.svg "Stencil Pipeline")

| Stage              | File ending | Description                                                                                                                                                                     |
|--------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stencil assembly   | .ac        | Defines the shape of the stencils and functions to be preprocessed before entering the stencil processing stage. Reading from input arrays is only possible during this stage.  |
| Stencil process    | .ac        | The functions executed on streams of data are defined here. Contains kernels, which are essentially main functions of GPU programs.                                            |
| Stencil definition | .ac        | All field identifiers and constant memory symbols are defined in this file.                                                                                                     |
| Any                |  .h         | Optional header files which can be included in any file.    

Compilation of the DSL files is integrated into `CMakelists.txt` provided with the library and
dependencies are recompiled if needed when calling `make`. All DSL files should reside in the same
directory and there should be only one `.ac` file. There may be any number of
optional `.h` files. When configuring the project, the user should pass the path to the DSL
directory as a cmake option like so: ```cmake -DDSL_MODULE_DIR="some/user/dir" ..```.

## Data Types

In addition to basic datatypes in C/C++/CUDA, such as int and int3, we provide the following datatypes with the DSL.

| Data type   |  Description                                                                                                                                                                                                                                                                                                                      | C/C++/CUDA equivalent                                                                                |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Scalar      |  32- or 64-bit floating-point number                                                                                                                                                                                                                                                                                              | float or double                                                                                      |
| Vector      | A tuple of three 32- or 64-bit floating-point numbers                                                                                                                                                                                                                                                                             | float3 or double3                                                                                    |
| Complex     | A tuple of two 32- or 64-bit floating-point numbers. The real part is stored in member .x, while the imaginary component is in .y. Basic operations, such as multiplication, are defined as built-in functions.                                                                                                                   | std::complex<float> or std::complex<double>                                                          |
| Matrix      | A tuple of three Vectors. Is stored in column-major order, f.ex. Matrix[i][j] is the component on row i, column j. (TODO recheck specs.)                                                                                                                                                                                          | float3[3] or double3[3]                                                                              |
| ScalarArray | A one-dimensional array of Scalars stored in device memory. Given mesh dimensions (mx, my, mz), consists of max(mx, max(my, mz)) elements.                                                                                                                                                                                        | float[] or double[]                                                                                  |
| ScalarField | A three-dimensional scalar field stored in row-wise scan order where coordinates `(i, j, k)` correspond to a one-dimensional index `i + j * mx + k * mx * my`. Consists of two buffers, one used for input and another one for output. | Two float[] or double[] arrays                                                                            |
| VectorField | A three-dimensional vector field. Consists of three `ScalarFields`. | Three `ScalarFields` stored contiguously in memory as a structure of arrays |

## Precision

`Scalars` are 32-bit floating-point numbers by default. Double precision can be turned on by setting cmake option `DOUBLE_PRECISION=ON`.
All real number literals are converted automatically to the correct precision. In cases where , the precision can be declared explicitly by appending `f` or `d` postfix to the real number. For example,
```C
1.0           // The same precision as Scalar/AcReal
1.0f          // Explicit float
1.0d          // Explicit double
(1.0f * 1.0d) // 1.0f is implicitly cast to double and the multiplication is done in double precision.
```

> **Note:** Only C++-style casting is currently supported. For example, `float(1.0)` is valid but `(float) 1.0` gives a syntax error.

## Control Flow

Conditional statements are expressed with the `if-else` construct. Unlike in C and C++, we require
that the scope of the `if-else` statement is explicitly declared using braces `{` and `}` in order
to avoid the ambiguity in the case
```C
if (a)
    b;
if (c)
    d;
else
    e;
```


The syntax for conditional statements, even if there is only a single `if`, is
```C
if (a) {
    b;
}
```

## Kernels

Kernels are small programs executed on the device. Each kernel comprises of all the pipeline stages
discussed in previous sections. Functions qualified with the type qualifier `Kernel` are analogous
to `main` functions of host code.

Kernels must be declared in stencil processing files. DSL kernels can be called from host code
using the API function
```C
AcResult acDevice_##identifier(const Device device, const Stream stream,
                                     const int3 start, const int3 end);
```
, where ##identifier is the name of the kernel function.

The following built-in variables are available in `Kernel`s.

| Built-in variable | Description                                                                                                                                                                               |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vertexIdx         | Holds the local index of the currently processed vertex.                                                                                                                                  |
| globalVertexIdx   | Holds the global index of the currently processed vertex. If there is only single device, then vertexIdx is the same as globalVertexIdx. Otherwise globalVertexIdx is offset accordingly. |
| globalGridN       | Holds the dimensions of the computational domain.                                                                                                                                         |

## Preprocessed Functions

The type qualifier `Preprocessed` indicates which functions should be evaluated immediately when
entering a `Kernel` function. The return values of `Preprocessed` functions are cached and calling
these functions during the stencil processing stage is essentially free. As main memory bandwidth is
significantly slower than on-chip memories and registers, declaring reading-heavy functions as
`Preprocessed` is critical for obtaining good performance in stencil codes.

The built-in variables `vertexIdx`, `globalVertexidx` and `globalGridN` are available in all
`Preprocessed` functions.

## Device Functions

The type qualifier `Device` indicates which functions can be called from `Kernel` functions or other `Device` functions.

## Uniforms

`Uniform`s are global device variables which stay constant for the duration of a kernel launch.
`Uniform`s can be updated between kernel launches using the `acLoadScalarUniform` and related functions
discussed in Section 'Loading and storing'.

`Uniform`s are declared in stencil definition headers. The header must be included in all files
which use those uniforms.

`Uniform`s can be of type `Scalar`, `Vector`, `int`, `int3`, `ScalarField` and `ScalarArray`. `Uniforms` of type `Scalar`, `Vector`, `int`, or `int3`, can be set to a default value like any other variable, f.ex. `uniform Scalar a = 1.0; uniform Scalar b = 2.0 * a;`. Default values can be overwritted at runtime with the uniform loader functions discussed above.

> **Note:** As of 2019-10-01, the types `ScalarField` (DSL) and `VertexBuffer` (CUDA) are aliases of the same type. This naming may be changed in the future.

> **Note:** As of 2019-10-01, `VectorField`s cannot be declared as uniforms. Instead, one should declare each component as a `ScalarField` and use them to construct a `VectorField` during the stencil processing stage. For example, `in VectorField(A, B, C);`, where `A`, `B` and `C` are `uniform ScalarField`s.


## Standard Libraries

The following table lists the standard libraries currently available.

| Built-in variable | Description                                                                                                                                                                               |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| stdderiv.h         | Contains functions for computing 2nd, 4th, 6th and 8th order derivatives (configured by defining the STENCIL_ORDER before including stdderiv.h)                                                                                                                                  |

Astaroth DSL libraries can be included in the same way as C/C++ headers. For example, `#include <stdderiv.h>`.

## Performance Considerations

Uniforms are as fast as compile-time constants as long as

1. The halting condition of a tight loop does not depend on an uniform or a variable, as this would prevent unrolling of the loop during compile-time.
2. Uniforms are not multiplied with each other. The result should be stored in an auxiliary uniform instead. For example, the result of `nx * ny` should be stored in a new `uniform nxy`
3. At least 32 neighboring streams in the x-axis access the same `uniform`. That is, the vertices at vertexIdx.x = i... i + 32 should access the same `uniform` where i is a multiple of 32.

## Notes on autotesting

We have model implementations in `src/utils/modelreduce.c` and `src/utils/modelsolver.c` which contain CPU versions of the GPU DSL/reduction code. However, these files are primarily meant for internal testing (finding parallelism bugs or issues with DSL translation) and show only that the GPU and CPU results match. There are significant benefits in doing the checks like this, notably the flexibility of being able to check the results with arbitrary input parameters, but the drawback is that these tests do not show whether the CPU result is also 'correct'.

For testing

* Correctness & implementation-specific: must create your own test
* Correctness & generic: we must create an additional testing module for Astaroth that tests 'correctness'
* CPU-GPU equivalency & generic: add CPU implementation to `src/utils/modelreduce.c`
* CPU-GPU equivalency & implementation-specific: add CPU implementation to `src/utils/modelsolver.c` or then create your own test

The model implementations are available for all projects that use the `astaroth_utils.h` interface. Used in f.ex. `samples/mpitest`. Note that standalone uses its own (deprecated) autotests in `standalone/model`.
