
*Miikka Vaisala: This is just something I have astarted to write up to make sense about the Astaroth 2.0. Starting for personally important notes to understand the code. Will be refined as my understanding improves.*

#Astaroth manual

## Compilation

See the `README.md`. At the moment, let us keep certaint things in one place.

## Simulation instructions

At the moment it is only possible to build and run in the `astaroth_2.0/build/` directory. Possibility to add separate run directories will be included later.

### Choosing physics

Runtime settings can be adjusted from `astaroth_2.0/include/astaroth.h` and `astaroth_2.0/config/astaroth.conf`.

Howeve, physics switches LENTROPY, LFORCING etc. do not work at the moment. There has been an issue to get pre-processor combatible with astaroth-domain-specific language in Astaroth 2.0. Therefore, all features are online by default.

To get the switcher working now, rename `astaroth_2.0/src/core/kernels/rk3handtuned.cuh` -> `rk3.cuh`. (**MV:** Not yet tested.)

How to use?

What kind of runtime settings?

### Setting initial conditions

Where can we effectively choose the initial condition?

### Launchin a run

`./ac_run -s` assuming you are doing a normal simulation. Basic code for this invocation can be found in the source file `astaroth_2.0/src/standalone/simulation.cc`.

Please note that launching `./ac_run -t` will *fail if entropy and forcing are in use*. Test is mainly for finding paralleization bugs. (In principle if hydro stuff and induction work, so will forcing and entropy.)

### Diagnostic variables

What is calculated?

Where it is saved?

### Simulation data

Saving output binaries is not enabled yet.

**MV:** I am planning to implement HDF5 format for the data. **TOP PRIORITY**.

#### Notes about data structures

- Configuration parameters have prefix `AC_`, such as `AC_dsx`.

- All configurations are stored in the struct `AcMeshInfo`, containing tables `int_params` ja `real_params`. **NOTE:** `int_params` and `real_params` require diligence. If you call e.g. `int_params[AC_dsx]`, the result will be something unexpected. So-far error checking with this has now been possible to be automated.


- All mesh data is stored to the struct `AcMesh`, containing both configuration values and vertex data (`lnrho`, `uux`, etc.)

- All essential tructs, macros and enumerators are found in astaroth.h for better reference.

- In the case there is changes in the data layout, better use macro `acVertexBufferIdx(i, j, k, mesh_info)`which transform indices from 3D to 1D. Therefore no need to start writing `i + j * mesh_info.int_params[AC_mx] + ...` which would affect the code readability.

- AcReal on generic floating point real number type used everywhere in the code. Currently can be either `float` or `double`. Possibly in the future also `half` or `long double` could become available.

Sample code:

```cpp
AcMeshInfo mesh_info;
// Loads data from astaroth.conf into the AcMeshInfo struct
load_config(&mesh_info);

// Allocates data on the host for the AcMesh struct using information found in mesh_info.
AcMesh* mesh = acmesh_create(mesh_info);

// Initializes mesh to InitType (specified in standalone/model/host_memory.h)
acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh); 

// Allocates data on the device for the AcMesh struct
acInit(mesh_info); 

acLoad(*mesh); // Loads the mesh to the device


const AcReal dt = 1.f;

// Synchronizes previous device commands
acSynchronize(); 

// Does a full rk3 integration step on the device
acIntegrate(dt); 

acSynchronize();

// Store data from device to host mesh
acStore(mesh); 

printf("nx: %d, dsx %f\n", 
        mesh->info.int_params[AC_nx], 
        double(mesh->info.real_params[AC_dsx]));
printf("First vertex of the computational domain: %f\n",        
double(mesh->vertex_buffer[VTXBUF_LNRHO][acVertexBufferIdx(3, 3, 3, mesh_info)]));

```


### Reading data

Depends on the output format. With HDF5 should be simple enough.

[Jupyter notebook](http://jupyter.org/) visualization?

Do we want to use [YT?](https://yt-project.org/)

### Live rendering

MV: Cool, but does not work for remote cluster so far. A GPU workstation is required.

##Multi-GPU

At the moment multi-GPU is not included in Astaroth 2.0. However, it has been implemented 1.0 (`astaroth_1.0/src/gpu/cuda/cuda_generic.cu`) could be essentially ported by copypasting to `astaroth_2.0/src/core/astaroth.cu` after we have clear idea how to run things with single GPU. Could be done overnight in principle.


## Profiling

The built-in beachmark is currently unreliable due to an unknown reason. Please use [nvprof and nvvp](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) for precise profiling. Also, NVIDIA suggests their [Nsight Systems](https://developer.nvidia.com/nsight-systems).



## ETC

**Note** `auto_optimize.sh` does not currently work, but it aims to tune thread block dimensions automatically.


