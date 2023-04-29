Created by *Miikka Väisälä*, at 2022/8

# Purpose of this simulation setup 

This is an example case of running a forced turbulence simulation with isothermal MHD. 
Similar to runs in Vaisala et al. (2021). 

# What is a successful test

A succesful test is that `standalone_mpi` runs properly on multiple GPUs
without the system crashing due to unnatural reasons, and results looks sensible
without numerical garbage. 

# Required LSWITCHES 

At the moment you need to set the switches manually. Default master branch
configuration will have different LSWITCHES.

* In `../../../acc-runtime/samples/mhd_modular/mhdsolver.ac`

```
LDENSITY (1)
LHYDRO (1)
LMAGNETIC (1)
LENTROPY (0)
LTEMPERATURE (0)
LFORCING (1)
LUPWD (1)
LSINK (0)
LBFIELD (1)
LSHOCK (0)
```

# Setting up and compiling.

Run `./my_cmake.sh`

# Running the simulation. 

Run e.g. `mpirun -n 4 ./ac_run_mpi -c astaroth.conf` or however you particular
system runs MPI. 

# Troubleshooting

It the case you get strange MPI errors, it might be that your particular system
has not been configured for GPUDirect RDMA. To run Astaroth without GPUDirect
RDMA, please set `-DUSE_CUDA_AWARE_MPI=OFF` in `my_cmake.sh`. 

On one machine we run into an issue that a wrong version of gcc was found. In
that case please either set flags `-DCMAKE_C_COMPILER=/path/to/gcc/`
`-DCMAKE_CXX_COMPILER=/path/to/gcc/` or set the environmental variables CC and
CXX with a correct path to your compiler. 
