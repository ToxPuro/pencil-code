# Building

`./build.sh <optional CMake parameters>`

# Examples
`./build.sh`  
`./build.sh -DUSE_HIP=OFF`

# Pytorch on LUMI

`module use /appl/local/csc/modulefiles/`
`module load pytorch`

# Initial results (2023-03-21)

Astaroth
```
[100%] Built target heat-equation
srun: job 3334221 queued and waiting for resources
srun: job 3334221 has been allocated resources
AC_mx: 262
AC_my: 262
AC_mz: 262
AC_nx: 256
AC_ny: 256
AC_nz: 256
Time elapsed: 0.834026 ms
20116.7 M elements per second
````

Pytorch (initial implementation with conv3d, no optimizations)
```
Convolution time elapsed: 4388.1683349609375 ms
Euler step time elapsed: 4388.787746429443 ms
M elements per second: 3.8227449057315033
```