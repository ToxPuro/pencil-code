# Plasma Physics Meets AI: Astaroth workshop

## Building the exercises

> `cd astaroth`

> `mkdir build && cd build`

> (Optional) Remove `CMakeCache.txt`  and do `make clean` in the build directory if there are issues and you have built Astaroth before. Previous runs may have set some cached CMake options that interfere with compiling the default samples.

> `cmake -DPROGRAM_MODULE_DIR=../samples/plasma-meets-ai-workshop/<exercise name> .. && make -j`

> `./<exercise name>`

> (Optional) Visualize the output with `../samples/plasma-meets-ai-workshop/animate-snapshots.py <list of .dat files, f.ex. *.dat or UUX*.dat>`. Requires `python`, `python-numpy`, and `python-matplotlib`.

> (Optional) Check the Section [Getting Started With astaroth](#getting-started-with-astaroth-detailed-instructions) at the bottom of the document for more information on getting Astaroth up and running.

## Exercise 1: Blur Filter

In this exercise, we will implement a blur filter. The DSL file to be modified is in `blur/blur.ac` and the main program is in `blur/blur.c`. The executable is `./blur` in the build directory.

## Exercise 2: Simulating Hydrodynamics

See `hydro/hydro.ac` and `hydro/hydro.c`. The executable is `./hydro` in the build directory.

## Exercise 3: Adding SGS stress to our hydrodynamics simulation

See `hydro-sgs/hydro-sgs.ac` and `hydro-sgs/hydro-sgs.c`. The executable is `./hydro-sgs` in the build directory.


# Getting started with Astaroth - detailed instructions

### Dependencies

> `flex bison cmake cuda`

#### Documentation
* Astaroth API and DSL documentation is located at
    * [General usage](https://bitbucket.org/jpekkila/astaroth/src/master/README.md)
    * [DSL](https://bitbucket.org/jpekkila/astaroth/src/master/acc-runtime/README.md)
    * [API](https://bitbucket.org/jpekkila/astaroth/src/master/doc/Astaroth_API_specification_and_user_manual/API_specification_and_user_manual.md)

* The code samples are located in
    * [astaroth/acc-runtime/samples](https://bitbucket.org/jpekkila/astaroth/src/master/acc-runtime/samples/)
    * [astaroth/samples](https://bitbucket.org/jpekkila/astaroth/src/master/samples/)

#### Issues

* The compilation fails for some reason

    > Ensure the dependencies `flex bison cmake cuda` are installed.

    > Clean the build directory completely (incl. `CMakeCache.txt` and do `make clean`) and try again.

* CMake available on the system is too old

    > You can build CMake from source or download the latest release from the official website

    Building from source
    ```
    git clone https://gitlab.kitware.com/cmake/cmake.git
    cd cmake && mkdir build && cd build
    git checkout c6ee02fc8db2b2881d6f314a37f193c8726b55ba
    cmake .. && make -j 16
    ```

    > (Optional) alias cmake to the new version `alias cmake="/<path to cmake root dir>/build/bin/cmake"`

* The dummy kernel fails or no devices are found

    > Ensure a GPU is visible in the system. F.ex. if using Slurm, you should pass a `--gres=gpu:v100:1`, or similar, flag when queuing for resources

* Missing modules / segmentation faults when using MPI

    > Ensure the proper modules are loaded. `module load gcc/8.3.0 cuda/10.1.168 cmake openmpi/4.0.3-cuda` or similar should work.

    > If there is no MPI implementation available that was built with GPU support, you can try setting `cmake -DUSE_CUDA_AWARE_MPI=OFF`