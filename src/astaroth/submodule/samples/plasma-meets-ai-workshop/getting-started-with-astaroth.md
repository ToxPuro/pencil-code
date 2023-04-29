### Getting started with Astaroth

### Dependencies

> `flex bison cmake cuda`

#### Quick start
* Clone the repository at bitbucket.org/jpekkila/astaroth
    >`git clone git@bitbucket.org:jpekkila/astaroth.git`

* Build the library
    > `cd astaroth`

    > `mkdir build && cd build`

    > (Optional) Remove `CMakeCache.txt`  and do `make clean` in the build directory if there are issues and you have built Astaroth before. Previous runs may have set some cached CMake options that interfere with compiling the default samples.
     
    > `cmake .. && make -j`

* Try running one of the samples
    > `./benchmark-device 32 32 32`

#### Additional information 
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