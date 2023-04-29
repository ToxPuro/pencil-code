# ACC - Astaroth Code Compiler

ACC is a source-to-source compiler for generating CUDA kernels from programs written in Astaroth Code (AC). This document focuses on how to build and run the compiler. For detailed description of code generation and compilation phases, we refer the reader to [J. Pekkilä, Astaroth: A Library for Stencil Computations on Graphics Processing Units. 2019.](http://urn.fi/URN:NBN:fi:aalto-201906233993), Section 4.3. We refer the reader to [Specification](doc/Astaroth_API_specification_and_user_manual/API_specification_and_user_manual.md) for a detailed description of AC syntax. 

ACC is automatically compiled and invoked when compiling the Astaroth Library, user intervention is not needed. The instructions presented in this file are only for developers looking to debug the AC compiler.

## Dependencies

`gcc flex bison`

## Building

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make -j`

## Usage

Script `compile_acc_module.sh` executes all compilation stages from preprocessing to linking AC standard libraries. The resulting cuda headers are placed in the current working directory. The script should be invoked as follows.

> `./compile_acc_module <a directory containing AC files>`

For preprocessing only, see `preprocess.sh`. The first parameter is regarded as the AC source file, while rest of the parameters are passed to gcc. For example:

> `./preprocess.sh file.ac -I dir` 

Preprocesses `file.ac` and searches `dir` for files to be included.

For invoking the code generator, pass preprocessed files that respect AC syntax to `acc`. 

For example:
> `acc < file.ac.preprocessed`

See [Building](#markdown-header-building) on how to obtain `acc`.

