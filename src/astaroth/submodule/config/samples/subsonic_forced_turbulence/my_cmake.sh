#!/bin/bash

# This is a sample script. Please copy it to the directory you want to run the
# code in and customize occordingly. 

# The following write the commit indentifier corresponding to the simulation
# run  into a file. This is to help keep track what version of the code was
# used to perform the simulation.

git rev-parse HEAD > COMMIT_CODE.log 

# Run cmake to construct makefiles
# In the case you compile in astaroth/build/ directory. Otherwise change ".." to
# the correct path to astaroth/CMakeLists.txt

case $HOSTNAME in 
  ( gp8.tiara.sinica.edu.tw | gp9.tiara.sinica.edu.tw | gp10.tiara.sinica.edu.tw | gp11.tiara.sinica.edu.tw ) 
     cmake -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DUSE_CUDA_AWARE_MPI=OFF -DDSL_MODULE_DIR=../../../acc-runtime/samples/mhd_modular/ -DCMAKE_CXX_COMPILER=/software/opt/gcc/8.3.0/bin/gcc -DCMAKE_C_COMPILER=/software/opt/gcc/8.3.0/bin/gcc ../../.. 
     ;;
  (*) 
     cmake -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DUSE_CUDA_AWARE_MPI=ON  ../../..
     ;;
esac

# Standard compilation

make -j 
