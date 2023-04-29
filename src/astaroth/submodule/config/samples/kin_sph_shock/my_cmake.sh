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

cmake -DDOUBLE_PRECISION=ON -DDSL_MODULE_DIR=acc/mhd_solver ../../../

# Standard compilation

make -j 
