This directory is used to test MPI with Astaroth.

Building:
cmake -DBUILD_SAMPLES=ON -DMPI_ENABLED=ON -DCMAKE_CXX_COMPILER=$(which mpicxx) .. && make -j

Running:
mpirun -np <nprocs> ./mpitest
or
srun <options> ./mpitest # With slurm
or
a batch script of your choice
