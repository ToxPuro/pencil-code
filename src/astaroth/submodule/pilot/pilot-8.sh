#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00

module purge
module load LUMI/22.08  partition/G
module load rocm
module load buildtools
module load cray-python

export MPICH_GPU_SUPPORT_ENABLED=1

# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --run-init-kernel
srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/scratch/project_462000120/jpekkila/mahti-512-varfile/var.dat # From varfile
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-snapshot # From snapshot
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./mpitest 64 64 64 # Autotest
