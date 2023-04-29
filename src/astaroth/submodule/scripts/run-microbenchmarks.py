#!/bin/python3
import os

# Set cmake, build dir, and srun based on the system
import socket
hostname = socket.gethostname()
if "mahti" in hostname or "puhti" in hostname or "uan" in hostname:
    build_dir='/users/pekkila/astaroth/build'
    if "mahti" in hostname:
        cmake='/users/pekkila/CMake/build/bin/cmake'
        srun='srun --account=project_2000403 --gres=gpu:a100:1 -t 00:14:59 -p gputest -n 1 -N 1 --pty'
    elif "puhti" in hostname:
        cmake='/users/pekkila/cmake/build/bin/cmake'
        srun='srun --account=project_2000403 --gres=gpu:v100:1 -t 00:14:59 -p gputest -n 1 -N 1 --pty'
    elif "uan" in hostname:
        build_dir='/pfs/lustrep1/users/pekkila/astaroth/build'
        cmake='cmake -DUSE_HIP=ON'
        srun='srun --account=project_462000120 --gres=gpu:1 -t 00:05:00 -p pilot -n 1 -N 1 --pty'
    else:
        print("Unknown hostname when setting srun")
        exit(-1)
elif "triton" in hostname:
    build_dir='/home/pekkilj1/astaroth/build'
    cmake='/home/pekkilj1/cmake/build/bin/cmake -DUSE_HIP=ON'
    srun=''
elif "cs-009" in hostname:
    build_dir='/m/home/home6/61/pekkilj1/unix/repositories/astaroth/build'
    cmake='/m/home/home6/61/pekkilj1/unix/repositories/cmake/build/bin/cmake'
    srun=''
else:
    print("Could not recognize the system")
    exit(-1)

# Check whether we're in the correct directory
cwd = os.getcwd()
if cwd != build_dir:
    print(f"Invalid dir {cwd}. Should be {build_dir}")
    exit(-1)

# Variable problem size
def benchmark_problem_size(out_file='problem-size'):
    cmd = ""
    problem_size     = 8 # Bytes
    working_set_size = 8 # Bytes
    max_problem_size = 1 * 1024**3    # 1 GiB
    while problem_size <= max_problem_size:
        cmd += f'./bwtest-benchmark {problem_size} {working_set_size} ; '
        problem_size *= 2

    os.system('echo "problemsize,workingsetsize,milliseconds,bandwidth" > bwtest-benchmark.csv')
    os.system(f'{srun} /bin/bash -c \"{cmd}\"')
    os.system(f'mv bwtest-benchmark.csv {out_file}.csv')

# Variable working set size
def benchmark_working_set_size(out_file='working-set-size'):
    cmd = ""
    problem_size = 256 * 1024**2 # Bytes, 256 MiB
    working_set_size = 8         # Bytes
    max_working_set_size = 8200  # r = 512, (512 * 2 + 1) * 8 bytes = 8200 bytes
    while working_set_size <= max_working_set_size:
        cmd += f'./bwtest-benchmark {problem_size} {working_set_size} ; '
        working_set_size *= 2

    os.system('echo "problemsize,workingsetsize,milliseconds,bandwidth" > bwtest-benchmark.csv')
    os.system(f'{srun} /bin/bash -c \"{cmd}\"')
    os.system(f'mv bwtest-benchmark.csv {out_file}.csv')

# Build
def build(use_smem=0, max_threads_per_block=0):
    os.system(f'{cmake} -DUSE_SMEM={use_smem} -DMAX_THREADS_PER_BLOCK={max_threads_per_block} .. && make -j')

build(use_smem=0, max_threads_per_block=0)
benchmark_problem_size('problem-size')
benchmark_working_set_size('working-set-size')

build(use_smem=0, max_threads_per_block=192)
benchmark_problem_size('problem-size-launch-bounds')
benchmark_working_set_size('working-set-size-launch-bounds')

build(use_smem=1, max_threads_per_block=0)
benchmark_problem_size('problem-size-smem')
benchmark_working_set_size('working-set-size-smem')

build(use_smem=1, max_threads_per_block=192)
benchmark_problem_size('problem-size-smem-launch-bounds')
benchmark_working_set_size('working-set-size-smem-launch-bounds')
