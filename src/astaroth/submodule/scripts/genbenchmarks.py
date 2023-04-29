#!/usr/bin/env python3
import os
import sys
import argparse
import socket
import math
import time
import subprocess
from contextlib import redirect_stdout

###
# Single node io scaling benchmarks
# scripts/genbenchmarks.py --task-type preprocess --partition gputest --max-threads-per-block-range 0 0 --implementations implicit
# scripts/genbenchmarks.py --task-type run --run-scripts benchmark-data/scripts/io-scaling-benchmark-[1-8].sh --run-dirs benchmark-data/builds/* --max-jobs-per-queue 2
# scripts/genbenchmarks.py --task-type postprocess
#
#
# 32 devices
# scripts/genbenchmarks.py --task-type preprocess --max-threads-per-block-range 0 0 --implementations implicit
# scripts/genbenchmarks.py --task-type run --run-scripts benchmark-data/scripts/io-scaling-benchmark-{1..32}.sh --run-dirs benchmark-data/builds/*
# 
###

# Parse arguments
parser = argparse.ArgumentParser(description='A tool for generating benchmarks',
epilog='''EXAMPLES:
    # Generate run scripts and build directories
    genbenchmarks.py --task-type preprocess # Generate makefiles and benchmark scripts
    genbenchmarks.py --task-type genscripts --partition eap # Update partition in all benchmark scripts
    genbenchmarks.py --task-type build --build-dirs benchmark-data/builds/* # Build benchmark directories (required to run)
    genbenchmarks.py --task-type run --run-dirs benchmark-data/builds/* --run-scripts benchmark-data/scripts/* --dryrun # Confirm everything is correct
    genbenchmarks.py --task-type run --run-dirs benchmark-data/builds/* --run-scripts benchmark-data/scripts/* # Do the actual run without --dryrun
    
See Unix globbing for passing files/directories to the script more easily.
    For example:
        ??.sh matches two characters
        *.sh matches any number of characters
        [1-8] matches a character in range 1-8
        {1..16} expands to 1,2,3,...,16
        ?([0-9]) matches zero or one number
        [0-9]?([0-9]) matches one number and an optional second number
        ?[0-9] matches one character and one number
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

## General arguments
parser.add_argument('--task-type', type=str, nargs='+', choices=['genmakefiles', 'genscripts', 'preprocess', 'build', 'run', 'postprocess', 'clean'], help='The type of the task performed with this script', required=True)
parser.add_argument('--dims', type=int, default=[256, 256, 256], nargs=3, help='The dimensions of the computational domain')
parser.add_argument('--dryrun', action='store_true', help='Do a dryrun without compiling or running. Prints os commands to stdout.')
## Preprocess arguments
parser.add_argument('--implementations', type=str, nargs='+', choices=['implicit', 'explicit'], default=['implicit', 'explicit'], help='The list of implementations used in testing')
parser.add_argument('--io-implementations', type=str, nargs='+', choices=['collective', 'distributed'], default=['distributed'], help='The list of IO implementations used in testing')
parser.add_argument('--max-threads-per-block-range', type=int, nargs=2, default=[0, 1024], help='The range for the maximum number of threads per block applied to launch bounds in testing (inclusive)')
parser.add_argument('--cmakelistdir', type=str, default='.', help='Directory containing the project CMakeLists.txt')
parser.add_argument('--use-hip', action='store_true', help='Compile with HIP support')
parser.add_argument('--account', type=str, help='The account used in tests')
parser.add_argument('--partition', type=str, help='The partition used for running the tests')
parser.add_argument('--num-devices', type=int, nargs=2, default=[1, 8192], help='The range for the number of devices generated for run scripts (inclusive)')
## Build arguments
parser.add_argument('--build-dirs', type=str, nargs='+', required='build' in sys.argv, help='A list of directories to build')
## Run arguments
parser.add_argument('--run-scripts', type=str, nargs='+', required='run' in sys.argv, help='A list of job scripts to run the tests')
parser.add_argument('--run-dirs', type=str, nargs='+', required='run' in sys.argv, help='A list of directories to run the tests in')
parser.add_argument('--max-jobs-per-queue', type=int, help='Limit the number of batch jobs submitted to the queue at a time')
## Clean arguments
parser.add_argument('--clean-dirs', type=str, nargs='+', required='clean' in sys.argv, help='A list of directories to clean')

## Parse
args = parser.parse_args()

benchmark_dir = 'benchmark-data'
scripts_dir    = f'{benchmark_dir}/scripts'
builds_dir     = f'{benchmark_dir}/builds'
output_dir     = f'{benchmark_dir}/output'

def syscall(cmd):
    if (args.dryrun):
        print(cmd)
    else:
        os.system(cmd)

import subprocess
import shlex
processes = []
def syscall_async(cmd):
    if (args.dryrun):
        print(cmd)
    else:
        global processes
        processes.append(subprocess.Popen(shlex.split(cmd)))

def syscalls_wait():
    global processes
    while processes:
        processes.pop(0).wait()

# System
class System:
    
    def __init__(self, id, account, partition, ngpus_per_node, modules, use_hip, gres='', additional_commands='', optimal_implementation=1, optimal_tpb=0, srun_params=''):
        self.id = id
        self.account = account
        self.partition = partition
        self.ngpus_per_node = ngpus_per_node
        self.modules = modules
        self.use_hip = use_hip
        self.gres = gres
        self.additional_commands = additional_commands
        self.optimal_implementation = optimal_implementation
        self.optimal_tpb = optimal_tpb
        self.srun_params = srun_params

    def load_modules(self):
        syscall(f'module purge')
        syscall(self.modules)
        
    def print_sbatch_header(self, ntasks, ngpus=-1):
        if ngpus < 0:
            ngpus = ntasks

        time = '00:14:59'

        gpualloc_per_node = min(ngpus, self.ngpus_per_node)
        ntasks_per_node = min(ntasks, self.ngpus_per_node)
        nodes = int(math.ceil(ngpus / self.ngpus_per_node))
        if nodes > 1 and ntasks != ngpus:
            print(f'ERROR: Insufficient ntasks ({ntasks}). Asked for {ngpus} devices but there are only {self.ngpus_per_node} devices per node.')
            assert(nodes == 1 or ntasks == ngpus)

        print('#!/bin/bash')
        if self.account:
            print(f'#SBATCH --account={self.account}')
        if self.gres:
            print(f'#SBATCH --gres={self.gres}')
        print(f'#SBATCH --partition={self.partition}')
        #print(f'#SBATCH --ntasks={ntasks}')
        print(f'#SBATCH --nodes={nodes}')
        print(f'#SBATCH --ntasks-per-node={ntasks_per_node}')
        print(f'#SBATCH --gpus-per-node={gpualloc_per_node}')
        print(f'#SBATCH --time={time}')
        #print('#SBATCH --accel-bind=g') # bind tasks to closest GPU
        #print('#SBATCH --hint=memory_bound') # one core per socket
        #print(f'#SBATCH --ntasks-per-socket={min(ntasks, ngpus/self.nsockets)}')
        #print('#SBATCH --cpu-bind=sockets')
        print(self.additional_commands)
    
        # Load modules
        #print(f'module purge')
        #print(self.modules)

mahti = System(id='a100', account='project_2000403', partition='gpusmall', ngpus_per_node=4, gres='gpu:a100',
               modules='module load gcc/9.4.0 openmpi/4.1.2-cuda cuda cmake', use_hip=False, optimal_implementation=1, optimal_tpb=0)
puhti = System(id='v100', account='project_2000403', partition='gpu', ngpus_per_node=4,
               gres='gpu:v100', modules='module load gcc cuda openmpi cmake', use_hip=False,
               additional_commands='''
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task=10
export UCX_RNDV_THRESH=16384
export UCX_RNDV_SCHEME=get_zcopy
export UCX_MAX_RNDV_RAILS=1''', optimal_implementation=1, optimal_tpb=0)
triton = System(id='mi100', account='', partition='gpu-amd', ngpus_per_node=1, gres='',
                modules='module load gcc bison flex cmake openmpi', use_hip=True, optimal_implementation=1, optimal_tpb=512)
lumi = System(id='mi250x', account='project_462000120', partition='dev-g', ngpus_per_node=8, gres='', additional_commands='''
''', srun_params='--cpu-bind=map_cpu:48,56,16,24,1,8,32,40', modules='''
        module load LUMI/22.08  partition/G
        module load rocm 
        module load buildtools
        module load cray-python
        export MPICH_GPU_SUPPORT_ENABLED=1
        export FI_CXI_DEFAULT_CQ_SIZE=300000
        ''', use_hip=True, optimal_implementation=1, optimal_tpb=512)

# Select system
hostname = socket.gethostname()
if 'mahti' in hostname:
    system = mahti
elif 'puhti' in hostname:
    system = puhti
elif 'uan' in hostname:
    system = lumi
elif 'triton' in hostname:
    system = triton
else:
    print(f'Unknown system {hostname}')
    exit(-1)
system.load_modules()

# Set system account
if args.account:
    system.account = args.account

# Set system partition
if args.partition:
    system.partition = args.partition

# Set problem size
nx = args.dims[0]
ny = args.dims[1]
nz = args.dims[2]

# Set device counts
min_devices = args.num_devices[0]
max_devices = args.num_devices[1]

# Microbenchmarks
def gen_microbenchmarks(system):
    with open(f'{scripts_dir}/microbenchmark.sh', 'w') as f:
        with redirect_stdout(f):
            # Create the batch script
            system.print_sbatch_header(ntasks=1)

            # Bandwidth
            problem_size     = 8 # Bytes
            working_set_size = 8 # Bytes
            max_problem_size = 1 * 1024**3    # 1 GiB
            while problem_size <= max_problem_size:
                print(f'srun {system.srun_params} ./bwtest-benchmark {problem_size} {working_set_size}')
                problem_size *= 2

            # Working set
            problem_size     = 256 * 1024**2 # Bytes, 256 MiB
            working_set_size = 8         # Bytes
            max_working_set_size = 8200  # r = 512, (512 * 2 + 1) * 8 bytes = 8200 bytes
            while working_set_size <= max_working_set_size:
                print(f'srun {system.srun_params} ./bwtest-benchmark {problem_size} {working_set_size}')
                working_set_size *= 2

# Device benchmarks
def gen_devicebenchmarks(system, nx, ny, nz):
    with open(f'{scripts_dir}/device-benchmark.sh', 'w') as f:
        with redirect_stdout(f):
            system.print_sbatch_header(1)
            print(f'srun {system.srun_params} ./benchmark-device {nx} {ny} {nz}')

# Intra-node benchmarks
def gen_nodebenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= min(system.ngpus_per_node, max_devices):
        with open(f'{scripts_dir}/node-scaling-strong-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(1, devices)
                print(f'srun {system.srun_params} ./benchmark-node {nx} {ny} {nz}')
        devices *= 2

# Strong scaling
def gen_strongscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= max_devices:
        with open(f'{scripts_dir}/strong-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun {system.srun_params} ./benchmark {nx} {ny} {nz}')
        devices *= 2

# Weak scaling
def gen_weakscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices):
    # Weak scaling
    devices = min_devices
    initial_nx = nx
    initial_ny = ny
    initial_nz = nz
    while devices <= max_devices:
        with open(f'{scripts_dir}/weak-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun {system.srun_params} ./benchmark {nx} {ny} {nz}')

        if devices <= system.ngpus_per_node:
            with open(f'{scripts_dir}/node-scaling-weak-benchmark-{devices}.sh', 'w') as f:
                with redirect_stdout(f):
                    system.print_sbatch_header(1, devices)
                    # Note: 1D decomposition here
                    nz_1d = int(initial_nz * (nx * ny * nz) / (initial_nx * initial_ny * initial_nz))
                    print(f'srun {system.srun_params} ./benchmark-node {initial_nx} {initial_ny} {nz_1d}')

        devices *= 2
        if nx < ny:
            nx *= 2
        elif ny < nz:
            ny *= 2
        else:
            nz *= 2

# IO benchmarks
def gen_iobenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= max_devices:
        with open(f'{scripts_dir}/io-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun {system.srun_params} ./mpi-io {nx} {ny} {nz} ${{SLURM_JOBID}}')
        devices *= 2

# Generate makefiles
if 'preprocess' in args.task_type or 'genmakefiles' in args.task_type:
    # Builds
    syscall(f'mkdir -p {builds_dir}')
    for implementation in args.implementations:
        for io_implementation in args.io_implementations:
            tpb = args.max_threads_per_block_range[0]
            while tpb <= args.max_threads_per_block_range[1]:

                impl_id     = 1 if implementation == 'implicit' else 2
                use_smem    = implementation == 'explicit'
                distributed = io_implementation == 'distributed'

                build_dir = f'{builds_dir}/implementation{impl_id}_maxthreadsperblock{tpb}_distributed{distributed}'
                syscall(f'mkdir -p {build_dir}')

                # Generate Makefile
                flags = f'''-DOPTIMIZE_MEM_ACCESSES=OFF -DMPI_ENABLED=ON -DSINGLEPASS_INTEGRATION=ON -DUSE_HIP={system.use_hip} -DIMPLEMENTATION={impl_id} -DUSE_SMEM={use_smem} -DMAX_THREADS_PER_BLOCK={tpb} -DUSE_DISTRIBUTED_IO={distributed}'''
                
                cmd = f'cmake {flags} -S {args.cmakelistdir} -B {build_dir}'
                syscall_async(cmd)

                build_info = f'{build_dir}/build-info-{system.id}.txt'
                syscall(f'date > {build_info}')
                syscall(f'echo {cmd} >> {build_info}')
                syscall(f'git -C {args.cmakelistdir} rev-parse HEAD >> {build_info}')

                tpb = 32 if tpb == 0 else 2*tpb
    syscalls_wait()    

# Generate scripts
if 'preprocess' in args.task_type or 'genscripts' in args.task_type:
    # Scripts
    syscall(f'mkdir -p {scripts_dir}')
    if not args.dryrun:
        gen_microbenchmarks(system)

        gen_devicebenchmarks(system, nx, ny, nz)
        gen_nodebenchmarks(system, nx, ny, nz, min_devices, max_devices)

        gen_strongscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices)
        gen_weakscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices)
        gen_iobenchmarks(system, nx, ny, nz, min_devices, max_devices)

    # Outputs
    syscall(f'mkdir -p {output_dir}') # temporarily here

# Outputs
#if 'preprocess' in args.task_type:
#    syscall(f'mkdir -p {output_dir}')

# Build
if 'build' in args.task_type:
    if args.build_dirs:
        for build_dir in args.build_dirs:
            syscall_async(f'make --directory={build_dir} -j')
        syscalls_wait()

# Run
if 'run' in args.task_type:
    for run_dir in args.run_dirs:
        for script in args.run_scripts:
            if args.max_jobs_per_queue:
                njobs = int(subprocess.check_output('squeue --me | wc -l', shell=True)) - 1
                while njobs >= args.max_jobs_per_queue :
                    print('Waiting for jobs to finish...')
                    os.system('squeue --me')
                    time.sleep(2)
                    njobs = int(subprocess.check_output('squeue --me | wc -l', shell=True)) - 1


            syscall(f'sbatch --chdir="{run_dir}" {script}')

            run_info = f'{output_dir}/' + f'run-info-{run_dir}_{script}-{system.id}.txt'.replace('/', '_')
            syscall(f'date > {run_info}')
            syscall(f'module list 2>> {run_info}')
            syscall(f'cat {run_dir}/build-info-{system.id}.txt >> {run_info}')
            syscall(f'cat {script} >> {run_info}')


# Postprocess
if 'postprocess' in args.task_type:
    import pandas as pd

    # Outputs
    syscall(f'mkdir -p {output_dir}')

    # Microbenchmarks
    outfile = f'{output_dir}/microbenchmark-{system.id}.csv'
    with open(outfile, 'w') as f:
        with redirect_stdout(f):
            print('usesmem,maxthreadsperblock,problemsize,workingsetsize,milliseconds,bandwidth')
    syscall(f'cat {builds_dir}/*/microbenchmark.csv >> {outfile}')

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'], keep='last')
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'{output_dir}/bandwidth-{system.id}.csv', index=False)

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'], keep='last')
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'{output_dir}/bandwidth-smem-{system.id}.csv', index=False)

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'], keep='last')
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'{output_dir}/workingset-{system.id}.csv', index=False)

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'], keep='last')
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'{output_dir}/workingset-smem-{system.id}.csv', index=False)

    # Device
    outfile = f'{output_dir}/device-benchmark-{system.id}.csv'
    with open(outfile, 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    syscall(f'cat {builds_dir}/*/device-benchmark.csv >> {outfile}')

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation'] == 1) & (df['nx'] == 256) & (df['ny'] == 256) & (df['nz'] == 256)]
    df = df.sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/implicit-{system.id}.csv', index=False)

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation'] == 2) & (df['nx'] == 256) & (df['ny'] == 256) & (df['nz'] == 256)]
    df = df.sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/explicit-{system.id}.csv', index=False)

    # Node
    outfile = f'{output_dir}/node-benchmark-{system.id}.csv'
    with open(outfile, 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    syscall(f'cat {builds_dir}/*/node-benchmark.csv >> {outfile}')

    '''
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation'] == 1)]
    df = df.sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/node-implicit-{system.id}.csv', index=False)

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation'] == 2)]
    df = df.sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/node-explicit-{system.id}.csv', index=False)
    '''

    '''
    # Find the best tpb
    best_tpb = -1
    best_ms = float('inf')
    for tpb in df['maxthreadsperblock'].drop_duplicates():
        ms = df.loc[(df['maxthreadsperblock'] == tpb) & (df['devices'] == 1)].sort_values(by=['devices'])['milliseconds'].iloc[0]
        if ms < best_ms:
            best_ms = ms
            best_tpb = tpb
    '''
    
    '''
    # Implicit full card
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['devices'] == 2) & (df['implementation'] == 1) & (df['nx'] == 256) & (df['ny'] == 256) & (df['nz'] == 256)].sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/node-2-implicit-{system.id}.csv', index=False)

    # Explicit full card
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['devices'] == 2) & (df['implementation'] == 2) & (df['nx'] == 256) & (df['ny'] == 256) & (df['nz'] == 256)].sort_values(by=['maxthreadsperblock'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df.to_csv(f'{output_dir}/node-2-explicit-{system.id}.csv', index=False)
    '''

    # Node scaling strong
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation']==system.optimal_implementation) & (df['maxthreadsperblock']==system.optimal_tpb)].sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df = df.loc[(df['nx'] == 256) & (df['ny'] == 256) & (df['nz'] == 256)]
    df.to_csv(f'{output_dir}/node-scaling-strong-{system.id}.csv', index=False)

    # Node scaling weak
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['implementation']==system.optimal_implementation) & (df['maxthreadsperblock']==system.optimal_tpb)].sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['implementation','maxthreadsperblock','nx','ny','nz','devices'], keep='last')
    df = df[df['nx']*df['ny']*df['nz'] == df['devices']*256*256*256]
    df.to_csv(f'{output_dir}/node-scaling-weak-{system.id}.csv', index=False)

    # Scaling
    outfile = f'{output_dir}/scaling-benchmark-{system.id}.csv'
    with open(outfile, 'w') as f:
        with redirect_stdout(f):
            print('devices,millisecondsmin,milliseconds50thpercentile,milliseconds90thpercentile,millisecondsmax,usedistributedcommunication,nx,ny,nz,dostrongscaling')
    syscall(f'cat {builds_dir}/*/scaling-benchmark.csv >> {outfile}')

    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['nx'] == nx) & (df['ny'] == ny) & (df['nz'] == nz)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'], keep='last')
    df.to_csv(f'{output_dir}/scaling-strong-{system.id}.csv', index=False)

    nn = 256*256*256 # nx * ny * nz
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['nx'] * df['ny'] * df['nz']) / df['devices'] == nn]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'], keep='last')
    # Hack start (replace intra-node results with P2P instead of MPI)
    #df2 = pd.read_csv(f'{output_dir}/node-scaling-weak-{system.id}.csv', comment='#')
    #df['milliseconds90thpercentile'].iloc[0:len(df2.milliseconds.values)-1] = df2.milliseconds.values[:-1]
    # Hack end
    df.to_csv(f'{output_dir}/scaling-weak-{system.id}.csv', index=False)

    # IO scaling
    outfile = f'{output_dir}/scaling-io-benchmark-{system.id}.csv'
    with open(outfile, 'w') as f:
        with redirect_stdout(f):
            print(f'devices,writemilliseconds,writebandwidth,readmilliseconds,readbandwidth,usedistributedio,nx,ny,nz')
    syscall(f'cat {builds_dir}/*/scaling-io-benchmark.csv >> {outfile}')

    # Collective
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usedistributedio'] == 0)]
    df = df.loc[(df['nx'] == nx) & (df['ny'] == ny) & (df['nz'] == nx)].sort_values(by=['devices'])
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'], keep='last')
    df.to_csv(f'{output_dir}/scaling-io-collective-{system.id}.csv', index=False)

    # Distributed
    df = pd.read_csv(outfile, comment='#')
    df = df.loc[(df['usedistributedio'] == 1)]
    df = df.loc[(df['nx'] == nx) & (df['ny'] == ny) & (df['nz'] == nx)].sort_values(by=['devices'])
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'], keep='last')
    df.to_csv(f'{output_dir}/scaling-io-distributed-{system.id}.csv', index=False)

if 'clean' in args.task_type:
    for dir in args.clean_dirs:
        syscall(f'rm {dir}/*.csv')
