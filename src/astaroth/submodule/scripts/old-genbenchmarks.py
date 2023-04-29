#!/usr/bin/env python3
# module load python-data # to load the dependencies
from contextlib import redirect_stdout
from contextlib import contextmanager
import os
import socket
import math
import sys
import argparse

_dryrun=False


class System:

    def __init__(self, id, account, partition, ngpus_per_node, modules, use_hip, gres='', additional_commands='', optimal_implementation=1, optimal_tpb=0):
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

    def load_modules(self):
        # Load modules
        os.system(f'module purge')
        os.system(self.modules)
        
    def print_sbatch_header(self, ntasks, ngpus=-1):
        if ngpus < 0:
            ngpus = ntasks

        time = '00:14:59'

        gpualloc_per_node = min(ngpus, self.ngpus_per_node)
        nodes = int(math.ceil(ngpus / self.ngpus_per_node))
        if nodes > 1 and ntasks != ngpus:
            print(f'ERROR: Insufficient ntasks ({ntasks}). Asked for {ngpus} devices but there are only {self.ngpus_per_node} devices per node.')
            assert(nodes == 1 or ntasks == ngpus)

        print('#!/bin/bash')
        if self.account:
            print(f'#SBATCH --account={self.account}')
        if self.gres:
            print(f'#SBATCH --gres={self.gres}:{gpualloc_per_node}')
        print(f'#SBATCH --partition={self.partition}')
        print(f'#SBATCH --ntasks={ntasks}')
        print(f'#SBATCH --nodes={nodes}')
        print(f'#SBATCH --time={time}')
        #print('#SBATCH --accel-bind=g') # bind tasks to closest GPU
        #print('#SBATCH --hint=memory_bound') # one core per socket
        #print(f'#SBATCH --ntasks-per-socket={min(ntasks, ngpus/self.nsockets)}')
        #print('#SBATCH --cpu-bind=sockets')
        print(self.additional_commands)
    
        # Load modules
        print(f'module purge')
        print(self.modules)

    def build(self, build_flags, cmakelistdir, do_compile):
        if do_compile:
            if _dryrun:
                print(f'cmake {build_flags} {cmakelistdir}')
                print('make -j')
            else:
                os.system(f'cmake {build_flags} {cmakelistdir}')
                os.system('make -j')


class FileStructure:

    def __init__(self, cmakelistdir='.'):

        initial_dir = os.getcwd()

        # Record the CMakeLists.txt dir
        os.chdir(cmakelistdir)
        print(f'cd {os.getcwd()}')
        if not os.path.isfile('CMakeLists.txt'):
            print(f'Could not find CMakeLists.txt in {os.getcwd()}. Please run the script in the dir containing the project CMakeLists.txt or give the directory as a parameter.')
            exit(-1)

        self.cmakelistdir = os.getcwd()
        os.chdir(initial_dir)
        print(f'cd {os.getcwd()}')

        # Create a new dir for the benchmark data
        os.system(f'mkdir -p benchmark-data')
        os.chdir('benchmark-data')
        print(f'cd {os.getcwd()}')
        self.base_dir = os.getcwd()

        os.system(f'mkdir -p builds')
        os.chdir('builds')
        print(f'cd {os.getcwd()}')
        self.build_dir = os.getcwd()

        os.chdir(self.base_dir)
        print(f'cd {os.getcwd()}')
        os.system(f'mkdir -p scripts')
        os.chdir('scripts')
        print(f'cd {os.getcwd()}')
        self.script_dir = os.getcwd()

        os.chdir(initial_dir)
        print(f'cd {os.getcwd()}')


def genbuilds(fs, do_compile):
    # Create build dirs
    num_implementations = 2
    max_threads_per_block = 1024
    for implementation in range(1, num_implementations+1):
        tpb = 0
        while tpb <= max_threads_per_block:

            os.chdir(fs.build_dir)
            print(f'cd {os.getcwd()}')
            dir = f'implementation{implementation}_maxthreadsperblock{tpb}'
            os.system(f'mkdir -p {dir}')
            os.chdir(dir)
            print(f'cd {os.getcwd()}')

            # Build
            use_smem = (implementation == 2) # tmp hack, note depends on implementation enum
            build_flags = f'-DUSE_HIP={system.use_hip} -DMPI_ENABLED=ON -DIMPLEMENTATION={implementation} -DMAX_THREADS_PER_BLOCK={tpb} -DUSE_SMEM={use_smem}'
            system.build(build_flags, fs.cmakelistdir, do_compile)

        
            if tpb == 0:
                tpb = 32
            else:
                tpb *= 2

    # Create collective and distributed IO builds
    distributed=False
    os.chdir(fs.build_dir)
    print(f'cd {os.getcwd()}')
    dir = f'implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}_distributed{distributed}'
    os.system(f'mkdir -p {dir}')
    os.chdir(dir)
    print(f'cd {os.getcwd()}')
    build_flags = f'-DUSE_DISTRIBUTED_IO={distributed} -DUSE_HIP={system.use_hip} -DMPI_ENABLED=ON -DIMPLEMENTATION={system.optimal_implementation} -DMAX_THREADS_PER_BLOCK={system.optimal_tpb} -DUSE_SMEM={use_smem}'
    system.build(build_flags, fs.cmakelistdir, do_compile)

    distributed=True
    os.chdir(fs.build_dir)
    print(f'cd {os.getcwd()}')
    dir = f'implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}_distributed{distributed}'
    os.system(f'mkdir -p {dir}')
    os.chdir(dir)
    print(f'cd {os.getcwd()}')
    build_flags = f'-DUSE_DISTRIBUTED_IO={distributed} -DUSE_HIP={system.use_hip} -DMPI_ENABLED=ON -DIMPLEMENTATION={system.optimal_implementation} -DMAX_THREADS_PER_BLOCK={system.optimal_tpb} -DUSE_SMEM={use_smem}'
    system.build(build_flags, fs.cmakelistdir, do_compile)

# Microbenchmarks
def gen_microbenchmarks(system, fs):
    with open('microbenchmark.sh', 'w') as f:
        with redirect_stdout(f):
            # Create the batch script
            system.print_sbatch_header(ntasks=1)

            # Bandwidth
            problem_size     = 8 # Bytes
            working_set_size = 8 # Bytes
            max_problem_size = 1 * 1024**3    # 1 GiB
            while problem_size <= max_problem_size:
                print(f'srun ./bwtest-benchmark {problem_size} {working_set_size}')
                problem_size *= 2

            # Working set
            problem_size     = 256 * 1024**2 # Bytes, 256 MiB
            working_set_size = 8         # Bytes
            max_working_set_size = 8200  # r = 512, (512 * 2 + 1) * 8 bytes = 8200 bytes
            while working_set_size <= max_working_set_size:
                print(f'srun ./bwtest-benchmark {problem_size} {working_set_size}')
                working_set_size *= 2

def run_microbenchmarks(fs):
    # Implicit
    os.chdir(f'{fs.build_dir}/implementation1_maxthreadsperblock0')
    print(f'cd {os.getcwd()}')
    if _dryrun:
        print(f'sbatch {fs.script_dir}/microbenchmark.sh')
    else:
        os.system(f'sbatch {fs.script_dir}/microbenchmark.sh')

    # Explicit
    os.chdir(f'{fs.build_dir}/implementation2_maxthreadsperblock0')
    print(f'cd {os.getcwd()}')
    if _dryrun:
        print(f'sbatch {fs.script_dir}/microbenchmark.sh')
    else:
        os.system(f'sbatch {fs.script_dir}/microbenchmark.sh')

# Device benchmarks
def gen_devicebenchmarks(system, fs, nx, ny, nz):
    with open('device-benchmark.sh', 'w') as f:
        with redirect_stdout(f):
            system.print_sbatch_header(1)
            print(f'srun ./benchmark-device {nx} {ny} {nz}')

def run_devicebenchmarks(fs):
    dirs = os.listdir(fs.build_dir)
    for dir in dirs:
        os.chdir(f'{fs.build_dir}/{dir}')
        print(f'cd {os.getcwd()}')

        if _dryrun:
            print(f'sbatch {fs.script_dir}/device-benchmark.sh')
        else:
            os.system(f'sbatch {fs.script_dir}/device-benchmark.sh')

# Intra-node benchmarks
def gen_nodebenchmarks(system, fs, nx, ny, nz):
    devices = 1
    while devices <= system.ngpus_per_node:
        with open(f'node-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(1, devices)
                print(f'srun ./benchmark-node {nx} {ny} {nz}')
        devices *= 2

def run_nodebenchmarks(fs):
    dirs = os.listdir(fs.build_dir)
    for dir in dirs:
        os.chdir(f'{fs.build_dir}/{dir}')
        print(f'cd {os.getcwd()}')

        if _dryrun:
            print(f'sbatch {fs.script_dir}/node-benchmark-2.sh')
        else:
            os.system(f'sbatch {fs.script_dir}/node-benchmark-2.sh')

# Strong scaling
def gen_strongscalingbenchmarks(system, fs, nx, ny, nz, max_devices):
    devices = 1
    while devices <= max_devices:
        with open(f'strong-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./benchmark {nx} {ny} {nz}')
        devices *= 2

def run_strongscalingbenchmarks(system, fs):
    os.chdir(f'{fs.build_dir}/implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}')
    print(f'cd {os.getcwd()}')

    scripts = filter(lambda x: 'strong-scaling-benchmark' in x, os.listdir(fs.script_dir)) # Note filter iterator: exhausted after one pass
    for script in scripts:
        if _dryrun:
            print(f'sbatch {fs.script_dir}/{script}')
        else:
            os.system(f'sbatch {fs.script_dir}/{script}')


# Weak scaling
def gen_weakscalingbenchmarks(system, fs, nx, ny, nz, max_devices):
    # Weak scaling
    devices = 1
    while devices <= max_devices:
        with open(f'weak-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./benchmark {nx} {ny} {nz}')
        devices *= 2
        if nz < ny:
            nz *= 2
        elif ny < nx:
            ny *= 2
        else:
            nx *= 2

def run_weakscalingbenchmarks(system, fs):
    os.chdir(f'{fs.build_dir}/implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}')
    print(f'cd {os.getcwd()}')

    scripts = filter(lambda x: 'weak-scaling-benchmark' in x, os.listdir(fs.script_dir)) # Note filter iterator: exhausted after one pass
    for script in scripts:
        if _dryrun:
            print(f'sbatch {fs.script_dir}/{script}')
        else:
            os.system(f'sbatch {fs.script_dir}/{script}')

# IO benchmarks
def gen_iobenchmarks(system, fs, nx, ny, nz, max_devices):
    devices = 1
    while devices <= max_devices:
        with open(f'io-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./mpi-io {nx} {ny} {nz}')
        devices *= 2

def run_ioscalingbenchmarks(system, fs):
    scripts = filter(lambda x: 'io-scaling-benchmark' in x, os.listdir(fs.script_dir))
    scripts = list(scripts) # Convert from iterator to list to enable multiple passes over the data

    # Collective
    os.chdir(f'{fs.build_dir}/implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}_distributedFalse')
    print(f'cd {os.getcwd()}')
    for script in scripts:
        if _dryrun:
            print(f'sbatch {fs.script_dir}/{script}')
        else:
            os.system(f'sbatch {fs.script_dir}/{script}')

    # Distributed
    os.chdir(f'{fs.build_dir}/implementation{system.optimal_implementation}_maxthreadsperblock{system.optimal_tpb}_distributedTrue')
    print(f'cd {os.getcwd()}')
    for script in scripts:
        if _dryrun:
            print(f'sbatch {fs.script_dir}/{script}')
        else:
            os.system(f'sbatch {fs.script_dir}/{script}')

def run_benchmarks(fs, run_benchmarks):

    if 'microbenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_microbenchmarks(fs)
    if 'devicebenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_devicebenchmarks(fs)
    if 'nodebenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_nodebenchmarks(fs)
    if 'strongscalingbenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_strongscalingbenchmarks(system, fs)
    if 'weakscalingbenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_weakscalingbenchmarks(system, fs)
    if 'ioscalingbenchmarks' in run_benchmarks or 'all' in run_benchmarks:
        run_ioscalingbenchmarks(system, fs)

def genbenchmarks(system, fs, do_compile, nx, ny, nz, max_devices):

    # Create batch scripts
    os.chdir(fs.script_dir)
    print(f'cd {os.getcwd()}')
    gen_microbenchmarks(system, fs)

    gen_devicebenchmarks(system, fs, nx, ny, nz)
    gen_nodebenchmarks(system, fs, nx, ny, nz)

    gen_strongscalingbenchmarks(system, fs, nx, ny, nz, max_devices)
    gen_weakscalingbenchmarks(system, fs, nx, ny, nz, max_devices)
    gen_iobenchmarks(system, fs, nx, ny, nz, max_devices)
    genbuilds(fs, do_compile)

# pip3 install --user pandas numpy
import pandas as pd
def postprocess(system, fs, nx, ny, nz):
    os.chdir(fs.base_dir)
    print(f'cd {os.getcwd()}')

    with open(f'microbenchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('usesmem,maxthreadsperblock,problemsize,workingsetsize,milliseconds,bandwidth')
    os.system(f'cat {fs.build_dir}/*/microbenchmark.csv >> microbenchmark.csv')

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'bandwidth-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'bandwidth-smem-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'workingset-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'workingset-smem-{system.id}.csv', index=False)

    # Device
    with open(f'device-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    os.system(f'cat {fs.build_dir}/*/device-benchmark.csv >> device-benchmark.csv')

    df = pd.read_csv('device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 1)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'implicit-{system.id}.csv', index=False)

    df = pd.read_csv('device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 2)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'explicit-{system.id}.csv', index=False)

    # Node
    with open(f'node-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    os.system(f'cat {fs.build_dir}/*/node-benchmark.csv >> node-benchmark.csv')

    df = pd.read_csv('node-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 1)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'node-implicit-{system.id}.csv', index=False)

    df = pd.read_csv('node-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 2)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'node-explicit-{system.id}.csv', index=False)

    # Scaling
    with open(f'scaling-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('devices,millisecondsmin,milliseconds50thpercentile,milliseconds90thpercentile,millisecondsmax,usedistributedcommunication,nx,ny,nz,dostrongscaling')
    os.system(f'cat {fs.build_dir}/*/scaling-benchmark.csv >> scaling-benchmark.csv')

    df = pd.read_csv('scaling-benchmark.csv', comment='#')
    df = df.loc[(df['nx'] == nx) & (df['ny'] == ny) & (df['nz'] == nz)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'scaling-strong-{system.id}.csv', index=False)

    nn = nx * ny * nz
    df = pd.read_csv('scaling-benchmark.csv', comment='#')
    df = df.loc[(df['nx'] * df['ny'] * df['nz']) / df['devices'] == nn]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'scaling-weak-{system.id}.csv', index=False)

    # IO scaling
    with open(f'scaling-io-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print(f'devices,writemilliseconds,writebandwidth,readmilliseconds,readbandwidth,usedistributedio,nx,ny,nz')
    os.system(f'cat {fs.build_dir}/*/scaling-io-benchmark.csv >> scaling-io-benchmark.csv')

    # Collective
    df = pd.read_csv('scaling-io-benchmark.csv', comment='#')
    df = df.loc[(df['usedistributedio'] == 0)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'scaling-io-collective-{system.id}.csv', index=False)

    # Distributed
    df = pd.read_csv('scaling-io-benchmark.csv', comment='#')
    df = df.loc[(df['usedistributedio'] == 1)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'scaling-io-distributed-{system.id}.csv', index=False)

# Systems
mahti = System(id='a100', account='project_2000403', partition='gpusmall', ngpus_per_node=4, gres='gpu:a100',
               modules='module load gcc/9.4.0 openmpi/4.1.2-cuda cuda cmake', use_hip=False, optimal_implementation='1', optimal_tpb='0')
puhti = System(id='v100', account='project_2000403', partition='gpu', ngpus_per_node=4,
               gres='gpu:v100', modules='module load gcc cuda openmpi cmake', use_hip=False,
               additional_commands='''
export UCX_RNDV_THRESH=16384
export UCX_RNDV_SCHEME=get_zcopy
export UCX_MAX_RNDV_RAILS=1''', optimal_implementation='1', optimal_tpb='0')
triton = System(id='mi100', account='', partition='gpu-amd', ngpus_per_node=1, gres='',
                modules='module load gcc bison flex cmake openmpi', use_hip=True, optimal_implementation='1', optimal_tpb='512')
lumi = System(id='mi250x', account='project_462000120', partition='pilot', ngpus_per_node=8, gres='gpu', modules='''
        module load CrayEnv
        module load PrgEnv-cray
        module load craype-accel-amd-gfx90a
        module load rocm
        module load buildtools
        module load cray-python
        ''', use_hip=True, optimal_implementation='1', optimal_tpb='512')

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

# Parse args
parser = argparse.ArgumentParser(description='A tool for generating benchmarks')

parser.add_argument('--build', action='store_true', help='Build benchmark directories')
parser.add_argument('--cmakelistdir', default='.', type=str, help='Directory containing the project CMakeLists.txt')
parser.add_argument('--dryrun', action='store_true', help='Do a dryrun without compiling or running. Prints commands to stdout.')
parser.add_argument('--account', type=str, help='Set the account to be used in the runs')
parser.add_argument('--partition', type=str, help='Set the partition that should be used for computations')
parser.add_argument('--postprocess', action='store_true', help='Postprocess the benchmark outputs')
parser.add_argument('--run', type=str, nargs='+', help='[microbenchmarks devicebenchmarks nodebenchmarks strongscalingbenchmarks weakscalingbenchmarks ioscalingbenchmarks all]')
parser.add_argument('--dims', type=int, default=[64, 64, 64], nargs=3, help='The dimensions of the computational domain')
parser.add_argument('--max-devices', type=int, default=4096, help='The maximum number of devices used in the benchmarks')

args = parser.parse_args()

# Toggle dryrun
_dryrun = args.dryrun

# Create the file structure
fs = FileStructure(args.cmakelistdir)

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

# Set problem scale
max_devices = args.max_devices

# Compile
if args.run:
    args.build = True
genbenchmarks(system, fs, args.build, nx, ny, nz, max_devices)

# Run
if args.run:
    run_benchmarks(fs, args.run)

# Postprocess
if args.postprocess:
    postprocess(system, fs, nx, ny, nz)
