#!/bin/python3
import os

# Set cmake, build dir, and srun based on the system
import socket
hostname = socket.gethostname()
if "mahti" in hostname or "puhti" in hostname or "uan" in hostname:
    build_dir='/users/pekkila/astaroth/build'
    if "mahti" in hostname:
        cmake='/users/pekkila/CMake/build/bin/cmake'
        srun='srun --account=project_2000403 --gres=gpu:a100:1 -t 00:14:59 -p gputest -n 1 -N 1'
    elif "puhti" in hostname:
        cmake='/users/pekkila/cmake/build/bin/cmake'
        srun='srun --account=project_2000403 --gres=gpu:v100:1 -t 00:14:59 -p gputest -n 1 -N 1'
    elif "uan" in hostname:
        build_dir='/pfs/lustrep1/users/pekkila/astaroth/build'
        cmake='cmake -DUSE_HIP=ON'
        srun='srun --account=project_462000120 --gres=gpu:1 -t 00:14:59 -p pilot -n 1 -N 1'
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
def benchmark_implementation(implementation=1):

    nn = 64
    max_threads_per_block = 1024
    tpb = 0

#    cmd = ""
    while tpb <= max_threads_per_block:
        #cmd += f'{cmake} -DIMPLEMENTATION={implementation} -DMAX_THREADS_PER_BLOCK={tpb} .. &&'
        #cmd += 'make -j &&'
        #cmd += f'./benchmark-device {nn} {nn} {nn} ;'
        os.system(f'{cmake} -DIMPLEMENTATION={implementation} -DMAX_THREADS_PER_BLOCK={tpb} .. && make -j')
        os.system(f'{srun} ./benchmark-device {nn} {nn} {nn}')
        if (tpb == 0):
            tpb = 32
        else:
            tpb *= 2

def benchmark_implementations(out_file='implementation'):
    max_implementations = 2

    os.system('echo "implementation,maxthreadsperblock,milliseconds" > device-benchmark.csv')
    for i in range(1, max_implementations+1):
        benchmark_implementation(i)
    os.system(f'mv device-benchmark.csv {out_file}.csv')

#benchmark_implementation(1, "implementation-1")
#benchmark_implementation(2, "implementation-2")
benchmark_implementations()

# Profile
def profile(implementation=1, max_threads_per_block=0, nn=128):
    with open('metrics.txt', 'w') as f:
        if False:
            f.write(
            '''
            # Perf counters group 1
            pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
            # Perf counters group 2
            pmc : TCC_HIT_sum, TCC_MISS_sum TCC_EA_RDREQ_sum TA_BUSY_avr
            # Perf counters group 3
            pmc: L2CacheHit MemUnitBusy LDSBankConflict
            # Perf counters group 4 (smem)
            pmc: GDSInsts SQ_INSTS_GDS FetchSize

            # Filter by dispatches range, GPU index and kernel names
            # supported range formats: "3:9", "3:", "3"
            #range: 0 : 160
            #gpu: 0 1 2 3
            gpu: 0
            #kernel: singlepass_solve
            ''')
        else:
            f.write(
                '''
                # Perf counters group 1
                pmc : VALUBusy SALUBusy MemUnitBusy

                # Filter by dispatches range, GPU index and kernel names
                # supported range formats: "3:9", "3:", "3"
                #range: 0 : 16
                #gpu: 0 1 2 3
                gpu: 0
                #kernel: singlepass_solve
                '''
            )
        

    cmd = ""
    cmd += f'{cmake} -DIMPLEMENTATION={implementation} -DMAX_THREADS_PER_BLOCK={max_threads_per_block} .. && '
    cmd += 'make -j && '
    cmd += f'rocprof --trace-start off -i metrics.txt ./benchmark-device {nn} {nn} {nn} ;'
    os.system(f'{srun} /bin/bash -c \"{cmd}\"')
    os.system(f'mv metrics.csv metrics-{implementation}-{max_threads_per_block}.csv')

#profile(implementation=1, max_threads_per_block=512, nn=256)
#profile(implementation=2, max_threads_per_block=512, nn=256)
