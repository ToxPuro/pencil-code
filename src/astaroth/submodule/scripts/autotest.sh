#!/bin/bash

# Usage:
# cd astaroth
# source ./sourceme.sh
# autotest.sh
#
# If you need slurm or to pass something before ./ac_run, export the variable
# SRUN_COMMAND before calling this script.
#
# F.ex. on Taito SRUN_COMMAND="srun --gres=gpu:k80:4 --mem=24000 -t 00:14:59 -p gputest --cpus-per-task 1 -n 1"
# export SRUN_COMMAND
# autotest.sh
echo "SRUN_COMMAND: " ${SRUN_COMMAND}

results=()

# Parameters
# $1: String testname
# $2: String stencil_define_file
# $3: String stencil_assembly_file
# $4: String stencil_process_file
test_solver() {
    TEST_DIR="tests/"$1
    ac_mkbuilddir.sh -b ${TEST_DIR}
    cd ${TEST_DIR}
    compile_acc.sh --header $2 --assembly $3 --process $4
    make -j
    ${SRUN_COMMAND} ./ac_run -t
    results+=(${TEST_DIR}" fail? "$?)
}

NAME="hydro"
HEADER="hydro_solver/stencil_defines.h"
ASSEMBLY="mhd_solver/stencil_assembly.sas"
PROCESS="mhd_solver/stencil_process.sps"
test_solver ${NAME} ${HEADER} ${ASSEMBLY} ${PROCESS}

NAME="magnetic"
HEADER="magnetic_solver/stencil_defines.h"
ASSEMBLY="mhd_solver/stencil_assembly.sas"
PROCESS="mhd_solver/stencil_process.sps"
test_solver ${NAME} ${HEADER} ${ASSEMBLY} ${PROCESS}

NAME="mhd"
HEADER="mhd_solver/stencil_defines.h"
ASSEMBLY="mhd_solver/stencil_assembly.sas"
PROCESS="mhd_solver/stencil_process.sps"
test_solver ${NAME} ${HEADER} ${ASSEMBLY} ${PROCESS}

# Print results
for i in "${results[@]}"; do echo "$i" ; done
