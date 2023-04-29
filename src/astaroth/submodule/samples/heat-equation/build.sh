#!/usr/bin/bash
if [[ -z "${ASTAROTH}" ]]; then
    echo "Environment variable ASTAROTH was not set"
    echo "Please 'export ASTAROTH=<path to Astaroth dir>' and run again"
    exit 1
fi
cmake -DBUILD_MHD_SAMPLES=OFF -DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=$ASTAROTH/samples/heat-equation/ -DPROGRAM_MODULE_DIR=$ASTAROTH/samples/heat-equation $@ $ASTAROTH && make -j