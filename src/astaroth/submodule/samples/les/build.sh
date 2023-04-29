#!/bin/bash
cmake="/m/home/home6/61/pekkilj1/unix/repositories/cmake/build/bin/cmake"
$cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=samples/les -DDSL_MODULE_DIR=../samples/les .. && make -j
#$cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=samples/les -DDSL_MODULE_DIR=../samples/les .. && make -j