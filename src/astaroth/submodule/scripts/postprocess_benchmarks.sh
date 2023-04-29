#!/bin/bash

OUTPUT=results.csv
rm -i $OUTPUT

# $1 input dir
process_input() {
    echo $1
    #cat $1/*.csv | sort -n
    cat $1/*.csv | sort -k1n -k3n | awk '!a[$1]++'
    echo ""
} >> $OUTPUT

process_input "benchmark_decomp_1D"
process_input "benchmark_decomp_2D"
process_input "benchmark_decomp_3D"
process_input "benchmark_decomp_1D_comm"
process_input "benchmark_decomp_2D_comm"
process_input "benchmark_decomp_3D_comm"

process_input "benchmark_meshsize_256"
process_input "benchmark_meshsize_512"
process_input "benchmark_meshsize_1024"
process_input "benchmark_meshsize_2048"

process_input "benchmark_stencilord_2"
process_input "benchmark_stencilord_4"
process_input "benchmark_stencilord_6"
process_input "benchmark_stencilord_8"

process_input "benchmark_timings_control"
process_input "benchmark_timings_comp"
process_input "benchmark_timings_comm"
process_input "benchmark_timings_default"
process_input "benchmark_timings_corners"

process_input "benchmark_weak_128"
process_input "benchmark_weak_256"
process_input "benchmark_weak_512"

cat $OUTPUT
