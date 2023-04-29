#!/bin/bash

shopt -s extglob
rm -vr !("astaroth.conf"|"clean_rundir.sh"|"my_cmake.sh"|"README.md"|"a2_timeseries.ts")

