#!/bin/bash
# Preprocesses the give file using GCC. This script is usually automatically called in
# ./compile.sh, but may be called also individually for debugging purposes.
cat $1 | gcc ${@:2} -x c -E - | sed "s/#.*//g" > $(basename -- "$1").preprocessed  #| tee ${PWD}/$(basename -- "$1").preprocessed
