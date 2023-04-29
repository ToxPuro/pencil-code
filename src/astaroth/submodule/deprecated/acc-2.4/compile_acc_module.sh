#!/bin/bash
ACC_DIR=$(readlink -f $(dirname $0)/../acc)
ACC_BINARY_DIR=$(pwd)/acc
ACC_STDLIB_DIR=${ACC_DIR}/stdlib/
MODULE_DIR=$(readlink -f $1)

echo "-- ACC binary dir:  ${ACC_BINARY_DIR}"
echo "-- ACC stdlib dir:  ${ACC_STDLIB_DIR}"
echo "-- ACC project dir: ${MODULE_DIR}"
for source in ${MODULE_DIR}/*.ac
do
    ${ACC_DIR}/preprocess.sh $source -I ${MODULE_DIR} -I ${ACC_STDLIB_DIR}
    ${ACC_BINARY_DIR}/acc < $(basename -- "$source").preprocessed
done
