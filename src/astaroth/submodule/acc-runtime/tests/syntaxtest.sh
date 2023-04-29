#!/bin/bash

ACC=./acc
if [ ! -f "$ACC" ]; then
    echo "ERROR: $ACC not found in the current working directory. Please build ACC and rerun the script in the build directory."
    exit 1
fi


DIR="$(dirname "$(readlink -f "$0")")"
for filename in $DIR/*.ac; do
  if ! $ACC "$filename";
  then
    echo "FAILURE when processing $filename"
    exit 1
  fi
  echo "$filename OK"
done
echo "SUCCESS"
exit 0
