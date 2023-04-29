#!/bin/bash
if [[ $1 == "DO" && $2 == "IT!" ]]; then
    find -name \*.h -o -name \*.cc -o -name \*.c | xargs clang-format -i -style=file
    echo "It is done."
else
    find -name \*.h -o -name \*.cc -o -name \*.c
    echo "I'm going to try to fix the style of these files."
    echo "If you're absolutely sure, give \"DO IT!\" as a parameter."
fi
