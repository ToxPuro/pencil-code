#!/bin/bash
if [ -z $AC_HOME ]
then
       echo "ASTAROTH_HOME environment variable not set, run \"source ./sourceme.sh\" in Astaroth home directory"
       exit 1
fi

# Exit if any of the following commands fail
set -e

TIARA_SETUP_DEFAULT=""
DOUBLE_DEFAULT="OFF"
DEBUG_MODE_DEFAULT="OFF"
BUILD_DIR_DEFAULT=${AC_HOME}/build/
ALTER_CONF_DEFAULT="OFF"

BUILD_DIR=${BUILD_DIR_DEFAULT}
TIARA_SETUP=${TIARA_SETUP_DEFAULT}
DOUBLE=${DOUBLE_DEFAULT}
DEBUG_MODE=${DEBUG_MODE_DEFAULT}
ALTER_CONF=${ALTER_CONF_DEFAULT}

while [ "$#" -gt 0 ]
do
	case $1 in
		-h|--help)
			echo "You can set up a build directory separe of the ASTAROTH_HOME"
			echo "Available flags:"
			echo "-b, --buildir [PATH] : Set build directory"
			echo "-t,--tiara : Use TIARA cluster setting for cmake"
			echo "-d, --double : Compile with double precision"
			echo "-e, --debug: : Compile in debug mode"
			echo "Example:"
			echo "ac_mkbuilddir.sh -b my_build_dir/"
			exit 0
			;;
		-b|--buildir)
			shift
                        BUILD_DIR=${1}
			shift
                        echo "Setting up build directory..."
			ALTER_CONF="ON"
			;;
		-t|--tiara)
			shift
                        TIARA_SETUP="-D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icpc"
                        echo "Using TIARA cluster compiler settings"
			;;
		-d|--double)
			shift
                        DOUBLE="ON"
                        echo "Double precision"
			;;
		-e|--debug)
			shift
                        DEBUG_MODE="ON"
                        echo "Debug mode compilation"
			;;
		*)
			break
	esac
done

echo "Creating build directory: ${BUILD_DIR}"

mkdir -p ${BUILD_DIR}

cd ${BUILD_DIR}

#Set up the astaroth.conf to be define and customized in the build directory to
#not always alter the default use i.e. for unit test etc.
#Assumed by default if you do this thing anyway.
echo "cp ${AC_HOME}/config/astaroth.conf ${PWD}"
cp ${AC_HOME}/config/astaroth.conf .

CONF_DIR="-D ASTAROTH_CONF_PATH=${PWD}"

#cmake -D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icpc -DDOUBLE_PRECISION=OFF -DBUILD_DEBUG=OFF ${AC_HOME}

echo "cmake ${TIARA_SETUP} ${CONF_DIR} -DDOUBLE_PRECISION=${DOUBLE} -DBUILD_DEBUG=${DEBUG_MODE} -DALTER_CONF=${ALTER_CONF} ${AC_HOME}"

cmake ${TIARA_SETUP} ${CONF_DIR} -DDOUBLE_PRECISION=${DOUBLE} -DBUILD_DEBUG=${DEBUG_MODE} -DALTER_CONF=${ALTER_CONF} ${AC_HOME}
