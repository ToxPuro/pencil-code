## CMake settings
cmake_minimum_required(VERSION 3.19)

## Project options
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" ON)
if (USE_HIP)
    find_program(CMAKE_HIP_COMPILER hipcc REQUIRED)
    set(CMAKE_C_COMPILER ${CMAKE_HIP_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_HIP_COMPILER})

    # Rocm
    #include_directories(/opt/rocm/include)
    #link_directories(/opt/rocm/lib)
    
    # Roctracer
    include_directories(/opt/rocm/roctracer/include)
    link_directories(/opt/rocm/roctracer/lib)

    # Rocrand
    include_directories(/opt/rocm/hiprand/include)
    include_directories(/opt/rocm/rocrand/include)
    link_directories(/opt/rocm/hiprand/lib)
    link_directories(/opt/rocm/rocrand/lib)
endif()

## Project settings
project(acc-runtime)

## Project-wide compilation flags
#set(COMMON_FLAGS "-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")
set(COMMON_FLAGS "-Wall -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")

## Include directories
include_directories(.)

## Compile options
if (NOT DEFINED IMPLEMENTATION)
    set(IMPLEMENTATION 1)
endif()
if (NOT DEFINED MAX_THREADS_PER_BLOCK)
    set(MAX_THREADS_PER_BLOCK 0)
endif()
add_compile_options(-DIMPLEMENTATION=${IMPLEMENTATION})
add_compile_options(-DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})
if (USE_HIP)
add_compile_options(-DAC_USE_HIP=1)
add_compile_definitions(GPU_API_INCLUDES="/opt/rocm/include -I /opt/rocm/roctracer/include ")
else()
add_compile_definitions(GPU_API_INCLUDES="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)

