add_executable(bwtest bwtest.cu)
add_executable(bwtest-stencil bwtest-stencil.cu)
add_executable(bwtest-benchmark bwtest-benchmark.cu)
add_executable(bwtest-io bwtest-io.cu)

if (USE_HIP)
set_source_files_properties(bwtest.cu bwtest-stencil.cu bwtest-benchmark.cu bwtest-io.cu PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(bwtest acc-runtime-headers)
target_link_libraries(bwtest-stencil acc-runtime-headers)

# MPI
if (MPI_ENABLED)
    target_link_libraries(bwtest-io MPI::MPI_C)
endif()

# OpenMP
find_package(OpenMP)
target_link_libraries(bwtest-io OpenMP::OpenMP_C)

target_link_libraries(bwtest-benchmark acc-runtime-headers) # temporarily for rocprof
target_include_directories(bwtest-benchmark PRIVATE ${CMAKE_SOURCE_DIR}/acc-runtime/api)
target_include_directories(bwtest-io PRIVATE ${CMAKE_SOURCE_DIR}/acc-runtime/api)

if(USE_SMEM)
    set(USE_SMEM 1)
else()
    set(USE_SMEM 0)
endif()

if(NOT DEFINED MAX_THREADS_PER_BLOCK)
    set(MAX_THREADS_PER_BLOCK 0)
endif()

## Options
# cmake -DUSE_SMEM=1 .. to use smem in bwtest-benchmark
target_compile_options(bwtest-benchmark PRIVATE -DUSE_SMEM=${USE_SMEM})
# cmake -DMAX_THREADS_PER_BLOCK=192 .. to set launch bounds in bwtest-benchmark
target_compile_options(bwtest-benchmark PRIVATE -DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})
