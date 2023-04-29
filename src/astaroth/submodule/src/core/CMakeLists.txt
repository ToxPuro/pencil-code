## Astaroth Core
if (BUILD_SHARED)
  add_library(astaroth_core SHARED device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)
else()
  add_library(astaroth_core STATIC device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)
endif()

if (USE_HIP)
    find_package(hip)
    target_link_libraries(astaroth_core kernels hip::host)
else()
    find_package(CUDAToolkit)
    target_link_libraries(astaroth_core kernels CUDA::cudart CUDA::cuda_driver)
endif()

## Definitions
if (MULTIGPU_ENABLED)
    add_definitions(-DAC_MULTIGPU_ENABLED=1)
else ()
    add_definitions(-DAC_MULTIGPU_ENABLED=0)
endif ()

if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()
if (USE_PERFSTUBS)
    find_package(Threads)
    # Would like to define scope here, but the other target_link_libraries are plain...
    target_link_libraries(astaroth_core perfstubs Threads::Threads)
endif()

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_C)
endif()


