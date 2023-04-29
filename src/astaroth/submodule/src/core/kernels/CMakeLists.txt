## Astaroth Kernels
if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CXX)
endif()

if (BUILD_SHARED)
  add_library(kernels SHARED kernels.cu)
else()
  add_library(kernels STATIC kernels.cu)
endif()

if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

target_link_libraries(kernels PUBLIC acc-runtime-headers)
target_compile_features(kernels PRIVATE cxx_std_17)
