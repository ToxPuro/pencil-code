## Astaroth Utils
# add_library(modelsolver modelsolver.cc)
# target_link_libraries(modelsolver astaroth_core)
if (BUILD_SHARED)
  add_library(astaroth_utils SHARED config_loader.c memory.c verification.c modelsolver.c modelreduce.c logging.c)
else()
  add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c modelreduce.c logging.c)
endif()

target_link_libraries(astaroth_utils astaroth_core)
# add_library(modelsolver modelsolver.cc) # Change to modelsolver-legacy-do-not-modify.c and enable -mavx below to use the pure-C model solution used since at least the Vaisala 2021 paper
# target_link_libraries(modelsolver astaroth_core)
#target_compile_options(modelsolver PRIVATE "-mavx")
target_compile_options(astaroth_utils PRIVATE "-mavx")
