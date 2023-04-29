set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)

set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/mhd-les CACHE BOOL "" FORCE)
add_executable(mhd-les mhd-les.c)
target_link_libraries(mhd-les astaroth_core astaroth_utils)
