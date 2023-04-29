set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)

set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/hydro CACHE BOOL "" FORCE)
add_executable(hydro hydro.c)
target_link_libraries(hydro astaroth_core astaroth_utils)
