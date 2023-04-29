set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)

set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/hydro-sgs CACHE BOOL "" FORCE)
add_executable(hydro-sgs hydro-sgs.c)
target_link_libraries(hydro-sgs astaroth_core astaroth_utils)