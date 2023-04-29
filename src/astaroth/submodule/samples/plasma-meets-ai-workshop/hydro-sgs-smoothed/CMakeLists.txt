set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)

set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/hydro-sgs-smoothed CACHE BOOL "" FORCE)
add_executable(hydro-sgs-smoothed hydro-sgs-smoothed.c)
target_link_libraries(hydro-sgs-smoothed astaroth_core astaroth_utils)