set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)

set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/blur CACHE BOOL "" FORCE)
add_executable(blur blur.c)
target_link_libraries(blur astaroth_core astaroth_utils)