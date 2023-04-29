##############################################
##  CMakeLists.txt for Astaroth Standalone  ##
##############################################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Files
file (GLOB SOURCES "*.cc" "model/*.cc")

## Find packages
find_package(OpenMP REQUIRED)
if (BUILD_RT_VISUALIZATION)
    add_definitions(-DAC_BUILD_RT_VISUALIZATION=1)
    # SDL 2
    set(SDL2_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include/)
    set(SDL2_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/build/)
    set(SDL2_LIBRARY "SDL2")
    include_directories(${SDL2_INCLUDE_DIR})
    link_directories(${SDL2_LIBRARY_DIR})
endif ()

# Compile and link
add_library(astaroth_standalone STATIC ${SOURCES})
target_link_libraries(astaroth_standalone PRIVATE OpenMP::OpenMP_CXX astaroth_core ${SDL2_LIBRARY})
target_compile_options(astaroth_standalone PRIVATE -pipe -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

add_executable(ac_run main.cc)
target_link_libraries(ac_run PRIVATE astaroth_standalone)
