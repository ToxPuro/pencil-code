include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)

    add_executable(fortrantest main.f90)
    target_link_libraries(fortrantest astaroth_core)
else()
    message(STATUS "Fortran compiler not found, skipping samples/fortrantest")
endif()

