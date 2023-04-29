## Compile the Astaroth Code compiler
find_package(BISON)
find_package(FLEX 2.5.5 REQUIRED)

bison_target(acc_parser acc.y ${CMAKE_CURRENT_BINARY_DIR}/acc.tab.c) # COMPILE_FLAGS -Wcounterexamples)
flex_target(acc_lexer acc.l ${CMAKE_CURRENT_BINARY_DIR}/acc.yy.c)
add_flex_bison_dependency(acc_lexer acc_parser)
add_compile_options(-mavx -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

add_executable(acc code_generator.c ${BISON_acc_parser_OUTPUTS} ${FLEX_acc_lexer_OUTPUTS})
target_include_directories(acc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
