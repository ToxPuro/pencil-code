add_executable(genbenchmarkscripts main.c)

add_custom_command(
  TARGET genbenchmarkscripts POST_BUILD
  COMMAND genbenchmarkscripts
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Generating benchmark scripts"
)
