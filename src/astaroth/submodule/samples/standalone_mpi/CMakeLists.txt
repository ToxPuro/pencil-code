add_executable(
ac_run_mpi
	main.cc
	host_memory.cc
	host_forcing.cc
	config_loader.cc
	simulation_rng.cc
)

target_include_directories(
ac_run_mpi
PRIVATE
	include
)

target_link_libraries(
ac_run_mpi
	astaroth_utils
	astaroth_core
)
