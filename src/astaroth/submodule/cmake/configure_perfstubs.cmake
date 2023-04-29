if (USE_PERFSTUBS)
	if (USE_VENDORED_PERFSTUBS)
		FetchContent_Declare(
			perfstubs
			SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/perfstubs
			BINARY_DIR ${PROJECT_BINARY_DIR}/3rdparty/perfstubs
		)
		FetchContent_MakeAvailable(perfstubs)
	else()
		find_package(perfstubs)
	endif()
endif()
