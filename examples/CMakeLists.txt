set(EXAMPLE_UTILITIES "${CMAKE_CURRENT_SOURCE_DIR}/utils")

add_subdirectory(sqpnp)

if (BUILD_TIMING_EXAMPLE)
	add_subdirectory(timing)
endif()