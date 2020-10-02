cmake_minimum_required(VERSION 3.10)

project(SQPNP VERSION 1.0.0)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Eigen3 REQUIRED)

option(BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(sqpnp)

if (BUILD_EXAMPLES)
	option(BUILD_TIMING_EXAMPLE "Build timing example" ON)
	if (BUILD_TIMING_EXAMPLE)
		add_subdirectory(3rdparty/sqpnp)
		set_target_properties(SQPnP PROPERTIES 
			FOLDER "3rdparty")
		set_target_properties(sqpnp_example PROPERTIES 
			FOLDER "3rdparty")
	endif()
	add_subdirectory(examples)
endif()