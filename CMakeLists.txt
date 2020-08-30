cmake_minimum_required(VERSION 3.10)

project(SQPNP_PROJ VERSION 1.0.0)

find_package(Eigen3 REQUIRED)

option(BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(sqpnp)
if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()