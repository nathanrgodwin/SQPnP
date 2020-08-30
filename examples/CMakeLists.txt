add_executable(SQPNP_Example main.cpp ../sqpnp/src/solvepnp.cpp ../sqpnp/src/solvesqp.cpp)

target_include_directories(SQPNP_Example 
	PRIVATE 
	${CMAKE_SOURCE_DIR}/sqpnp/include
	${EIGEN3_INCLUDE_DIR})

