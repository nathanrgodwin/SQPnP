

add_library(SQPNP src/solvepnp.cpp src/solvesqp.cpp "src/solvesqp.h")

target_include_directories(SQPNP PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/include/ 
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/src)