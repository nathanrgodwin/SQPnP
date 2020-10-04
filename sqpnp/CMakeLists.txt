add_library(CV_SQPNP src/sqpnp.cpp)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

target_include_directories(CV_SQPNP PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include/
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR})

target_link_libraries(CV_SQPNP PUBLIC
	${OpenCV_LIBS})