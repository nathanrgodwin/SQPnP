add_executable(SQPNP_example main.cpp)

target_include_directories(SQPNP_example
    PRIVATE
        ${EXAMPLE_UTILITIES})

target_link_libraries(SQPNP_example
    PRIVATE
        CV_SQPNP)
