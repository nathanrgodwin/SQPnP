add_executable(SQPNP_timing main.cpp)

target_include_directories(SQPNP_timing
    PRIVATE
        ${EXAMPLE_UTILITIES})

target_link_libraries(SQPNP_timing
    PRIVATE
        CV_SQPNP
        SQPnP)
