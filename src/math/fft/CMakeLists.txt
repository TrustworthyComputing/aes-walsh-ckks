add_library(fft_lib
    fft.cpp
)

target_link_libraries(fft_lib
    PUBLIC
        util_lib
        modarith_lib
)

target_include_directories(fft_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
