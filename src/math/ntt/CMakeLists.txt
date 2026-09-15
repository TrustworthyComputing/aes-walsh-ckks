add_library(ntt_lib
    ntt.cpp
)

target_link_libraries(ntt_lib
    PUBLIC
        modarith_lib
        util_lib
)

target_include_directories(ntt_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
