add_library(modarith_lib
    modarith.cpp
)

target_include_directories(modarith_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/math
)
