add_library(ckks_params_lib
    ckks_params.cpp
)

target_include_directories(ckks_params_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ckks_params_lib
    PUBLIC
        ckks_context_lib
)
