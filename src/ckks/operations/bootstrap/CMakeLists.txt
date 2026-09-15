add_library(ckks_bootstrap_lib
    ckks_bootstrap_evalmod_inplace.cpp
    ckks_c2s_generalized.cpp
    ckks_linear_transform.cpp
    ckks_modraise.cpp
    ckks_s2c_generalized.cpp
)

target_include_directories(ckks_bootstrap_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ckks_bootstrap_lib
    PUBLIC
        ckks_operations_lib
        ckks_encoding_lib
        OpenMP::OpenMP_CXX
)
