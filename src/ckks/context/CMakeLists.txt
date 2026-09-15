add_library(ckks_context_lib
    ckks_context.cpp
    conjugation_key_generation.cpp
    secret_key_generation.cpp
    relinearization_key_generation.cpp
    rotation_key_generation.cpp
    ../operations/bootstrap/ckks_bootstrap_dft.cpp
)

target_include_directories(ckks_context_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../operations/bootstrap
)

target_link_libraries(ckks_context_lib
    PUBLIC
        ckks_encoding_lib
        ckks_params_lib
        math_lib
        fft_lib
        OpenMP::OpenMP_CXX
)
