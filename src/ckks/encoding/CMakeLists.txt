add_library(ckks_encoding_lib
    ckks_encoding.cpp
)

target_include_directories(ckks_encoding_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ckks_encoding_lib
    PUBLIC
        ckks_ciphertext_lib
        ckks_context_lib
        fft_lib
        OpenMP::OpenMP_CXX
)
