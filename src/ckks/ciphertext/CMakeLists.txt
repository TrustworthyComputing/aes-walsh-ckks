add_library(ckks_ciphertext_lib
    ckks_ciphertext.cpp
)

target_include_directories(ckks_ciphertext_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ckks_ciphertext_lib
    PUBLIC
        util_lib
)
