add_library(ckks_operations_lib
    arith/ckks_addition.cpp
    arith/ckks_mul.cpp
    arith/ckks_rescale.cpp
    circuits/aes/ckks_aes_ctr.cpp
    circuits/aes/ckks_aes_ctr_walsh_sbox.cpp
    circuits/ckks_walsh_sbox.cpp
    evalpoly/ckks_evalpoly.cpp
    keyswitch/ckks_conjugate.cpp
    keyswitch/ckks_keyswitch.cpp
    keyswitch/ckks_relin.cpp
    keyswitch/ckks_rotation.cpp
)

target_include_directories(ckks_operations_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/arith
        ${CMAKE_CURRENT_SOURCE_DIR}/circuits
        ${CMAKE_CURRENT_SOURCE_DIR}/circuits/aes
        ${CMAKE_CURRENT_SOURCE_DIR}/evalpoly
        ${CMAKE_CURRENT_SOURCE_DIR}/keyswitch
)

target_link_libraries(ckks_operations_lib
    PUBLIC
        ckks_ciphertext_lib
        ckks_context_lib
        math_lib
        OpenMP::OpenMP_CXX
)

add_subdirectory(bootstrap)
