add_subdirectory(ciphertext)
add_subdirectory(context)
add_subdirectory(encoding)
add_subdirectory(operations)
add_subdirectory(parameters)

add_library(ckks_lib INTERFACE)

target_link_libraries(ckks_lib
    INTERFACE
        ckks_ciphertext_lib
        ckks_context_lib
        ckks_encoding_lib
        ckks_operations_lib
        ckks_bootstrap_lib
        ckks_params_lib
)
