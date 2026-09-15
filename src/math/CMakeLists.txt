add_subdirectory(modarith)
add_subdirectory(ntt)
add_subdirectory(fft)

add_library(math_lib 
    aes/aes.cpp
    binboot_k_bound.cpp
    poly_approx.cpp
    polynomial.cpp
)

target_link_libraries(math_lib PUBLIC
    util_lib
    modarith_lib
    ntt_lib
    fft_lib
    mpfr
    gmp
)

# Make math headers visible to dependents
target_include_directories(math_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
