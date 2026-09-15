add_executable(aes_bench aes_bench.cpp)

# The bootstrap and circuit archives call into each other. Repeating the pair
# keeps the static-library link portable to the CMake 3.18 minimum.
target_link_libraries(
    aes_bench
    PRIVATE
        ckks_lib
        ckks_bootstrap_lib
        ckks_operations_lib
        ckks_bootstrap_lib
        OpenMP::OpenMP_CXX
)
