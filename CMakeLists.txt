cmake_minimum_required(VERSION 3.18)
project(walsh-aes LANGUAGES CXX)
find_package(OpenMP REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WALSH_AES_ENABLE_NATIVE_ARCH "Use -march=native for optimized builds" ON)

# Apply baseline optimization flags to all C++ targets.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
if(WALSH_AES_ENABLE_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

add_subdirectory(src/math)
add_subdirectory(src/ckks)
add_subdirectory(src/util)
add_subdirectory(bench)

target_compile_options(ntt_lib PRIVATE -O3)
target_compile_options(modarith_lib PRIVATE -O3)
if(WALSH_AES_ENABLE_NATIVE_ARCH)
    target_compile_options(modarith_lib PRIVATE -march=native)
endif()
