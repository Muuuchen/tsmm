cmake_minimum_required(VERSION 3.12)
project(tsmm
  VERSION 1.0.0
  LANGUAGES CXX
  DESCRIPTION "Tensor SIMD Matrix Multiplication"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -mkl -O3")
option(TSMM_ENABLE_AVX512 "Enable AVX512 instructions" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


add_executable(tsmm)
target_sources(tsmm
  PRIVATE
    src/tsmm.cpp
)
add_executable(bench_blas)
target_sources(bench_blas
  PRIVATE
    src/bench.cpp  
)
target_link_directories(
  bench_blas
  PRIVATE
  /opt/intel/oneapi/mkl/latest/lib/
)

target_link_libraries(bench_blas
  PRIVATE
  mkl_intel_lp64
  mkl_sequential
  mkl_core
)
target_include_directories(tsmm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_compile_options(tsmm
  PRIVATE
    -O3
    -g
    -march=native
    $<$<BOOL:${TSMM_ENABLE_AVX512}>:-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl>
)



# 安装规则
install(TARGETS tsmm
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
