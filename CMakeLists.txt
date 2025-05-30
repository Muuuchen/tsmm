# cmake_minimum_required(VERSION 3.12)
# project(tsmm
#   VERSION 1.0.0
#   LANGUAGES CXX
#   DESCRIPTION "Tensor SIMD Matrix Multiplication"
# )

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
# option(TSMM_ENABLE_AVX512 "Enable AVX512 instructions" ON)
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(OpenMP REQUIRED)


# find_package(MKL REQUIRED)
# if(MKL_FOUND)
#     message(STATUS "MKL found: ${MKL_INCLUDE_DIRS}")
    

#     set(MKL_LIBRARIES "")
    

#     if(WIN32)
#         list(APPEND MKL_LIBRARIES mkl_intel_lp64_dll mkl_sequential_dll mkl_core_dll)
#     else()

#         if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#             list(APPEND MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)
#         else()

#             list(APPEND MKL_LIBRARIES mkl_intel_cilkrts mkl_sequential mkl_core)
#         endif()
        

#         list(APPEND MKL_LIBRARIES pthread m dl)
#     endif()

#     if(DEFINED ENV{MKLROOT})
#         link_directories($ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64)
#     endif()
# else()
#     message(FATAL_ERROR "MKL not found. Please install MKL or check your environment.")
# endif()

# add_executable(tsmm)
# target_sources(tsmm
#   PRIVATE
#     src/tsmm.cpp
# )
# add_executable(bench_blas)
# target_sources(bench_blas
#   PRIVATE
#     src/bench.cpp  
# )
# add_executable(tsmm_opt)
# target_sources(tsmm_opt
#   PRIVATE
#     src/tsmm_opt.cpp  
# )

# target_include_directories(tsmm
#     PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
#         ${MKL_INCLUDE_DIRS}
# )



cmake_minimum_required(VERSION 3.12)
project(tsmm
  VERSION 1.0.0
  LANGUAGES CXX
  DESCRIPTION "Tensor SIMD Matrix Multiplication"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
option(TSMM_ENABLE_AVX512 "Enable AVX512 instructions" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenMP REQUIRED)


# find_package(MKL REQUIRED)
# if(MKL_FOUND)
#     message(STATUS "MKL found: ${MKL_INCLUDE_DIRS}")
    

#     set(MKL_LIBRARIES "")
    

#     if(WIN32)
#         list(APPEND MKL_LIBRARIES mkl_intel_lp64_dll mkl_sequential_dll mkl_core_dll)
#     else()

#         if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#             list(APPEND MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)
#         else()

#             list(APPEND MKL_LIBRARIES mkl_intel_cilkrts mkl_sequential mkl_core)
#         endif()
        

#         list(APPEND MKL_LIBRARIES pthread m dl)
#     endif()

#     if(DEFINED ENV{MKLROOT})
#         link_directories($ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64)
#     endif()
# else()
#     message(FATAL_ERROR "MKL not found. Please install MKL or check your environment.")
# endif()

# add_executable(tsmm)
# target_sources(tsmm
#   PRIVATE
#     src/tsmm.cpp
# )
# add_executable(bench_blas)
# target_sources(bench_blas
#   PRIVATE
#     src/bench.cpp  
# )
add_executable(tsmm_opt)
target_sources(tsmm_opt
  PRIVATE
    src/tsmm_opt.cpp  
)

# target_include_directories(tsmm
#     PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
#         ${MKL_INCLUDE_DIRS}
# )

# target_include_directories(bench_blas
#     PUBLIC
#         ${MKL_INCLUDE_DIRS}
# )
target_include_directories(tsmm_opt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
# target_compile_options(tsmm 
#   PRIVATE
#     -O3
#     -g
#     -march=native
#     $<$<BOOL:${TSMM_ENABLE_AVX512}>:-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl >
# )
target_compile_options(tsmm_opt
  PRIVATE
    -O3
    -g
    -march=native
    -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl  # 明确指定需要的AVX512特性
    $<$<BOOL:${TSMM_ENABLE_AVX512}>:-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl>
)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(tsmm PUBLIC OpenMP::OpenMP_CXX)
#     target_link_libraries(bench_blas PUBLIC OpenMP::OpenMP_CXX)
# endif()

# target_link_libraries(bench_blas
#     PRIVATE
#     ${MKL_LIBRARIES}
# )


install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
