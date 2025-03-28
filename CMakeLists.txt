cmake_minimum_required(VERSION 3.10)
project(tensor_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "-O3 -march=native -pg")

include_directories(./include)
add_executable(tsmm tsmm.cpp)