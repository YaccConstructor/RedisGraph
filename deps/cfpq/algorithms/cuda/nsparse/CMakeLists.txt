cmake_minimum_required(VERSION 3.10)
project(nsparse CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --expt-relaxed-constexpr --expt-extended-lambda -g")

add_library(nsparse INTERFACE)
target_include_directories(nsparse INTERFACE include/)

add_subdirectory(test)
