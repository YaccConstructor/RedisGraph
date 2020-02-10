cmake_minimum_required(VERSION 3.10)
project(nsparse CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -O3")


add_library(nsparse_cfpq src/nsparse.cpp src/nsparse.cu)
target_include_directories(nsparse_cfpq PUBLIC include/)
target_link_libraries(nsparse_cfpq grammar response ${GRAPH_BLAS_LIBRARY} cudart)

add_library(nsparse INTERFACE)
target_include_directories(nsparse INTERFACE include/)
add_subdirectory(test)
