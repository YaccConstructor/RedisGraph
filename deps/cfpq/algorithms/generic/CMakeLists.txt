cmake_minimum_required(VERSION 3.10)
project(generic CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(generic INTERFACE)
target_link_libraries(generic INTERFACE grammar response ${GRAPH_BLAS_LIBRARY})
target_include_directories(generic INTERFACE include/)
