cmake_minimum_required(VERSION 3.7.2)
project(cpu-graphblas C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

add_library(${PROJECT_NAME} src/cpu_graphblas.c)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} grammar response ${GRAPH_BLAS_LIBRARY} m)
