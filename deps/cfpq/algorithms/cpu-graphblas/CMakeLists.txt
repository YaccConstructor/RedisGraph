cmake_minimum_required(VERSION 3.10)
project(cpu-graphblas CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_library(${PROJECT_NAME} src/cpu_graphblas.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} generic)
