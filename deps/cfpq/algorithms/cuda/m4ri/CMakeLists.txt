cmake_minimum_required(VERSION 3.7.2)
project(m4ri CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_library(${PROJECT_NAME}
        src/m4ri.cpp
        src/methodOf4RusBooleanSemiringGpu.cu
        src/methodOf4RusBooleanSemiringMatrix.cu
        src/gpuMemoryManagement.cu
        )

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} grammar response ${GRAPH_BLAS_LIBRARY} cudart)
