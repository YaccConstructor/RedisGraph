cmake_minimum_required(VERSION 3.10)
project(sparse CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_library(${PROJECT_NAME}
        src/sparse.cpp
        src/sparse_impl.cu
        )

add_executable(example
        src/dummy.cpp
        src/example.cu)

target_include_directories(example PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(example grammar response ${GRAPH_BLAS_LIBRARY} cudart)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} grammar response ${GRAPH_BLAS_LIBRARY} cudart)
