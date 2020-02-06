cmake_minimum_required(VERSION 3.10)
project(graphblast CXX CUDA)

# check GraphBLAST header
if (NOT DEFINED GRAPH_BLAST_HEADER)
    message(FATAL_ERROR "GRAPH_BLAST_HEADER not found")
endif ()

message(STATUS "Using ${GRAPH_BLAST_HEADER} as GraphBLAST include path")

# check moderngpu header
if (NOT DEFINED MODERN_GPU_HEADER)
    message(FATAL_ERROR "MODERN_GPU_HEADER not found")
endif ()

message(STATUS "Using ${MODERN_GPU_HEADER} as moderngpu include path")

# check cub header
if (NOT DEFINED CUB_HEADER)
    message(FATAL_ERROR "CUB_HEADER not found")
endif ()

message(STATUS "Using ${CUB_HEADER} as cub include path")

# check moderngpu sources
if (NOT DEFINED MODERN_GPU_SOURCE)
    message(FATAL_ERROR "MODERN_GPU_SOURCE not found")
endif ()

message(STATUS "Using ${MODERN_GPU_SOURCE} as moderngpu source path")

# find BOOST
find_package(Boost COMPONENTS program_options REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")

cuda_add_library(${PROJECT_NAME}
        src/graphblast.cpp
        src/graphblast_impl.cu
        ${MODERN_GPU_SOURCE}/mgpucontext.cu
        ${MODERN_GPU_SOURCE}/mgpuutil.cpp
        )

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${GRAPH_BLAST_HEADER}
        ${MODERN_GPU_HEADER}
        ${CUB_HEADER}
        ${CUDA_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME} grammar response ${GRAPH_BLAS_LIBRARY} cudart cusparse ${Boost_LIBRARIES})
