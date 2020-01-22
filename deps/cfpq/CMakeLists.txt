cmake_minimum_required(VERSION 3.10)
project(cfpq)

# find GraphBLAS include path if not specified
if (NOT DEFINED GRAPH_BLAS_HEADER)
    find_path(GRAPH_BLAS_HEADER GraphBLAS.h)
    if (${GRAPH_BLAS_HEADER} STREQUAL "GRAPH_BLAS_HEADER-NOTFOUND")
        message(FATAL_ERROR "GRAPH_BLAS_HEADER not found")
    endif ()
endif ()

message(STATUS "Using ${GRAPH_BLAS_HEADER} as GraphBLAS include path")

# find GraphBLAS library path if not specified
if (NOT DEFINED GRAPH_BLAS_LIBRARY)
    find_library(GRAPH_BLAS_LIBRARY graphblas)
    if (${GRAPH_BLAS_LIBRARY} STREQUAL "GRAPH_BLAS_LIBRARY-NOTFOUND")
        message(FATAL_ERROR "GRAPH_BLAS_LIBRARY not found")
    endif ()
endif ()

message(STATUS "Using ${GRAPH_BLAS_LIBRARY} as GraphBLAS library path")

include_directories(${GRAPH_BLAS_HEADER})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(grammar)
add_subdirectory(response)
add_subdirectory(algorithms)
