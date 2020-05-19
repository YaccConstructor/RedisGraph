cmake_minimum_required(VERSION 3.10)
project(cusparse CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -Xptxas -O3 -Xptxas -v")


add_library(cusparse_cfpq src/cusparse_cfpq.cpp src/cusparse.cu)
target_include_directories(cusparse_cfpq PUBLIC include/)
target_include_directories(cusparse_cfpq PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cusparse_cfpq generic cudart cusparse)
