cmake_minimum_required(VERSION 3.10)
project(cuda)

find_package(CUDA REQUIRED)

add_subdirectory(sparse)
add_subdirectory(nsparse)
add_subdirectory(cusparse)
