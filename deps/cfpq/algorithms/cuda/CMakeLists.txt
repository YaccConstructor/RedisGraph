cmake_minimum_required(VERSION 3.10)
project(cuda)

find_package(CUDA REQUIRED)

add_subdirectory(m4ri)
add_subdirectory(sparse)
add_subdirectory(nsparse)
add_subdirectory(cusparse)
#add_subdirectory(graphblast)
