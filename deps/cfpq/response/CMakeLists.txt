cmake_minimum_required(VERSION 3.10)
project(response C)


add_library(${PROJECT_NAME} src/response.c)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} grammar)
