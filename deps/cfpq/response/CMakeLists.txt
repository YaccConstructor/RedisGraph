cmake_minimum_required(VERSION 3.7.2)
project(response C)


add_library(${PROJECT_NAME} src/response.c)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} grammar)
