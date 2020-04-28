cmake_minimum_required(VERSION 3.10)
project(cfpq_test CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} src/cfpq_test.cpp src/triplet_loader.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} gtest_main grammar cpu-graphblas sparse m4ri nsparse_cfpq cusparse_cfpq)

add_test(
        NAME ${PROJECT_NAME}
        COMMAND ${PROJECT_BINARY_DIR}/${PROJECT_NAME}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

file(COPY resources DESTINATION ${PROJECT_BINARY_DIR})
