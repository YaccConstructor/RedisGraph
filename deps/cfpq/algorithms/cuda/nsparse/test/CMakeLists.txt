cmake_minimum_required(VERSION 3.10)
project(nsparse_test CXX CUDA)

add_executable(${PROJECT_NAME} src/nsparse_test.cu src/utils.cpp)

target_link_libraries(${PROJECT_NAME} gtest_main nsparse)

add_test(
        NAME ${PROJECT_NAME}
        COMMAND ${PROJECT_BINARY_DIR}/${PROJECT_NAME}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)
