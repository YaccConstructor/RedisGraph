cmake_minimum_required(VERSION 3.10)
project(grammar C)

# libgrammar.a
add_library(${PROJECT_NAME} src/grammar.c src/helpers.c src/item_mapper.c)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# grammar-example
add_executable(${PROJECT_NAME}-example src/load_and_stdout.c)
target_link_libraries(${PROJECT_NAME}-example ${PROJECT_NAME})

# resources for grammar-example
file(COPY ${PROJECT_SOURCE_DIR}/resources/toy_cfg.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
