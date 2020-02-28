cmake_minimum_required(VERSION 3.5.1)

project(lab_2)

set(SRC src/main.c)
set(CMAKE_C_STANDARD 99)

add_executable(${PROJECT_NAME} ${SRC})


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")

include(common_lab)

