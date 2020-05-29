cmake_minimum_required(VERSION 3.5.1)

project(lab_5)

set(SRC src/main.c src/tasks_manager.c src/tasks_manager.h)
set(CMAKE_C_STANDARD 99)

add_executable(${PROJECT_NAME} ${SRC})
include_directories(/usr/include/mpi/)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")

include(common_lab)
