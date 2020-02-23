cmake_minimum_required(VERSION 3.5.1)

project(lab_1)

set(SRC src/main.c src/cgm.c )
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp -pg -lrt -lm")

add_executable(${PROJECT_NAME} ${SRC})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")

include(common_lab)

