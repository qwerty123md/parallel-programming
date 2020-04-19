cmake_minimum_required(VERSION 3.6)

project(lab_1)

set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

set(SRC src/main.c src/cgm.c )
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -lrt -lm -fopenmp")

add_executable(${PROJECT_NAME} ${SRC})

include_directories(/usr/include/mpi/)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")

include(common_lab)