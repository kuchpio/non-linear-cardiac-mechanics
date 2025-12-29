cmake_minimum_required(VERSION 3.12.0)
project(non-linear-cardiac-mechanics LANGUAGES CXX C)

add_subdirectory(src/neo-hookean-cube/)
add_subdirectory(src/cubeAndBeam/)
add_subdirectory(src/heart/)
