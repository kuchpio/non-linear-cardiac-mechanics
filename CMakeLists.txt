cmake_minimum_required(VERSION 3.15)
project(non-linear-cardiac-mechanics LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(src/common/cmake-common.cmake)

add_subdirectory(src/non-linear-elasticity/)
add_subdirectory(src/examples/)

if(EXISTS ${CMAKE_SOURCE_DIR}/heart-mesh/heart_mesh.msh AND EXISTS ${CMAKE_SOURCE_DIR}/heart-mesh/material_coordinates.txt)
    file(COPY heart-mesh/heart_mesh.msh DESTINATION ${CMAKE_BINARY_DIR}/src/examples/heart)
    file(COPY heart-mesh/material_coordinates.txt DESTINATION ${CMAKE_BINARY_DIR}/src/examples/heart)
else()
    message(WARNING "One of files heart-mesh/heart_mesh.msh or heart-mesh/material_coordinates.txt does not exist. 
To generate it, enter heart-mesh/ directory, run the python script heart-gen.py, and then re-run cmake configuration.")
endif()
