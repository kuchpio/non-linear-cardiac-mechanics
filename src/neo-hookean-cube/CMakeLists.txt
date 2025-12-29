cmake_minimum_required(VERSION 3.12.0)
project(neoHookeCube LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(squeeze src/squeeze.cpp src/NeoHookeCube.cpp)
deal_ii_setup_target(squeeze)

add_executable(twist src/twist.cpp src/NeoHookeCube.cpp)
deal_ii_setup_target(twist)

add_executable(press src/press.cpp src/NeoHookeCube.cpp)
deal_ii_setup_target(press)

file(
    DOWNLOAD 
    https://raw.githubusercontent.com/michelebucelli/nmpde-labs-aa-25-26/refs/heads/main/lab-05/mesh/mesh-cube-10.msh
    ${CMAKE_CURRENT_BINARY_DIR}/mesh-cube-10.msh
)
