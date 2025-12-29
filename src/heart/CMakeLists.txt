cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(heartNeoHookeContract src/heartNeoHookeContract.cpp src/NonLinearDiffusion.cpp)
deal_ii_setup_target(heartNeoHookeContract)

add_executable(heartNeoHookeExpand src/heartNeoHookeExpand.cpp src/NonLinearDiffusion.cpp)
deal_ii_setup_target(heartNeoHookeExpand)
