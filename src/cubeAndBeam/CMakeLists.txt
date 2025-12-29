cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(cubeStretchNeoHooke src/cubeStretchNeoHooke.cpp src/NonLinearDiffusion.cpp)
deal_ii_setup_target(cubeStretchNeoHooke)

add_executable(beamBendNeoHooke src/beamBendNeoHooke.cpp src/NonLinearDiffusion.cpp)
deal_ii_setup_target(beamBendNeoHooke)

add_executable(beamBendGuccione src/beamBendGuccione.cpp src/NonLinearDiffusion.cpp)
deal_ii_setup_target(beamBendGuccione)
