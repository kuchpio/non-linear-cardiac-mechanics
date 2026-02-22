cmake_minimum_required(VERSION 3.12.0)

add_executable(squeeze squeeze.cpp)
target_link_libraries(squeeze nle)
deal_ii_setup_target(squeeze)

add_executable(twist twist.cpp)
target_link_libraries(twist nle)
deal_ii_setup_target(twist)

add_executable(press press.cpp)
target_link_libraries(press nle)
deal_ii_setup_target(press)

add_executable(cylinder cylinder.cpp)
target_link_libraries(cylinder nle)
deal_ii_setup_target(cylinder)

add_executable(hexahedralPulled hexahedralPulled.cpp)
target_link_libraries(hexahedralPulled nle)
deal_ii_setup_target(hexahedralPulled)

file(
    DOWNLOAD 
    https://raw.githubusercontent.com/michelebucelli/nmpde-labs-aa-25-26/refs/heads/main/lab-05/mesh/mesh-cube-10.msh
    ${CMAKE_CURRENT_BINARY_DIR}/mesh-cube-10.msh
)
