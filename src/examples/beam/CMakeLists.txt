cmake_minimum_required(VERSION 3.12.0)

add_executable(bendNeoHooke bendNeoHooke.cpp)
target_link_libraries(bendNeoHooke nle)
deal_ii_setup_target(bendNeoHooke)

add_executable(bendGuccione bendGuccione.cpp)
target_link_libraries(bendGuccione nle)
deal_ii_setup_target(bendGuccione)
