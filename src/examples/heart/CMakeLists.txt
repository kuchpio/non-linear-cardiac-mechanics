cmake_minimum_required(VERSION 3.12.0)

add_executable(neo-hookean-heart-expand neo-hookean-heart.cpp)
target_link_libraries(neo-hookean-heart-expand nle)
target_compile_definitions(neo-hookean-heart-expand PUBLIC -DEXPAND)
deal_ii_setup_target(neo-hookean-heart-expand)

add_executable(neo-hookean-heart-contract neo-hookean-heart.cpp)
target_link_libraries(neo-hookean-heart-contract nle)
deal_ii_setup_target(neo-hookean-heart-contract)

add_executable(guccione-heart-expand guccione-heart.cpp)
target_link_libraries(guccione-heart-expand nle)
target_compile_definitions(guccione-heart-expand PUBLIC -DEXPAND)
deal_ii_setup_target(guccione-heart-expand)

add_executable(guccione-heart-contract guccione-heart.cpp)
target_link_libraries(guccione-heart-contract nle)
deal_ii_setup_target(guccione-heart-contract)
