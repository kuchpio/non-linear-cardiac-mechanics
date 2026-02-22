add_library(nle STATIC 
    include/NonLinearElasticitySolver.hpp src/NonLinearElasticitySolver.cpp
    include/MaterialParameters.hpp
    include/ConstitutiveLaw.hpp
    include/NeoHookeConstitutiveLaw.hpp
    include/GuccioneConstitutiveLaw.hpp
    include/MeshProvider.hpp
    include/FileMeshProvider.hpp
)
target_include_directories(nle PUBLIC "include")
deal_ii_setup_target(nle)
