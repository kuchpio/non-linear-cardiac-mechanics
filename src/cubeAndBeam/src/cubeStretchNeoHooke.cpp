#include "NonLinearDiffusion.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const bool isCube = true;
  const bool isGuccione = false;
  const unsigned int degree         = 1;
  int neumannBoundaryTypeId = 0;
  std::vector<int> dirichletBoundaries = {0};

  NonLinearDiffusion problem(isCube, isGuccione, degree, dirichletBoundaries, neumannBoundaryTypeId);

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}