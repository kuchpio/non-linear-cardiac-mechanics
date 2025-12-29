#include "NonLinearDiffusion.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  bool heartExpand = false;
  const unsigned int degree = 1;

  NonLinearDiffusion problem(heartExpand, degree);

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}