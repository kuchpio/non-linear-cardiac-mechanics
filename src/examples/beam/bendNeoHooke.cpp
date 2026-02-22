#include "NonLinearElasticitySolver.hpp"
#include "BeamMeshProvider.hpp"
#include "BeamMaterialParameters.hpp"
#include <mpi.h>

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = NonLinearElasticitySolver::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const auto comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  BeamMeshProvider<dim> meshProvider;
  const unsigned int degree = 1;
  BeamMaterialParameters<dim> materialParameters;
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  Functions::ZeroFunction<dim> zero_function;
  dirichletBoundary.emplace(0, &zero_function);
  std::map<types::boundary_id, NonLinearElasticitySolver::TractionFun> neumannBoundary;
  neumannBoundary.emplace(4, [](const Point<dim> &) {
    return -0.004;
  });

  NonLinearElasticitySolver problem(meshProvider, degree, materialParameters, false,
                       dirichletBoundary, neumannBoundary,
                       "out-bend-neo-hooke");

  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  problem.setup();
  MPI_Barrier(comm);
  double t_setup_local = MPI_Wtime() - t0;
  double t_setup = 0.0;
  MPI_Reduce(&t_setup_local, &t_setup, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  
  MPI_Barrier(comm);
  t0 = MPI_Wtime();
  problem.solve_newton();
  MPI_Barrier(comm);
  double t_solve_local = MPI_Wtime() - t0;
  double t_solve = 0.0;
  MPI_Reduce(&t_solve_local, &t_solve, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  
  if (rank == 0) {
    std::cout << "Setup time (max rank):  " << t_setup << " s\n";
    std::cout << "Solve time (max rank):  " << t_solve << " s\n";
    std::cout << "Total (setup+solve):    " << (t_setup + t_solve) << " s\n";
  }
  problem.output();

  return 0;
}
