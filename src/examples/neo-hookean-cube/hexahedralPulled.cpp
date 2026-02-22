#include "NonLinearElasticitySolver.hpp"
#include "HyperCubeMeshProvider.hpp"
#include "NeoHookeanMaterialParameters.hpp"

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = NonLinearElasticitySolver::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  HyperCubeMeshProvider<dim> meshProvider(0, 1);
  const unsigned int degree = 1;
  const auto shearModulus = [](const Point<dim> &) { return 1.0; };
  const auto bulkModulus = [](const Point<dim> &) { return 10.0; };
  NeoHookeanMaterialParameters<dim> materialParameters(shearModulus, bulkModulus);
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  Functions::ZeroFunction<dim> zero_function;
  dirichletBoundary.emplace(0, &zero_function);
  std::map<types::boundary_id, NonLinearElasticitySolver::TractionFun> neumannBoundary;
  neumannBoundary.emplace(1, [](const Point<dim> &) {
    return Tensor<1, dim>({0.5, 0.0, 0.0});
  });

  NonLinearElasticitySolver problem(meshProvider, degree, materialParameters, false,
                       dirichletBoundary, neumannBoundary,
                       "out-hexahedral-pulled");

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
