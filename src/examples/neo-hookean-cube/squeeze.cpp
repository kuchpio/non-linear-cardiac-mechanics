#include "NonLinearElasticitySolver.hpp"
#include "FileMeshProvider.hpp"
#include "NeoHookeanMaterialParameters.hpp"

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = NonLinearElasticitySolver::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  FileMeshProvider<dim, FE_SimplexP, QGaussSimplex> meshProvider("mesh-cube-10.msh");
  const unsigned int degree = 1;
  const auto shearModulus = [](const Point<dim> &p) { return p[0] < 0.5 ? 1e3 : 5e3; };
  const auto bulkModulus = [](const Point<dim> &) { return 1e4; };
  NeoHookeanMaterialParameters<dim> materialParameters(shearModulus, bulkModulus);
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  Functions::ZeroFunction<dim> zero_function;
  dirichletBoundary.emplace(2, &zero_function);
  std::map<types::boundary_id, NonLinearElasticitySolver::TractionFun> neumannBoundary;
  neumannBoundary.emplace(4, [](const Point<dim> &p) { return Tensor<1, dim>({0.0, 0.0, p[1] > 0.5 ? 8e2 : 0.0}); });
  neumannBoundary.emplace(5, [](const Point<dim> &p) { return Tensor<1, dim>({0.0, 0.0, p[1] > 0.5 ? -8e2 : 0.0}); });

  NonLinearElasticitySolver problem(meshProvider, degree, materialParameters, false,
                             dirichletBoundary, neumannBoundary, "out-squeeze");

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
