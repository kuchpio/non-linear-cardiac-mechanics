#include "HeartMaterialParameters.hpp"
#include "NonLinearElasticitySolver.hpp"
#include "FileMeshProvider.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = NonLinearElasticitySolver::dim;

#ifdef EXPAND
  constexpr bool expand = true;
#else
  constexpr bool expand = false;
#endif

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  FileMeshProvider<dim, FE_Q, QGauss> meshProvider("heart_mesh.msh");
  const unsigned int degree = 1;
  HeartMaterialParameters materialParameters("material_coordinates.txt");
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  Functions::ZeroFunction<dim> zero_function;
  dirichletBoundary.emplace(3, &zero_function);
  std::map<types::boundary_id, NonLinearElasticitySolver::TractionFun> neumannBoundary;
  neumannBoundary.emplace(expand ? 1 : 2, [](const Point<dim> &) {
    if constexpr (expand) {
        return -6.0;
    } else {
        return -2.0;
    }
  });

  NonLinearElasticitySolver problem(meshProvider, degree, materialParameters, true,
                       dirichletBoundary, neumannBoundary,
                       expand ? "out-guccione-expand" : "out-guccione-contract");

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
