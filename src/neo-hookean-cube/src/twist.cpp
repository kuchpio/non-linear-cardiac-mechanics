#include "NeoHookeCube.hpp"

// Main function.
int main(int argc, char *argv[]) {
  constexpr unsigned int dim = NeoHookeCube::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string mesh_file_name = "mesh-cube-10.msh";
  const unsigned int degree = 1;
  const auto shearModulus = [](const Point<dim> &) { return 1e4; };
  const auto bulkModulus = [](const Point<dim> &) { return 1e5; };
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  Functions::ZeroFunction<dim> zero_function;
  dirichletBoundary.emplace(2, &zero_function);
  std::map<types::boundary_id,
           const std::function<Tensor<1, dim>(const Point<dim> &)>>
      neumannBoundary;
  neumannBoundary.emplace(0, [](const Point<dim> &p) {
    return Tensor<1, dim>({0.0, 0.0, p[1] > 0.5 ? 1.5e3 : 0.0});
  });
  neumannBoundary.emplace(1, [](const Point<dim> &p) {
    return Tensor<1, dim>({0.0, 0.0, p[1] > 0.5 ? -1.5e3 : 0.0});
  });

  NeoHookeCube problem(mesh_file_name, degree, shearModulus, bulkModulus,
                       dirichletBoundary, neumannBoundary,
                       "out-twist");

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
