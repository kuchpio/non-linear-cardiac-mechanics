#include "FileMeshProvider.hpp"
#include "NeoHookeanMaterialParameters.hpp"
#include "NonLinearElasticitySolver.hpp"

constexpr unsigned int dim = NonLinearElasticitySolver::dim;

class CylinderDisplacement : public Function<dim> {
public:
  CylinderDisplacement(double _alphaMax, double _radius, double _height)
      : alphaMax(_alphaMax), radius(_radius), height(_height) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override {
    double alpha = alphaMax * (2 * p[0] - 1);
    values[0] = 0.5 + radius * std::sin(alpha) - p[0];
    values[1] = height - radius * std::cos(alpha) - p[1];
    values[2] = 0.0;
  }

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    Vector<double> v;
    vector_value(p, v);
    return v[component];
  }

private:
  double alphaMax;
  double radius;
  double height;
};

// Main function.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  FileMeshProvider<dim, FE_SimplexP, QGaussSimplex> meshProvider("mesh-cube-10.msh");
  const unsigned int degree = 1;
  const auto shearModulus = [](const Point<dim> &) { return 1e4; };
  const auto bulkModulus = [](const Point<dim> &) { return 1e4; };
  NeoHookeanMaterialParameters<dim> materialParameters(shearModulus,
                                                       bulkModulus);
  std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  CylinderDisplacement bottomDisplacement(M_PI / 12.0, 2.1, 2);
  dirichletBoundary.emplace(2, &bottomDisplacement);
  CylinderDisplacement topDisplacement(M_PI / 6.0, 0.9, 2);
  dirichletBoundary.emplace(3, &topDisplacement);
  std::map<types::boundary_id, NonLinearElasticitySolver::TractionFun> neumannBoundary;

  NonLinearElasticitySolver problem(meshProvider, degree, materialParameters, false,
                                    dirichletBoundary, neumannBoundary,
                                    "out-cylinder");

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
