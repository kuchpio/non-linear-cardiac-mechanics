#ifndef NON_LINEAR_ELASTICITY_SOLVER_HPP
#define NON_LINEAR_ELASTICITY_SOLVER_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/differentiation/ad.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <variant>

#include "ConstitutiveLaw.hpp"
#include "NeoHookeConstitutiveLaw.hpp"
#include "GuccioneConstitutiveLaw.hpp"
#include "MeshProvider.hpp"
#include "MaterialParameters.hpp"

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NonLinearElasticitySolver
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  using GlobalTractionFun = const std::function<Tensor<1, dim>(const Point<dim> &)>;
  using NormalTractionFun = const std::function<double(const Point<dim> &)>;
  using TractionFun = std::variant<GlobalTractionFun, NormalTractionFun>;

  // Constructor.
  NonLinearElasticitySolver(const MeshProvider<dim>& meshProvider_, const unsigned int &r_, 
          MaterialParameters<dim>& materialParameters_, bool isGuccione,
          const std::map<types::boundary_id, const Function<dim> *> &dirichletBoundary_,
          const std::map<types::boundary_id, TractionFun> &neumannBoundary_,
          const std::string& outputFilename_
    )
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
    , materialParameters(materialParameters_)
    , dirichletBoundary(dirichletBoundary_)
    , neumannBoundary(neumannBoundary_)
    , outputFilename(outputFilename_)
    , meshProvider(meshProvider_)
  {
      if (isGuccione) {
          constitutiveLaw = std::make_unique<GuccioneConstitutiveLaw<dim, ADNumber>>();
      } else {
          constitutiveLaw = std::make_unique<NeoHookeConstitutiveLaw<dim, ADNumber>>();
      }
  }

  // Initialization.
  void
  setup();

  // Solve the problem using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output() const;

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the tangent problem.
  void
  solve_system();

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Discretization. ///////////////////////////////////////////////////////////

  // Polynomial degree.
  const unsigned int r;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  
  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  AffineConstraints<double> zeroConstraintsOnDirichletBoundary;
  AffineConstraints<double> dirichletConstraints;

  MaterialParameters<dim>& materialParameters;
  const std::map<types::boundary_id, const Function<dim> *> dirichletBoundary;
  const std::map<types::boundary_id, TractionFun> neumannBoundary;
  const std::string outputFilename;

  // Defining the helper functions for AD
  using ADHelper = Differentiation::AD::ResidualLinearization<
      Differentiation::AD::NumberTypes::sacado_dfad, double>;
  using ADNumber = typename ADHelper::ad_type;

  std::unique_ptr<ConstitutiveLaw<dim, ADNumber>> constitutiveLaw;
  const MeshProvider<dim>& meshProvider;
};

#endif
