#ifndef NON_LINEAR_DIFFUSION_HPP
#define NON_LINEAR_DIFFUSION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/manifold_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

inline Point<3> transformPointToHeart(const Point<3> &u, double z_base, double d)
{
  const double xi = u[0];
  const double theta_hat = u[1];
  const double phi = u[2];

  const double theta_0 =
      std::acos(z_base / (d * std::cosh(xi)));

  const double theta =
      theta_0 + theta_hat * (numbers::PI - theta_0);

  return Point<3>(
      d * std::sinh(xi) * std::sin(theta) * std::cos(phi),
      d * std::sinh(xi) * std::sin(theta) * std::sin(phi),
      d * std::cosh(xi) * std::cos(theta));
}

// Class representing the non-linear diffusion problem.
class NonLinearDiffusion
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // TO MAKE THE HEART MODEL (from cube to fold into heart)
  class HeartManifold : public ChartManifold<3, 3>
  {
  public:
    HeartManifold(double _d, double _z_base) : d(_d), z_base(_z_base) {}

    virtual Point<3>
    push_forward(const Point<3> &u) const override
    {
      return transformPointToHeart(u, z_base, d);
    }

    virtual Point<dim>
    pull_back(const Point<dim> & /*x*/) const override
    {
      AssertThrow(false, ExcNotImplemented());
      return {};
    }

    virtual std::unique_ptr<Manifold<3, 3>>
    clone() const override
    {
      return std::make_unique<HeartManifold>(*this);
    }

  private:
    const double d;
    const double z_base;
  };

  // Constructor.
  NonLinearDiffusion(bool heartExpand_, const unsigned int &r_)
      : heartExpand(heartExpand_)
      , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      , pcout(std::cout, mpi_rank == 0)
      , r(r_)
      , mesh(MPI_COMM_WORLD)
  {
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
  
  const bool heartExpand;

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

  // for heart model to wrap periodically properly
  AffineConstraints<double> constraints;
};

#endif