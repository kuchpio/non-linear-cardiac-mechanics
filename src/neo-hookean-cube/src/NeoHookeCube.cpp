#include "NeoHookeCube.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

void NeoHookeCube::setup() {
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // First we read the mesh from file into a serial (i.e. not parallel)
    // triangulation.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    // Then, we copy the triangulation into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    // Notice that we write here the number of *global* active cells (across all
    // processes).
    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    FE_SimplexP<dim> fe_scalar(r);
    fe = std::make_unique<FESystem<dim>>(fe_scalar, dim);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all process
    // retrieve the information they need for the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    // Then, we use the sparsity pattern to initialize the system matrix. Since
    // the sparsity pattern is partitioned by row, so will the matrix.
    pcout << "  Initializing the system matrix" << std::endl;
    jacobian_matrix.reinit(sparsity);

    // Finally, we initialize the right-hand side and solution vectors.
    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void NeoHookeCube::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe, *quadrature_face,
                                   update_values | update_quadrature_points |
                                       update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Defining the helper functions for AD
  using ADHelper = Differentiation::AD::ResidualLinearization<
      Differentiation::AD::NumberTypes::sacado_dfad, double>;
  using ADNumber = typename ADHelper::ad_type;

  // This class allows us to access vector-valued shape functions, so that we
  // don't have to worry about dealing with their components, but we can
  // directly use the vectorial form of the weak formulation.
  FEValuesExtractors::Vector displacement(0);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices); // we move this earlier, otherwise no
                                        // convergence. not fully sure why.

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    // change for AD:
    // 1) Initialize the AD helper for this specific cell
    const unsigned int n_independent_variables = dof_indices.size();
    const unsigned int n_dependent_variables = dof_indices.size();
    ADHelper ad_helper(n_independent_variables, n_dependent_variables);

    // 2) register the local DoF values in the helper
    ad_helper.register_dof_values(solution, dof_indices);

    // 3) Get sensitive values (?)
    const std::vector<ADNumber> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

    // 4) Build AD residual vector (must be explicitly zero-initialized)
    std::vector<ADNumber> residual_ad(n_dependent_variables, ADNumber(0.0));

    // We need to compute the Jacobian matrix and the residual for current
    // cell. This requires knowing the value and the gradient of u^{(k)}
    // (stored inside solution) on the quadrature nodes of the current cell.
    std::vector<Tensor<2, dim, ADNumber>> grad_d(n_q,
                                                 Tensor<2, dim, ADNumber>());
    fe_values[displacement].get_function_gradients_from_local_dof_values(
        dof_values_ad, grad_d);

    for (unsigned int q = 0; q < n_q; ++q) {
      const double shearModulus_loc =
          shearModulus(fe_values.quadrature_point(q));
      const double bulkModulus_loc = bulkModulus(fe_values.quadrature_point(q));

      Tensor<2, dim, ADNumber> F =
          unit_symmetric_tensor<dim, ADNumber>() + grad_d[q];
      ADNumber J = determinant(F);
      Tensor<2, dim, ADNumber> FTinv = invert(transpose(F));
      ADNumber I = F.norm_square();

      Tensor<2, dim, ADNumber> P =
          shearModulus_loc * std::pow(J, -2.0 / 3.0) * (F - I * FTinv / 3.0) +
          bulkModulus_loc * (J - 1.0) * J * FTinv;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        residual_ad[i] +=
            (scalar_product(P, fe_values[displacement].gradient(i, q))) *
            fe_values.JxW(q);
      }
    }

    // Boundary integral for Neumann BCs.
    if (cell->at_boundary()) {
      for (unsigned int f = 0; f < cell->n_faces(); ++f) {
        if (cell->face(f)->at_boundary()) {
          auto it = neumannBoundary.find(cell->face(f)->boundary_id());
          if (it == neumannBoundary.end())
            continue;
          auto h = it->second;

          fe_face_values.reinit(cell, f);

          for (unsigned int q = 0; q < n_q_face; ++q) {
            const Tensor<1, dim> h_loc = h(fe_face_values.quadrature_point(q));
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              residual_ad[i] -=
                  scalar_product(h_loc,
                                 fe_face_values[displacement].value(i, q)) *
                  fe_face_values.JxW(q);
            }
          }
        }
      }
    }

    // 1) Register residual and extract
    //   cell_rhs     = -R
    //   cell_matrix = dR/du
    ad_helper.register_residual_vector(residual_ad);
    ad_helper.compute_residual(cell_rhs);
    cell_rhs *= -1.0; // Newton RHS should be the negative residual
    ad_helper.compute_linearization(cell_matrix);

    jacobian_matrix.add(dof_indices, cell_matrix);
    residual_vector.add(dof_indices, cell_rhs);
  }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    VectorTools::interpolate_boundary_values(dof_handler, dirichletBoundary,
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, jacobian_matrix,
                                       delta_owned, residual_vector, true);
  }
}

void NeoHookeCube::solve_system() {
  SolverControl solver_control(10000, 1e-6 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

  // These settings are important for vector-valued problems (elasticity)
  amg_data.constant_modes.resize(1);
  amg_data.constant_modes[0].resize(dof_handler.n_dofs());

  // Set constant modes (rigid body modes) for better convergence
  // For 3D elasticity, there are 6 rigid body modes (3 translations + 3
  // rotations)
  ComponentMask components(dim, true);
  DoFTools::extract_constant_modes(dof_handler, components,
                                   amg_data.constant_modes);

  amg_data.elliptic = true;
  amg_data.higher_order_elements = (fe->degree > 1);
  amg_data.smoother_sweeps = 3;
  amg_data.aggregation_threshold = 0.02;

  preconditioner.initialize(jacobian_matrix, amg_data);

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;
}

void NeoHookeCube::solve_newton() {
  pcout << "===============================================" << std::endl;

  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-7;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    pcout << "Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance) {
      solve_system();

      solution_owned += delta_owned;
      solution = solution_owned;
    } else {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }

  pcout << "===============================================" << std::endl;
}

void NeoHookeCube::output() const {
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  // By passing these two additional arguments to add_data_vector, we specify
  // that the three components of the solution are actually the three components
  // of a vector, so that the visualization program can take that into account.
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  std::vector<std::string> solution_names(dim, "d");

  data_out.add_data_vector(dof_handler, solution, solution_names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", outputFilename, 0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << outputFilename << "." << std::endl;

  pcout << "===============================================" << std::endl;
}
