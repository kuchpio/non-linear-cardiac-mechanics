#include "NonLinearDiffusion.hpp"

#include <deal.II/base/function.h>
#include <deal.II/differentiation/ad.h>

#include <deal.II/numerics/data_out_faces.h>

#include <map>

void NonLinearDiffusion::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    // CUBE
    if (isCube)
    {
      GridGenerator::hyper_cube(mesh_serial, 0, 1, true);
      mesh_serial.refine_global(2);
    }
    else 
    // RECTANGLE -- BEAM
    {
      Point<dim> corner1(0., 0., 0.);
      Point<dim> corner2(10., 1., 1.);

      std::vector<unsigned int> subdivisions(dim);
      subdivisions[0] = 10;
      subdivisions[1] = 1;
      subdivisions[2] = 1;

      GridGenerator::subdivided_hyper_rectangle(mesh_serial, subdivisions, corner1, corner2, true);

      mesh_serial.refine_global(1);
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

    FE_Q<dim> fe_scalar(r);
    fe = std::make_unique<FESystem<dim>>(fe_scalar, dim);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGauss<dim - 1>>(fe->degree + 1);

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

void NonLinearDiffusion::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe, *quadrature_face, update_values | update_normal_vectors | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Defining the helper functions for AD
  using ADHelper =
      Differentiation::AD::ResidualLinearization<
          Differentiation::AD::NumberTypes::sacado_dfad,
          double>;
  using ADNumber = typename ADHelper::ad_type;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices); 

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    // change for AD:
    // 1) Initialize the AD helper for this specific cell
    const unsigned int n_independent_variables = dof_indices.size();
    const unsigned int n_dependent_variables = dofs_per_cell;
    ADHelper ad_helper(n_independent_variables, n_dependent_variables);

    // 2) register the local DoF values in the helper
    ad_helper.register_dof_values(solution, dof_indices);

    // 3) Get sensitive values (?)
    const std::vector<ADNumber> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

    // 4) Build AD residual vector (must be explicitly zero-initialized)
    std::vector<ADNumber> residual_ad(n_dependent_variables, ADNumber(0.0));

    // 5) Remove next code for AD, we dont need these anymore

    // We need to compute the Jacobian matrix and the residual for current
    // cell. This requires knowing the value and the gradient of u^{(k)}
    // (stored inside solution) on the quadrature nodes of the current
    // cell. This can be accomplished through
    // FEValues::get_function_values and FEValues::get_function_gradients.
    // fe_values.get_function_values(solution, solution_loc);
    // fe_values.get_function_gradients(solution, solution_gradient_loc);

    // This class allows us to access vector-valued shape functions, so that we
    // don't have to worry about dealing with their components, but we can
    // directly use the vectorial form of the weak formulation.
    FEValuesExtractors::Vector displacement(0);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      Tensor<2, dim, ADNumber> P;

      // GUCCIONE
      if (isGuccione)
      {

        // Guccione material parameters
        const double b_ff = 8;   // fiber direction
        const double b_ss = 2;   // sheet direction
        const double b_nn = 2;   // normal direction
        const double b_fs = 4;   // fiber-sheet coupling
        const double b_fn = 4;   // fiber-normal coupling
        const double b_sn = 2;   // sheet-normal coupling
        const double C_param = 2; // scaling parameter
        // const double B_bulk = 10.0; // bulk modulus for volumetric penalty

        // u(x_q) and grad u(x_q) as AD quantities
        Tensor<1, dim, ADNumber> u_q;
        for (unsigned int d = 0; d < dim; ++d)
          u_q[d] = ADNumber(0.0);
        Tensor<2, dim, ADNumber> grad_u_q;
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            grad_u_q[d1][d2] = ADNumber(0.0);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          u_q += dof_values_ad[j] * fe_values[displacement].value(j, q);
          grad_u_q += dof_values_ad[j] * fe_values[displacement].gradient(j, q);
        }

        // Deformation gradient and related tensors
        Tensor<2, dim, ADNumber> F = unit_symmetric_tensor<dim, ADNumber>() + grad_u_q;
        ADNumber J = determinant(F);
        Tensor<2, dim, ADNumber> C = transpose(F) * F; // Right Cauchy-Green tensor

        // Green-Lagrange strain tensor: E = 0.5 * (C - I)
        Tensor<2, dim, ADNumber> E = 0.5 * (C - unit_symmetric_tensor<dim, ADNumber>());

        // Define fiber directions (these should ideally come from your mesh data)
        // For now, using simple orthogonal directions
        Tensor<1, dim, ADNumber> f_fiber{{1.0, 0.0, 0.0}};  // fiber direction
        Tensor<1, dim, ADNumber> s_sheet{{0.0, 0.1, 0.0}};  // sheet direction
        Tensor<1, dim, ADNumber> n_normal{{0.0, 0.0, 1.0}}; // normal direction

        // Compute strain components in fiber coordinate system
        ADNumber E_ff = scalar_product(f_fiber, E * f_fiber);
        ADNumber E_ss = scalar_product(s_sheet, E * s_sheet);
        ADNumber E_nn = scalar_product(n_normal, E * n_normal);
        ADNumber E_fs = scalar_product(f_fiber, E * s_sheet);
        ADNumber E_fn = scalar_product(f_fiber, E * n_normal);
        ADNumber E_sn = scalar_product(s_sheet, E * n_normal);

        ADNumber E_sf = scalar_product(s_sheet, E * f_fiber);
        ADNumber E_nf = scalar_product(n_normal, E * f_fiber);
        ADNumber E_ns = scalar_product(n_normal, E * s_sheet);

        // Compute Q (exponent)
        ADNumber Q = b_ff * E_ff * E_ff +
                    b_ss * E_ss * E_ss +
                    b_nn * E_nn * E_nn +
                    b_fs * (E_fs * E_fs + E_sf * E_sf) + 
                    b_fn * (E_fn * E_fn + E_nf * E_nf) + 
                    b_sn * (E_sn * E_sn + E_ns * E_ns);


        // Guccione strain energy: W = C/2 * (exp(Q) - 1)
        // Add volumetric penalty: W_vol = B/2 * (J - 1) * log(J)
        ADNumber W_iso = C_param / 2.0 * (exp(Q) - 1.0);
        // ADNumber W_vol = B_bulk / 2.0 * (J - 1.0) * log(J);

        // Second Piola-Kirchhoff stress: S = dW/dE
        // For Guccione: S_iso = C * exp(Q) * dQ/dE
        ADNumber exp_Q = exp(Q);

        // Derivatives of Q with respect to E components
        Tensor<2, dim, ADNumber> dQ_dE;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            dQ_dE[i][j] = ADNumber(0.0);

        // Build dQ/dE tensor
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
          {
            dQ_dE[i][j] += 2.0 * b_ff * E_ff * (f_fiber[i] * f_fiber[j]);
            dQ_dE[i][j] += 2.0 * b_ss * E_ss * (s_sheet[i] * s_sheet[j]);
            dQ_dE[i][j] += 2.0 * b_nn * E_nn * (n_normal[i] * n_normal[j]);
            dQ_dE[i][j] += 4.0 * b_fs * E_fs * (f_fiber[i] * s_sheet[j] + s_sheet[i] * f_fiber[j]) / 2.0;
            dQ_dE[i][j] += 4.0 * b_fn * E_fn * (f_fiber[i] * n_normal[j] + n_normal[i] * f_fiber[j]) / 2.0;
            dQ_dE[i][j] += 4.0 * b_sn * E_sn * (s_sheet[i] * n_normal[j] + n_normal[i] * s_sheet[j]) / 2.0;
          }

        // Second Piola-Kirchhoff stress
        Tensor<2, dim, ADNumber> S_iso = C_param * exp_Q * dQ_dE;

        // Volumetric part: S_vol = B * [(J-1) + log(J)] * J * C^{-1}
        // Tensor<2, dim, ADNumber> C_inv = invert(C);
        // Tensor<2, dim, ADNumber> S_vol = B_bulk * ((J - 1.0) + log(J)) * J * C_inv;

        // Total second Piola-Kirchhoff stress // TODO do we need volumetric term???
        // Tensor<2, dim, ADNumber> S = S_iso + S_vol;
        Tensor<2, dim, ADNumber> S = S_iso;

        // First Piola-Kirchhoff stress: P = F * S
        P = F * S;
      }
      // NEO-HOOKE
      else 
      {
        double mu_1 = 1.0;
        const double k_1 = 10.0;

        if (neumannBoundaryTypeId == 1) mu_1 = 10.0;

        // u(x_q) and grad u(x_q) as AD quantities
        Tensor<1, dim, ADNumber> u_q;
        for (unsigned int d = 0; d < dim; ++d)
          u_q[d] = ADNumber(0.0);
        Tensor<2, dim, ADNumber> grad_u_q;
        for (unsigned int d1 = 0; d1 < dim; ++d1)
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            grad_u_q[d1][d2] = ADNumber(0.0);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          u_q += dof_values_ad[j] * fe_values[displacement].value(j, q);
          grad_u_q += dof_values_ad[j] * fe_values[displacement].gradient(j, q);
        }

        Tensor<2, dim, ADNumber> F = unit_symmetric_tensor<dim, ADNumber>() + grad_u_q;
        ADNumber J = determinant(F);
        Tensor<2, dim, ADNumber> F_inv_T = transpose(invert(F));
        Tensor<2, dim, ADNumber> B = F * transpose(F);
        ADNumber trB = trace(B);

        // Compressible Neo-Hookean: P = μ J^(-2/3) [F - (trB/3)F^(-T)] + κ(J-1)J F^(-T)
        P = mu_1 * std::pow(J, -2.0 / 3.0) * (F - (trB / 3.0) * F_inv_T) + k_1 * (J - 1.0) * J * F_inv_T;
      }


      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        residual_ad[i] +=
            (scalar_product(P, fe_values[displacement].gradient(i, q))) *
            fe_values.JxW(q);
      }
    }

    // Boundary integral for Neumann BCs.
    if (cell->at_boundary())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        if (!(cell->face(f)->at_boundary())) continue;

        int boundaryId = cell->face(f)->boundary_id(); 

        if ((boundaryId == 1 && neumannBoundaryTypeId == 0) ||
            (boundaryId == 4 && neumannBoundaryTypeId == 1) ||
            (boundaryId == 4 && neumannBoundaryTypeId == 2))
        {
          fe_face_values.reinit(cell, f);

          for (unsigned int q = 0; q < n_q_face; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              Tensor<1, dim, ADNumber> traction{{0.0, 0.0, 0.0}};

              if (neumannBoundaryTypeId == 0) {
                // pull in X axis
                Tensor<1, dim, ADNumber> t{{0.5, 0.0, 0.0}};
                traction = t;
              } else if (neumannBoundaryTypeId == 1 || neumannBoundaryTypeId == 2) {
                // force from bottom Z face
                Tensor<1, dim, ADNumber> t{{0.0, 0.0, 0.004}};
                traction = t;
              } 

              residual_ad[i] -= traction * fe_face_values[displacement].value(i, q) *
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

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> zero_function;

    for (int dirichletBoundaryId : dirichletBoundaries)
      boundary_functions[dirichletBoundaryId] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
        boundary_values, jacobian_matrix, delta_owned, residual_vector, true);
  }
}

void NonLinearDiffusion::solve_system()
{
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

  // These settings are important for vector-valued problems (elasticity)
  amg_data.constant_modes.resize(1);
  amg_data.constant_modes[0].resize(dof_handler.n_dofs());

  // Set constant modes (rigid body modes) for better convergence
  // For 3D elasticity, there are 6 rigid body modes (3 translations + 3 rotations)
  ComponentMask components(dim, true);
  DoFTools::extract_constant_modes(dof_handler,
                                   components,
                                   amg_data.constant_modes);

  amg_data.elliptic = true; // Problem is elliptic
  amg_data.higher_order_elements = (fe->degree > 1);
  amg_data.smoother_sweeps = 2;
  amg_data.aggregation_threshold = 0.02;

  preconditioner.initialize(jacobian_matrix, amg_data);

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;
}

void NonLinearDiffusion::solve_newton()
{
  pcout << "===============================================" << std::endl;

  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-6;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
  {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    pcout << "Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance)
    {
      solve_system();

      solution_owned += delta_owned;
      solution = solution_owned;
    }
    else
    {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }

  pcout << "===============================================" << std::endl;
}

void NonLinearDiffusion::output() const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  // By passing these two additional arguments to add_data_vector, we specify
  // that the three components of the solution are actually the three components
  // of a vector, so that the visualization program can take that into account.
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  std::vector<std::string> solution_names(dim, "u");

  data_out.add_data_vector(dof_handler,
                           solution,
                           solution_names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-linearelasticity";
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << "." << std::endl;

  pcout << "===============================================" << std::endl;
}
