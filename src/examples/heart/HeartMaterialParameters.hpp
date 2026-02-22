#ifndef HEART_MATERIAL_PARAMETERS_HPP
#define HEART_MATERIAL_PARAMETERS_HPP

#include "MaterialParameters.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/numbers.h>
#include <iostream>
#include <fstream>
#include <sstream>

class HeartMaterialParameters : public MaterialParameters<3> {
public:
  static constexpr unsigned int dim = 3;

  HeartMaterialParameters(const std::string& materialFile_) 
      : materialFile(materialFile_) 
  {}

  void initialize(
          const dealii::Triangulation<dim>& mesh, 
          dealii::ConditionalOStream& pcout
    ) override {
    using namespace dealii;

    pcout << "Loading material coordinates" << std::endl;

    // Read material coordinates from file
    std::ifstream mat_file(materialFile);
    if (!mat_file) {
      pcout << "Error: Could not open material_coordinates.txt" << std::endl;
      std::abort();
    }

    std::string line;
    std::getline(mat_file, line); // Skip header

    std::map<types::global_dof_index, dealii::Vector<double>> node_to_material;


    while (std::getline(mat_file, line)) {
      std::istringstream iss(line);
      unsigned int node_tag;
      double xi, theta_hat, cosphi, sinphi;

      if (iss >> node_tag >> xi >> theta_hat >> cosphi >> sinphi) {
        dealii::Vector<double> v(4);
        v[0]=xi; v[1]=theta_hat; v[2]=cosphi; v[3]=sinphi;
        node_to_material[node_tag] = v;
      }

    }
    mat_file.close();

    pcout << "  Read material coordinates for " << node_to_material.size()
          << " nodes" << std::endl;

    // Initialize FE space for material coordinates (4 components: xi, thetahat,
    // cosphi, sinphi)
    fe_material = std::make_unique<FESystem<dim>>(FE_Q<dim>(1), 4);
    dof_handler_material.reinit(mesh);
    dof_handler_material.distribute_dofs(*fe_material);

    // Initialize material solution vector
    IndexSet locally_owned_dofs_mat = dof_handler_material.locally_owned_dofs();
    IndexSet locally_relevant_dofs_mat =
        DoFTools::extract_locally_relevant_dofs(dof_handler_material);

    TrilinosWrappers::MPI::Vector material_owned;
    material_owned.reinit(locally_owned_dofs_mat, MPI_COMM_WORLD);
    material_solution.reinit(locally_owned_dofs_mat, locally_relevant_dofs_mat,
                             MPI_COMM_WORLD);

    // Build a mapping from Cartesian positions to material coordinates
    std::vector<Point<3>> node_positions;
    std::vector<dealii::Vector<double>> mat_coords;

    const double d = 2.91;
    const double z_base = 1.19;

    for (const auto &entry : node_to_material) {
      const dealii::Vector<double> &mat_coord = entry.second;

      double xi        = mat_coord[0];
      double theta_hat = mat_coord[1];
      double cosphi    = mat_coord[2];
      double sinphi    = mat_coord[3];

      // Transform to Cartesian
      double theta_0 = std::acos(z_base / (d * std::cosh(xi)));
      double theta = theta_0 + theta_hat * (numbers::PI - theta_0);


      Point<3> cartesian(d * std::sinh(xi) * std::sin(theta) * cosphi,
                         d * std::sinh(xi) * std::sin(theta) * sinphi,
                         d * std::cosh(xi) * std::cos(theta));

      node_positions.push_back(cartesian);
      mat_coords.push_back(mat_coord);
    }

    // Now assign material coordinates to vertices by position matching
    std::vector<types::global_dof_index> dof_indices(
        fe_material->dofs_per_cell);

    // Track which DoFs we've set to avoid overwriting with worse matches
    std::map<types::global_dof_index, double> dof_best_distance;

    for (const auto &cell : dof_handler_material.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(dof_indices);

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        const Point<dim> &vertex_pos = cell->vertex(v);

        // Find closest node in our material coordinates
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;

        for (size_t i = 0; i < node_positions.size(); ++i) {
          double dist = vertex_pos.distance(node_positions[i]);
          if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
          }
        }

        if (closest_idx >= 0 && min_dist < 1e-4) {
          const dealii::Vector<double> &mat_coord = mat_coords[closest_idx];


          // Get DoF indices for this vertex
          unsigned int xi_dof     = fe_material->component_to_system_index(0, v);
          unsigned int theta_dof  = fe_material->component_to_system_index(1, v);
          unsigned int cosphi_dof = fe_material->component_to_system_index(2, v);
          unsigned int sinphi_dof = fe_material->component_to_system_index(3, v);


          auto global_xi     = dof_indices[xi_dof];
          auto global_theta  = dof_indices[theta_dof];
          auto global_cosphi = dof_indices[cosphi_dof];
          auto global_sinphi = dof_indices[sinphi_dof];

          // Only set if this is a better match than what we had before
          auto it_xi = dof_best_distance.find(global_xi);

          if (it_xi == dof_best_distance.end() || min_dist < it_xi->second) {
            material_owned[global_xi]     = mat_coord[0];
            material_owned[global_theta]  = mat_coord[1];
            material_owned[global_cosphi] = mat_coord[2];
            material_owned[global_sinphi] = mat_coord[3];
            dof_best_distance[global_xi]     = min_dist;
            dof_best_distance[global_theta]  = min_dist;
            dof_best_distance[global_cosphi] = min_dist;
            dof_best_distance[global_sinphi] = min_dist;

          }
        } else if (min_dist >= 1e-4) {
          pcout << "Warning: No close match found for vertex at ("
                << vertex_pos[0] << ", " << vertex_pos[1] << ", "
                << vertex_pos[2] << "), min_dist = " << min_dist << std::endl;
        }
      }
    }

    material_owned.compress(VectorOperation::insert);
    material_solution = material_owned;

    pcout << "  Material coordinates loaded and interpolated" << std::endl;
    pcout << "  Set material coordinates for " << dof_best_distance.size() / 4
      << " unique DoFs" << std::endl;

  }

  void computeAssembly(const dealii::Quadrature<dim>& quadrature) override {
      fe_values_material = std::make_unique<dealii::FEValues<dim>>(*fe_material, quadrature, dealii::update_values);
      mat_vals.resize(quadrature.size(), dealii::Vector<double>(4));
  }

  void computeCell(const typename dealii::DoFHandler<dim>::active_cell_iterator& cell) override {
      // Get material coordinates at this quadrature point
      fe_values_material->reinit(cell);  // Same cell as displacement
      fe_values_material->get_function_values(material_solution, mat_vals);
  }

  void computeLocal(const dealii::Point<dim> &, unsigned int q) override {
      using namespace dealii;

      double xi         = mat_vals[q][0];
      double theta_hat  = mat_vals[q][1];
      double cosphi     = mat_vals[q][2];
      double sinphi     = mat_vals[q][3];
      const double r = std::sqrt(cosphi*cosphi + sinphi*sinphi);
      if (r > 1e-12) {
        cosphi /= r;
        sinphi /= r;
      } else {
        // fallback (should never happen)
        cosphi = 1.0;
        sinphi = 0.0;
      }

      // Compute actual theta from theta_hat
      const double d = 2.91;
      const double z_base = 1.19;
      const double theta_0 = std::acos(z_base / (d * std::cosh(xi)));
      const double theta = theta_0 + theta_hat * (numbers::PI - theta_0);

      // Compute basis vectors in prolate ellipsoidal coordinates
      Tensor<1, 3> g_xi, g_theta, g_phi;

      g_xi[0] = d * std::cosh(xi) * std::sin(theta) * cosphi;
      g_xi[1] = d * std::cosh(xi) * std::sin(theta) * sinphi;
      g_xi[2] = d * std::sinh(xi) * std::cos(theta);

      g_theta[0] = d * std::sinh(xi) * std::cos(theta) * cosphi;
      g_theta[1] = d * std::sinh(xi) * std::cos(theta) * sinphi;
      g_theta[2] = -d * std::cosh(xi) * std::sin(theta);

      g_phi[0] = -d * std::sinh(xi) * std::sin(theta) * sinphi;
      g_phi[1] =  d * std::sinh(xi) * std::sin(theta) * cosphi;
      g_phi[2] = 0.0;

      // Normalize to get unit vectors
      Tensor<1, 3> e_xi    = g_xi / g_xi.norm();      // radial (transmural)
      Tensor<1, 3> e_theta = g_theta / g_theta.norm(); // longitudinal
      Tensor<1, 3> e_phi   = g_phi / g_phi.norm();     // circumferential

      // BELOW FOUR LINES -- SYMMETRIC CASE
      // For symmetric case (Picture C of paper 2): orthogonal triplet
      // Tensor<1, dim, double> f_fiber  = e_theta;  // fiber direction (longitudinal)
      // Tensor<1, dim, double> s_sheet  = e_phi;    // sheet direction (circumferential)
      // Tensor<1, dim, double> n_normal = e_xi;     // normal direction (radial)

      // BELOW: the, f, s directions are as in picture 2.B of paper2
      const double alpha_endo = -60.0 * numbers::PI / 180.0;  // = -π/3 ≈ -1.0472
      const double alpha_epi  =  60.0 * numbers::PI / 180.0;  // =  π/3 ≈  1.0472
      const double xi_endo = 0.6; // from heart generation in setup()
      const double xi_epi = 1.02; // from heart generation in setup() 
      double wall_fraction = (xi - xi_endo) / (xi_epi - xi_endo); // TODO double check this
      const double alpha = alpha_endo + (alpha_epi - alpha_endo) * wall_fraction;

      n_normal = e_xi; 
      f_fiber =
          cos(alpha) * e_phi   // circumferential
        + sin(alpha) * e_theta; // longitudinal

      f_fiber /= f_fiber.norm();

      s_sheet = cross_product_3d(n_normal, f_fiber);
      s_sheet /= s_sheet.norm();
  }

  double shearModulus() const override { return 0.0; };
  double bulkModulus() const override { return 10.0; };
  double b_ff() const override { return 1.0; }; // Fiber-Fiber
  double b_ss() const override { return 2.0; }; // Sheet-Sheet
  double b_nn() const override { return 2.0; }; // Normal-Normal
  double b_fs() const override { return 8.0; }; // Fiber-Sheet
  double b_fn() const override { return 8.0; }; // Fiber-Normal
  double b_sn() const override { return 2.0; }; // Sheet-Normal
  double C() const override { return 10.0; };    // Guccione scaling parameter
  dealii::Tensor<1, dim> fiberDir() const override {
    return f_fiber;
  };
  dealii::Tensor<1, dim> sheetDir() const override {
    return s_sheet;
  }
  dealii::Tensor<1, dim> normalDir() const override {
    return n_normal;
  }

private:
  std::string materialFile;

  // ====== Computed during initialization ======
  // Finite element for material coordinates
  std::unique_ptr<dealii::FiniteElement<dim>> fe_material;
  dealii::DoFHandler<dim> dof_handler_material;
  dealii::TrilinosWrappers::MPI::Vector material_solution;

  // ====== Computed per assembly ======
  std::unique_ptr<dealii::FEValues<dim>> fe_values_material;

  // ====== Computed per cell ======
  std::vector<dealii::Vector<double>> mat_vals;

  // ====== Computed per quadrature point ======
  dealii::Tensor<1, dim> f_fiber;
  dealii::Tensor<1, dim> s_sheet;
  dealii::Tensor<1, dim> n_normal;
};

#endif
