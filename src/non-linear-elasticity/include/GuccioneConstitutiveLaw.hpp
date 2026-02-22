#ifndef GUCCIONE_CONSTITUTIVE_LAW_HPP
#define GUCCIONE_CONSTITUTIVE_LAW_HPP

#include "ConstitutiveLaw.hpp"

template <unsigned int dim, class NumberType>
class GuccioneConstitutiveLaw : public ConstitutiveLaw<dim, NumberType> {
public:
    dealii::Tensor<2, dim, NumberType> computePK1(
            const dealii::Tensor<2, dim, NumberType>& grad_displacement, 
            const MaterialParameters<dim>& materialParameters
        ) const override {
        using namespace dealii;

        // Guccione material parameters
        const double b_ff = materialParameters.b_ff(); // fiber direction
        const double b_ss = materialParameters.b_ss(); // sheet direction
        const double b_nn = materialParameters.b_nn(); // normal direction
        const double b_fs = materialParameters.b_fs(); // fiber-sheet coupling
        const double b_fn = materialParameters.b_fn(); // fiber-normal coupling
        const double b_sn = materialParameters.b_sn(); // sheet-normal coupling
        const double C_param = materialParameters.C(); // scaling parameter
        const double B_bulk = materialParameters.bulkModulus(); // bulk modulus for volumetric penalty

        // Define fiber directions (these should ideally come from your mesh
        // data) For now, using simple orthogonal directions
        Tensor<1, dim> f_fiber = materialParameters.fiberDir(); // fiber direction
        Tensor<1, dim> s_sheet = materialParameters.sheetDir(); // sheet direction
        Tensor<1, dim> n_normal = materialParameters.normalDir(); // normal direction

        // Deformation gradient and related tensors
        Tensor<2, dim, NumberType> F = unit_symmetric_tensor<dim, NumberType>() + grad_displacement;
        NumberType J = determinant(F);
        Tensor<2, dim, NumberType> C = transpose(F) * F; // Right Cauchy-Green tensor

        // Green-Lagrange strain tensor: E = 0.5 * (C - I)
        Tensor<2, dim, NumberType> E = 0.5 * (C - unit_symmetric_tensor<dim, NumberType>());

        // Compute strain components in fiber coordinate system
        NumberType E_ff = scalar_product(f_fiber, E * f_fiber);
        NumberType E_ss = scalar_product(s_sheet, E * s_sheet);
        NumberType E_nn = scalar_product(n_normal, E * n_normal);
        NumberType E_fs = scalar_product(f_fiber, E * s_sheet);
        NumberType E_fn = scalar_product(f_fiber, E * n_normal);
        NumberType E_sn = scalar_product(s_sheet, E * n_normal);

        NumberType E_sf = scalar_product(s_sheet, E * f_fiber);
        NumberType E_nf = scalar_product(n_normal, E * f_fiber);
        NumberType E_ns = scalar_product(n_normal, E * s_sheet);

        // Compute Q (exponent)
        NumberType Q = b_ff * E_ff * E_ff +
                  b_ss * E_ss * E_ss +
                  b_nn * E_nn * E_nn +
                  b_fs * (E_fs * E_fs + E_sf * E_sf) + 
                  b_fn * (E_fn * E_fn + E_nf * E_nf) + 
                  b_sn * (E_sn * E_sn + E_ns * E_ns);

        // Second Piola-Kirchhoff stress: S = dW/dE
        // For Guccione: S_iso = C * exp(Q) * dQ/dE
        NumberType exp_Q = exp(Q);

        // Derivatives of Q with respect to E components
        Tensor<2, dim, NumberType> dQ_dE;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            dQ_dE[i][j] = NumberType(0.0);

        // Build dQ/dE tensor
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j) {
            dQ_dE[i][j] += 2.0 * b_ff * E_ff * (f_fiber[i] * f_fiber[j]);
            dQ_dE[i][j] += 2.0 * b_ss * E_ss * (s_sheet[i] * s_sheet[j]);
            dQ_dE[i][j] += 2.0 * b_nn * E_nn * (n_normal[i] * n_normal[j]);
            dQ_dE[i][j] += 4.0 * b_fs * E_fs * (f_fiber[i] * s_sheet[j] + s_sheet[i] * f_fiber[j]) / 2.0;
            dQ_dE[i][j] += 4.0 * b_fn * E_fn * (f_fiber[i] * n_normal[j] + n_normal[i] * f_fiber[j]) / 2.0;
            dQ_dE[i][j] += 4.0 * b_sn * E_sn * (s_sheet[i] * n_normal[j] + n_normal[i] * s_sheet[j]) / 2.0;
          }

        // Total second Piola-Kirchhoff stress
        Tensor<2, dim, NumberType> S = 0.5 * C_param * exp_Q * dQ_dE;

        // First Piola-Kirchhoff stress: P = F * S
        return F * S + B_bulk * (J - 1.0) * J * invert(transpose(F));
    }
};

#endif
