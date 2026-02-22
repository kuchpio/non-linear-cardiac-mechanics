#ifndef NEO_HOOKE_CONSTITUTIVE_LAW_HPP
#define NEO_HOOKE_CONSTITUTIVE_LAW_HPP

#include "ConstitutiveLaw.hpp"

template <unsigned int dim, class NumberType>
class NeoHookeConstitutiveLaw : public ConstitutiveLaw<dim, NumberType> {
public:
    dealii::Tensor<2, dim, NumberType> computePK1(
            const dealii::Tensor<2, dim, NumberType>& grad_displacement, 
            const MaterialParameters<dim>& materialParameters
        ) const override {
        using namespace dealii;

        double shearModulus = materialParameters.shearModulus();
        double bulkModulus = materialParameters.bulkModulus();

        Tensor<2, dim, NumberType> F = 
            unit_symmetric_tensor<dim, NumberType>() + grad_displacement;
        NumberType J = determinant(F);
        Tensor<2, dim, NumberType> FTinv = invert(transpose(F));
        NumberType I = F.norm_square();

        return shearModulus * std::pow(J, -2.0 / 3.0) * (F - I * FTinv / 3.0) +
            bulkModulus * (J - 1.0) * J * FTinv;
    }
};

#endif
