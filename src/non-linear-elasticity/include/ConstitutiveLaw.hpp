#ifndef CONSTITUTIVE_LAW_HPP
#define CONSTITUTIVE_LAW_HPP

#include <deal.II/base/tensor.h>

#include "MaterialParameters.hpp"

template <unsigned int dim, class NumberType>
class ConstitutiveLaw {
public:
    /* Computes 1st Piola-Kirchhoff stress tensor */
    virtual dealii::Tensor<2, dim, NumberType> computePK1(
            const dealii::Tensor<2, dim, NumberType>& grad_displacement, 
            const MaterialParameters<dim>& materialParameters) const = 0;
    virtual ~ConstitutiveLaw() = default;
};

#endif
