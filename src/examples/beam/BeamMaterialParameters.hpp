#ifndef BEAM_MATERIAL_PARAMETERS_HPP
#define BEAM_MATERIAL_PARAMETERS_HPP 

#include "MaterialParameters.hpp"

template<unsigned int dim>
class BeamMaterialParameters : public MaterialParameters<dim> {
public:
    void initialize(const dealii::Triangulation<dim>&, dealii::ConditionalOStream&) override { /* NOP */ }
    void computeAssembly(const dealii::Quadrature<dim>&) override { /* NOP */ }
    void computeCell(const typename dealii::DoFHandler<dim>::active_cell_iterator&) override { /* NOP */ }
    void computeLocal(const dealii::Point<dim> &, unsigned int) override { /* NOP */ }

    double shearModulus() const override { return 10.0; };
    double bulkModulus() const override { return 10.0; };
    double b_ff() const override { return 8.0; }; // Fiber-Fiber
    double b_ss() const override { return 2.0; }; // Sheet-Sheet
    double b_nn() const override { return 2.0; }; // Normal-Normal
    double b_fs() const override { return 4.0; }; // Fiber-Sheet
    double b_fn() const override { return 4.0; }; // Fiber-Normal
    double b_sn() const override { return 2.0; }; // Sheet-Normal
    double C() const override { return 2.0; }; // Guccione scaling parameter
    dealii::Tensor<1, dim> fiberDir() const override { 
        return dealii::Tensor<1, dim>({1.0, 0.0, 0.0}); 
    };
    dealii::Tensor<1, dim> sheetDir() const override {
        return dealii::Tensor<1, dim>({0.0, 1.0, 0.0}); 
    }
    dealii::Tensor<1, dim> normalDir() const override {
        return dealii::Tensor<1, dim>({0.0, 0.0, 1.0}); 
    }
};

#endif
