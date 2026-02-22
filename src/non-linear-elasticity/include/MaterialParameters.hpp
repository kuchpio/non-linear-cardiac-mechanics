#ifndef MATERIAL_PARAMETERS_HPP
#define MATERIAL_PARAMETERS_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

template<unsigned int dim>
class MaterialParameters {
public:

    virtual void initialize(const dealii::Triangulation<dim>& mesh, dealii::ConditionalOStream& pcout) = 0;
    virtual void computeAssembly(const dealii::Quadrature<dim>& quadrature) = 0;
    virtual void computeCell(const typename dealii::DoFHandler<dim>::active_cell_iterator& cell) = 0;
    virtual void computeLocal(const dealii::Point<dim> &point, unsigned int q) = 0;

    virtual double shearModulus() const = 0;
    virtual double bulkModulus() const = 0;
    virtual double b_ff() const = 0; // Fiber-Fiber
    virtual double b_ss() const = 0; // Sheet-Sheet
    virtual double b_nn() const = 0; // Normal-Normal
    virtual double b_fs() const = 0; // Fiber-Sheet
    virtual double b_fn() const = 0; // Fiber-Normal
    virtual double b_sn() const = 0; // Sheet-Normal
    virtual double C() const = 0; // Guccione scaling parameter
    virtual dealii::Tensor<1, dim> fiberDir() const = 0;
    virtual dealii::Tensor<1, dim> sheetDir() const = 0;
    virtual dealii::Tensor<1, dim> normalDir() const = 0;

    virtual ~MaterialParameters() = default;
};

#endif
