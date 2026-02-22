#ifndef HYPER_CUBE_MESH_PROVIDER_HPP
#define HYPER_CUBE_MESH_PROVIDER_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>

#include "MeshProvider.hpp"

template<unsigned int dim>
class HyperCubeMeshProvider : public MeshProvider<dim> {
public:
    HyperCubeMeshProvider(const double _left, const double _right) 
        : left(_left), right(_right)
    { }

    dealii::Triangulation<dim> createMesh() const override {
        dealii::Triangulation<dim> mesh;

        dealii::GridGenerator::hyper_cube(mesh, left, right, true);
        mesh.refine_global(2);

        return mesh;
    }

    std::unique_ptr<dealii::FiniteElement<dim>> createElement(unsigned int degree) const override {
        return std::make_unique<dealii::FE_Q<dim>>(degree);
    }

    std::unique_ptr<dealii::Quadrature<dim>> createQuadrature(unsigned int degree) const override {
        return std::make_unique<dealii::QGauss<dim>>(degree);
    }

    std::unique_ptr<dealii::Quadrature<dim - 1>> createFaceQuadrature(unsigned int degree) const override {
        return std::make_unique<dealii::QGauss<dim - 1>>(degree);
    }
private:
    const double left;
    const double right;
};

#endif
