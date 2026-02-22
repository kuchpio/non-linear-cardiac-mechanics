#ifndef BEAM_MESH_PROVIDER_HPP
#define BEAM_MESH_PROVIDER_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>

#include <vector>

#include "MeshProvider.hpp"

template<unsigned int dim>
class BeamMeshProvider : public MeshProvider<dim> {
public:
    dealii::Triangulation<dim> createMesh() const override {
        dealii::Triangulation<dim> mesh;

        dealii::Point<dim> corner1(0., 0., 0.);
        dealii::Point<dim> corner2(10., 1., 1.);

        std::vector<unsigned int> subdivisions(dim);
        int numThreads = 2; // for weak scaling
        subdivisions[0] = 10 * numThreads;
        subdivisions[1] = 2;
        subdivisions[2] = 2;

        dealii::GridGenerator::subdivided_hyper_rectangle(mesh, subdivisions,
                                                corner1, corner2, true);
        mesh.refine_global(1);

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
};

#endif
