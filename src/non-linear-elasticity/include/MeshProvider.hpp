#ifndef MESH_PROVIDER_HPP
#define MESH_PROVIDER_HPP

#include <deal.II/fe/fe.h>
#include <deal.II/grid/tria.h>

template <unsigned int dim>
class MeshProvider {
public:
    virtual dealii::Triangulation<dim> createMesh() const = 0;
    virtual std::unique_ptr<dealii::FiniteElement<dim>> createElement(unsigned int degree) const = 0;
    virtual std::unique_ptr<dealii::Quadrature<dim>> createQuadrature(unsigned int degree) const = 0;
    virtual std::unique_ptr<dealii::Quadrature<dim - 1>> createFaceQuadrature(unsigned int degree) const = 0;
    virtual ~MeshProvider() = default;
};

#endif
