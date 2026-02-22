#ifndef FILE_MESH_PROVIDER_HPP
#define FILE_MESH_PROVIDER_HPP

#include <deal.II/grid/grid_in.h>

#include "MeshProvider.hpp"

template<unsigned int dim, 
    template<unsigned int> class FiniteElementType, 
    template<unsigned int> class QuadratureType
>
class FileMeshProvider : public MeshProvider<dim> {
public:
    FileMeshProvider(const std::string& meshFileName_) : meshFileName(meshFileName_) 
    { }

    dealii::Triangulation<dim> createMesh() const override {
        dealii::Triangulation<dim> mesh;

        dealii::GridIn<dim> grid;
        grid.attach_triangulation(mesh);

        std::ifstream gridInputFile(meshFileName);
        grid.read_msh(gridInputFile);

        return mesh;
    }

    std::unique_ptr<dealii::FiniteElement<dim>> createElement(unsigned int degree) const override {
        return std::make_unique<FiniteElementType<dim>>(degree);
    }

    std::unique_ptr<dealii::Quadrature<dim>> createQuadrature(unsigned int degree) const override {
        return std::make_unique<QuadratureType<dim>>(degree);
    }

    std::unique_ptr<dealii::Quadrature<dim - 1>> createFaceQuadrature(unsigned int degree) const override {
        return std::make_unique<QuadratureType<dim - 1>>(degree);
    }
private:
  const std::string meshFileName;
};

#endif
