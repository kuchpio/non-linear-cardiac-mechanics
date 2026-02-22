#!/usr/bin/env python3
"""
Heart mesh generator using Gmsh with prolate ellipsoidal coordinates.
Generates structured hexahedral mesh.

Usage:
    python heart_mesh.py              # Generate mesh
    python heart_mesh.py -gui         # Generate and show in GUI
"""

import gmsh
import sys
import math

def prolate_to_cartesian(xi, theta_hat, phi, d=2.91, z_base=1.19):
    """
    Convert prolate ellipsoidal coordinates to Cartesian.
    """
    theta_0 = math.acos(z_base / (d * math.cosh(xi)))
    theta = theta_0 + theta_hat * (math.pi - theta_0)
    
    x = d * math.sinh(xi) * math.sin(theta) * math.cos(phi)
    y = d * math.sinh(xi) * math.sin(theta) * math.sin(phi)
    z = d * math.cosh(xi) * math.cos(theta)
    
    return x, y, z


def create_heart_mesh_structured(n_xi=12, n_theta=24, n_phi=48):
    """
    Create a structured hexahedral heart mesh.
    """
    # Parameters
    xi_min = 0.6
    xi_max = 1.02
    d = 2.91
    z_base = 1.19
    
    gmsh.initialize()
    gmsh.model.add("heart")
    gmsh.option.setNumber("General.Terminal", 1)
    
    print(f"Creating structured heart mesh: {n_xi}×{n_theta}×{n_phi} elements")
    
    # Create discrete volume entity first
    vol_tag = gmsh.model.addDiscreteEntity(3)
    
    # Create all nodes
    nodes = {}
    node_tag = 1
    
    for i in range(n_xi + 1):
        xi = xi_min + i * (xi_max - xi_min) / n_xi
        for j in range(n_theta + 1):
            theta_hat = j / n_theta
            for k in range(n_phi):  # Note: n_phi, not n_phi+1 (periodic)
                phi = k * 2 * math.pi / n_phi
                x, y, z = prolate_to_cartesian(xi, theta_hat, phi, d, z_base)
                nodes[(i, j, k)] = (node_tag, x, y, z)
                node_tag += 1
    
    print(f"Created {node_tag - 1} nodes")
    
    # Create hexahedral elements
    elements = []
    elem_tag = 1
    
    # Boundary face tracking
    endo_faces = []  # xi = xi_min
    epi_faces = []   # xi = xi_max
    base_faces = []  # theta = 0
    
    for i in range(n_xi):
        for j in range(n_theta):
            for k in range(n_phi):
                k_plus = (k + 1) % n_phi  # Periodic in phi
                
                # Node ordering for hexahedron (Gmsh hex8 convention):
                # Bottom face (j): 0-1-2-3, Top face (j+1): 4-5-6-7
                n0 = nodes[(i, j, k)][0]
                n1 = nodes[(i+1, j, k)][0]
                n2 = nodes[(i+1, j, k_plus)][0]
                n3 = nodes[(i, j, k_plus)][0]
                n4 = nodes[(i, j+1, k)][0]
                n5 = nodes[(i+1, j+1, k)][0]
                n6 = nodes[(i+1, j+1, k_plus)][0]
                n7 = nodes[(i, j+1, k_plus)][0]
                
                elements.append((elem_tag, [n0, n1, n2, n3, n4, n5, n6, n7]))
                
                # Track boundary faces
                if i == 0:  # Endocardium
                    endo_faces.append((elem_tag, [n0, n3, n7, n4]))
                if i == n_xi - 1:  # Epicardium
                    epi_faces.append((elem_tag, [n1, n2, n6, n5]))
                if j == 0:  # Base
                    base_faces.append((elem_tag, [n0, n1, n2, n3]))
                
                elem_tag += 1
    
    print(f"Created {len(elements)} hexahedral elements")
    
    # Add nodes to volume entity
    node_tags = []
    node_coords = []
    for key in sorted(nodes.keys()):
        tag, x, y, z = nodes[key]
        node_tags.append(tag)
        node_coords.extend([x, y, z])
    
    gmsh.model.mesh.addNodes(3, vol_tag, node_tags, node_coords)
    
    # Add hexahedral elements
    # Element type 5 is 8-node hexahedron
    elem_tags = []
    elem_node_tags = []
    for tag, node_list in elements:
        elem_tags.append(tag)
        elem_node_tags.extend(node_list)
    
    gmsh.model.mesh.addElementsByType(vol_tag, 5, elem_tags, elem_node_tags)
    
    # Create physical groups for boundaries
    def add_boundary_surfaces(face_list, phys_id, name):
        if not face_list:
            return
        
        # Create a discrete surface entity
        surf_tag = gmsh.model.addDiscreteEntity(2)
        
        # Collect all nodes that are on this boundary (to add to surface)
        boundary_nodes = set()
        for _, face_nodes in face_list:
            boundary_nodes.update(face_nodes)
        
        # Add nodes to surface
        surf_node_tags = list(boundary_nodes)
        surf_node_coords = []
        for node_tag in surf_node_tags:
            # Find coordinates
            for key, (tag, x, y, z) in nodes.items():
                if tag == node_tag:
                    surf_node_coords.extend([x, y, z])
                    break
        
        gmsh.model.mesh.addNodes(2, surf_tag, surf_node_tags, surf_node_coords)
        
        # Add quad elements (type 3 = 4-node quadrangle)
        quad_tags = []
        quad_nodes = []
        quad_tag = 1
        
        for _, face_nodes in face_list:
            quad_tags.append(quad_tag)
            quad_nodes.extend(face_nodes)
            quad_tag += 1
        
        gmsh.model.mesh.addElementsByType(surf_tag, 3, quad_tags, quad_nodes)
        
        # Create physical group
        gmsh.model.addPhysicalGroup(2, [surf_tag], phys_id)
        gmsh.model.setPhysicalName(2, phys_id, name)
        
        print(f"  {name}: {len(face_list)} faces (boundary {phys_id})")
    
    print("Creating physical groups...")
    add_boundary_surfaces(endo_faces, 1, "Endocardium")
    add_boundary_surfaces(epi_faces, 2, "Epicardium")
    add_boundary_surfaces(base_faces, 3, "Base")
    
    # Add physical volume
    gmsh.model.addPhysicalGroup(3, [vol_tag], 1)
    gmsh.model.setPhysicalName(3, 1, "Heart")
    
    print("Finalizing mesh topology...")
    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.createGeometry()
    
    print("Mesh created successfully!")
    
    # Save material coordinates (xi, theta_hat, phi) as a text file
    print("Saving material coordinates...")
    
    # Create mapping from node tags to material coordinates
    node_to_param = {}  # Maps node_tag to (xi, theta_hat, phi)
    
    for i in range(n_xi + 1):
        xi = xi_min + i * (xi_max - xi_min) / n_xi
        for j in range(n_theta + 1):
            theta_hat = j / n_theta
            for k in range(n_phi):
                phi = k * 2 * math.pi / n_phi
                node_tag, x, y, z = nodes[(i, j, k)]
                cosphi = math.cos(phi)
                sinphi = math.sin(phi)
                node_to_param[node_tag] = (xi, theta_hat, cosphi, sinphi)

    
    # Save to text file for easy loading in deal.II
    with open("material_coordinates.txt", "w") as f:
        f.write("# node_tag xi theta_hat cosphi sinphi\n")
        for node_tag in sorted(node_to_param.keys()):
            xi, theta_hat, cosphi, sinphi = node_to_param[node_tag]
            f.write(f"{node_tag} {xi:.10f} {theta_hat:.10f} {cosphi:.10f} {sinphi:.10f}\n")

    
    print(f"  Material coordinates saved for {len(node_to_param)} nodes")
    print(f"  File: material_coordinates.txt")
    
    return gmsh.model


if __name__ == "__main__":
    show_gui = '-gui' in sys.argv
    
    # Mesh resolution
    n_xi = 12
    n_theta = 24
    n_phi = 48
    
    # Check for refinement argument
    for i, arg in enumerate(sys.argv):
        if arg == '-refine' and i + 1 < len(sys.argv):
            try:
                factor = int(sys.argv[i + 1])
                n_xi *= factor
                n_theta *= factor
                n_phi *= factor
                print(f"Refining mesh by factor {factor}")
            except:
                pass
    
    create_heart_mesh_structured(n_xi, n_theta, n_phi)
    
    # Save mesh in MSH2 format (better compatibility with deal.II)
    output_file = "heart_mesh.msh"
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)  # Save all entities
    gmsh.write(output_file)
    
    print(f"\n{'='*60}")
    print(f"Mesh generation complete!")
    print(f"{'='*60}")
    print(f"Mesh file: {output_file}")
    print(f"Material coordinates: material_coordinates.txt")
    print(f"Format: MSH 2.2 (compatible with deal.II)")
    print(f"Total elements: {n_xi * n_theta * n_phi}")
    print(f"Total nodes: {(n_xi + 1) * (n_theta + 1) * n_phi}")
    print(f"\nBoundary IDs:")
    print(f"  1 = Endocardium (inner surface, xi={0.6})")
    print(f"  2 = Epicardium (outer surface, xi={1.02})")
    print(f"  3 = Base (theta_hat=0)")
    print(f"{'='*60}")
    
    if show_gui:
        gmsh.fltk.run()
    
    gmsh.finalize()
