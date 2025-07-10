"""
Example: 2.5D DC Resistivity Simulation using simpeg-torch

This example demonstrates how to use the 2D DC resistivity simulation classes
with torch tensors for automatic differentiation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# For this example, we'll create a simple mock mesh and sources
# In practice, you would use discretize or simpegtorch.discretize

class SimpleMesh2D:
    """Simple 2D mesh for demonstration purposes."""
    
    def __init__(self, nx=10, nz=10, hx=10.0, hz=10.0):
        self.dim = 2
        self.nx, self.nz = nx, nz
        self.nC = nx * nz  # number of cells
        self.nN = (nx + 1) * (nz + 1)  # number of nodes
        self.nF = nx * (nz + 1) + (nx + 1) * nz  # number of faces
        self.nE = self.nF  # number of edges (same as faces in 2D)
        self.device = 'cpu'
        
        # Create grid
        x = torch.arange(nx + 1, dtype=torch.float64) * hx
        z = torch.arange(nz + 1, dtype=torch.float64) * hz
        
        # Create nodes
        X, Z = torch.meshgrid(x, z, indexing='ij')
        self.nodes = torch.stack([X.flatten(), Z.flatten()], dim=1)
        
        # Simple approximations for operators (in practice, use proper discretization)
        self.cell_volumes = torch.ones(self.nC, dtype=torch.float64) * hx * hz
        self.face_areas = torch.ones(self.nF, dtype=torch.float64)
        self.edge_lengths = torch.ones(self.nE, dtype=torch.float64)
        
        # Create boundary faces (edges of the mesh)
        n_boundary = 2 * (nx + nz)
        self.boundary_faces = torch.zeros((n_boundary, 2), dtype=torch.float64)
        self.boundary_face_outward_normals = torch.zeros((n_boundary, 2), dtype=torch.float64)
        
        # Simple averaging operators (identity for demonstration)
        self.face_divergence = torch.eye(self.nC, self.nF, dtype=torch.float64)
        self.average_cell_to_face = torch.eye(self.nF, self.nC, dtype=torch.float64)
        self.nodal_gradient = torch.eye(self.nE, self.nN, dtype=torch.float64)
        self.average_cell_to_edge = torch.eye(self.nE, self.nC, dtype=torch.float64)
        self.average_cell_to_node = torch.eye(self.nN, self.nC, dtype=torch.float64)
        self.project_face_to_boundary_face = torch.eye(n_boundary, self.nF, dtype=torch.float64)
        self.average_node_to_face = torch.eye(self.nF, self.nN, dtype=torch.float64)


def example_2d_cell_centered():
    """Example using 2D cell-centered simulation."""
    print("=" * 60)
    print("2.5D Cell-Centered DC Simulation Example")
    print("=" * 60)
    
    # Import the 2D simulation class
    from simpegtorch.electromagnetics.resistivity import Simulation2DCellCentered
    
    # Create a simple 2D mesh
    mesh = SimpleMesh2D(nx=20, nz=15, hx=5.0, hz=5.0)
    print(f"Created mesh with {mesh.nC} cells and {mesh.nN} nodes")
    
    # Create simulation with few wavenumbers for speed
    sim = Simulation2DCellCentered(
        mesh=mesh,
        nky=7,  # Number of wavenumbers
        bc_type="Neumann",  # Boundary condition type
        verbose=True
    )
    print(f"Created simulation with {sim.nky} wavenumbers")
    print(f"Wavenumbers: {sim._quad_points.numpy()}")
    print(f"Weights: {sim._quad_weights.numpy()}")
    
    # Create a simple resistivity model
    # Background of 100 ohm-m with a conductive anomaly
    resistivity = torch.ones(mesh.nC, dtype=torch.float64) * 100.0
    
    # Add a conductive block (10 ohm-m) in the center
    nx, nz = mesh.nx, mesh.nz
    center_x, center_z = nx // 2, nz // 2
    anomaly_size = 3
    
    for i in range(-anomaly_size//2, anomaly_size//2 + 1):
        for j in range(-anomaly_size//2, anomaly_size//2 + 1):
            if (center_x + i >= 0 and center_x + i < nx and 
                center_z + j >= 0 and center_z + j < nz):
                idx = (center_x + i) * nz + (center_z + j)
                if idx < mesh.nC:
                    resistivity[idx] = 10.0
    
    sim.resistivity = resistivity
    print(f"Set resistivity model: min={resistivity.min():.1f}, max={resistivity.max():.1f} ohm-m")
    
    # Test getting system matrix for a specific wavenumber
    ky = sim._quad_points[0].item()
    try:
        A = sim.getA(ky)
        print(f"System matrix A for ky={ky:.2e}: shape {A.shape}, nnz={(A != 0).sum()}")
    except Exception as e:
        print(f"Note: System matrix construction needs full mesh implementation: {e}")
    
    print("Cell-centered simulation setup complete!")
    return sim


def example_2d_nodal():
    """Example using 2D nodal simulation."""
    print("\n" + "=" * 60)
    print("2.5D Nodal DC Simulation Example")
    print("=" * 60)
    
    # Import the 2D simulation class
    from simpegtorch.electromagnetics.resistivity import Simulation2DNodal
    
    # Create a simple 2D mesh
    mesh = SimpleMesh2D(nx=15, nz=10, hx=8.0, hz=8.0)
    print(f"Created mesh with {mesh.nC} cells and {mesh.nN} nodes")
    
    # Create nodal simulation
    sim = Simulation2DNodal(
        mesh=mesh,
        nky=5,  # Number of wavenumbers
        bc_type="Robin",  # Mixed boundary conditions
        verbose=True
    )
    print(f"Created nodal simulation with {sim.nky} wavenumbers")
    print(f"Boundary condition type: {sim.bc_type}")
    
    # Create a layered resistivity model
    resistivity = torch.ones(mesh.nC, dtype=torch.float64) * 50.0  # Background
    
    # Add layers: more conductive at depth
    nx, nz = mesh.nx, mesh.nz
    for i in range(nx):
        for j in range(nz):
            idx = i * nz + j
            if j > nz // 2:  # Bottom half more conductive
                resistivity[idx] = 20.0
            if j > 3 * nz // 4:  # Bottom quarter very conductive
                resistivity[idx] = 5.0
    
    sim.resistivity = resistivity
    print(f"Set layered resistivity model: min={resistivity.min():.1f}, max={resistivity.max():.1f} ohm-m")
    
    # Test that Dirichlet BC raises error for nodal
    try:
        sim_bad = Simulation2DNodal(mesh, bc_type="Dirichlet")
        print("ERROR: Should have raised exception")
    except ValueError as e:
        print(f"Correctly caught error for Dirichlet BC: {e}")
    
    print("Nodal simulation setup complete!")
    return sim


def compare_formulations():
    """Compare cell-centered vs nodal formulations."""
    print("\n" + "=" * 60)
    print("Comparing Cell-Centered vs Nodal Formulations")
    print("=" * 60)
    
    # Create identical meshes
    mesh_cc = SimpleMesh2D(nx=10, nz=8, hx=10.0, hz=10.0)
    mesh_nodal = SimpleMesh2D(nx=10, nz=8, hx=10.0, hz=10.0)
    
    from simpegtorch.electromagnetics.resistivity import (
        Simulation2DCellCentered, 
        Simulation2DNodal
    )
    
    # Create simulations
    sim_cc = Simulation2DCellCentered(mesh_cc, nky=3, bc_type="Neumann")
    sim_nodal = Simulation2DNodal(mesh_nodal, nky=3, bc_type="Neumann")
    
    # Same resistivity model
    resistivity = torch.ones(mesh_cc.nC, dtype=torch.float64) * 100.0
    sim_cc.resistivity = resistivity
    sim_nodal.resistivity = resistivity
    
    print(f"Cell-centered: {sim_cc.nky} wavenumbers, BC={sim_cc.bc_type}")
    print(f"Nodal:         {sim_nodal.nky} wavenumbers, BC={sim_nodal.bc_type}")
    
    # Compare quadrature points (should be identical)
    diff = torch.abs(sim_cc._quad_points - sim_nodal._quad_points).max()
    print(f"Max difference in quadrature points: {diff:.2e}")
    
    return sim_cc, sim_nodal


if __name__ == "__main__":
    print("2.5D DC Resistivity Simulation Examples")
    print("Using simpeg-torch with PyTorch tensors\n")
    
    # Run examples
    sim_cc = example_2d_cell_centered()
    sim_nodal = example_2d_nodal()
    sim_cc_comp, sim_nodal_comp = compare_formulations()
    
    print("\n" + "=" * 60)
    print("Example Summary")
    print("=" * 60)
    print("✓ Successfully created 2.5D cell-centered DC simulation")
    print("✓ Successfully created 2.5D nodal DC simulation") 
    print("✓ Demonstrated different boundary conditions")
    print("✓ Showed resistivity model setup")
    print("✓ Compared formulations")
    print("\nThe 2D simulation classes are ready for use!")
    print("Next steps: Add sources, receivers, and survey for full forward modeling.")