"""
Tests for 2D DC resistivity simulations.

This module tests the 2.5D DC resistivity simulation classes to ensure they
can be instantiated and run basic operations without errors.
"""

import torch
import pytest
import numpy as np


class MockMesh2D:
    """Mock 2D mesh for testing purposes."""
    
    def __init__(self, nx=10, nz=8):
        self.dim = 2
        self.nx, self.nz = nx, nz
        self.nC = nx * nz
        self.nN = (nx + 1) * (nz + 1)
        self.nF = nx * (nz + 1) + (nx + 1) * nz
        self.nE = self.nF
        self.device = 'cpu'
        
        # Create simple mesh properties
        self.nodes = torch.randn(self.nN, 2, dtype=torch.float64)
        self.cell_volumes = torch.ones(self.nC, dtype=torch.float64)
        self.face_areas = torch.ones(self.nF, dtype=torch.float64)
        self.edge_lengths = torch.ones(self.nE, dtype=torch.float64)
        
        # Boundary faces
        n_boundary = 2 * (nx + nz)
        self.boundary_faces = torch.randn(n_boundary, 2, dtype=torch.float64)
        self.boundary_face_outward_normals = torch.randn(n_boundary, 2, dtype=torch.float64)
        
        # Mock operators
        self.face_divergence = torch.eye(self.nC, self.nF, dtype=torch.float64)
        self.average_cell_to_face = torch.eye(self.nF, self.nC, dtype=torch.float64)
        self.nodal_gradient = torch.eye(self.nE, self.nN, dtype=torch.float64)
        self.average_cell_to_edge = torch.eye(self.nE, self.nC, dtype=torch.float64)
        self.average_cell_to_node = torch.eye(self.nN, self.nC, dtype=torch.float64)
        self.project_face_to_boundary_face = torch.eye(n_boundary, self.nF, dtype=torch.float64)
        self.average_node_to_face = torch.eye(self.nF, self.nN, dtype=torch.float64)


@pytest.fixture
def mock_mesh_2d():
    """Fixture providing a mock 2D mesh."""
    return MockMesh2D()


@pytest.fixture
def simple_resistivity():
    """Fixture providing a simple resistivity model."""
    def _resistivity(nC):
        return torch.ones(nC, dtype=torch.float64) * 100.0
    return _resistivity


def test_import_2d_simulations():
    """Test that 2D simulation classes can be imported."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import (
        BaseDCSimulation2D,
        Simulation2DCellCentered,
        Simulation2DNodal,
        Simulation2DCellCentred  # UK spelling
    )
    
    # Check that classes exist
    assert BaseDCSimulation2D is not None
    assert Simulation2DCellCentered is not None
    assert Simulation2DNodal is not None
    assert Simulation2DCellCentred is Simulation2DCellCentered


def test_import_from_main_module():
    """Test that 2D simulations can be imported from main module."""
    from simpegtorch.electromagnetics.resistivity import (
        Simulation2DCellCentered,
        Simulation2DNodal,
        BaseDCSimulation2D
    )
    
    assert Simulation2DCellCentered is not None
    assert Simulation2DNodal is not None
    assert BaseDCSimulation2D is not None


def test_cell_centered_instantiation(mock_mesh_2d):
    """Test that cell-centered 2D simulation can be instantiated."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    # Test basic instantiation
    sim = Simulation2DCellCentered(mock_mesh_2d, nky=5, verbose=False)
    
    assert sim.nky == 5
    assert sim.mesh is mock_mesh_2d
    assert sim.bc_type == "Robin"  # default
    assert len(sim._quad_points) == 5
    assert len(sim._quad_weights) == 5
    
    # Test different boundary conditions
    sim_neumann = Simulation2DCellCentered(mock_mesh_2d, bc_type="Neumann")
    assert sim_neumann.bc_type == "Neumann"
    
    sim_dirichlet = Simulation2DCellCentered(mock_mesh_2d, bc_type="Dirichlet")
    assert sim_dirichlet.bc_type == "Dirichlet"


def test_nodal_instantiation(mock_mesh_2d):
    """Test that nodal 2D simulation can be instantiated."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DNodal
    
    # Test basic instantiation
    sim = Simulation2DNodal(mock_mesh_2d, nky=3, verbose=False)
    
    assert sim.nky == 3
    assert sim.mesh is mock_mesh_2d
    assert sim.bc_type == "Robin"  # default
    assert len(sim._quad_points) == 3
    assert len(sim._quad_weights) == 3
    
    # Test Neumann BC
    sim_neumann = Simulation2DNodal(mock_mesh_2d, bc_type="Neumann")
    assert sim_neumann.bc_type == "Neumann"
    
    # Test that Dirichlet BC raises error
    with pytest.raises(ValueError, match="Dirichlet BC not supported"):
        Simulation2DNodal(mock_mesh_2d, bc_type="Dirichlet")


def test_wrong_mesh_dimension():
    """Test that 3D mesh raises error for 2D simulation."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    # Create 3D mesh mock
    class MockMesh3D:
        def __init__(self):
            self.dim = 3
    
    mesh_3d = MockMesh3D()
    
    with pytest.raises(ValueError, match="2D simulation requires 2D mesh"):
        Simulation2DCellCentered(mesh_3d)


def test_resistivity_setting(mock_mesh_2d, simple_resistivity):
    """Test setting resistivity model."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    sim = Simulation2DCellCentered(mock_mesh_2d, nky=3)
    resistivity = simple_resistivity(mock_mesh_2d.nC)
    
    # Test setting resistivity
    sim.resistivity = resistivity
    assert torch.allclose(sim.resistivity, resistivity)
    assert sim.resistivity.shape == (mock_mesh_2d.nC,)


def test_quadrature_properties(mock_mesh_2d):
    """Test quadrature point and weight properties."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    sim = Simulation2DCellCentered(mock_mesh_2d, nky=7)
    
    # Check quadrature properties
    assert sim._quad_points.dtype == torch.float64
    assert sim._quad_weights.dtype == torch.float64
    assert len(sim._quad_points) == 7
    assert len(sim._quad_weights) == 7
    assert torch.all(sim._quad_points > 0)  # All wavenumbers should be positive
    assert torch.all(sim._quad_weights > 0)  # All weights should be positive


def test_system_matrix_construction(mock_mesh_2d, simple_resistivity):
    """Test that system matrix can be constructed without errors."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    sim = Simulation2DCellCentered(mock_mesh_2d, nky=3)
    sim.resistivity = simple_resistivity(mock_mesh_2d.nC)
    
    # Test getting system matrix for first wavenumber
    ky = sim._quad_points[0].item()
    A = sim.getA(ky)
    
    assert isinstance(A, torch.Tensor)
    assert A.shape == (mock_mesh_2d.nC, mock_mesh_2d.nC)
    assert A.dtype == torch.float64


def test_nodal_system_matrix(mock_mesh_2d, simple_resistivity):
    """Test nodal system matrix construction."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DNodal
    
    sim = Simulation2DNodal(mock_mesh_2d, nky=3, bc_type="Neumann")
    sim.resistivity = simple_resistivity(mock_mesh_2d.nC)
    
    # Test getting system matrix
    ky = sim._quad_points[0].item()
    A = sim.getA(ky)
    
    assert isinstance(A, torch.Tensor)
    # Nodal formulation uses nodes, but we constructed mock operators for simplicity
    assert A.dtype == torch.float64


def test_rhs_construction(mock_mesh_2d):
    """Test RHS construction."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    # Create mock survey
    class MockSurvey:
        def __init__(self, n_sources=2):
            self.source_list = [MockSource() for _ in range(n_sources)]
    
    class MockSource:
        def evaluate(self, sim):
            return torch.randn(sim.mesh.nC, dtype=torch.float64)
    
    survey = MockSurvey(n_sources=3)
    sim = Simulation2DCellCentered(mock_mesh_2d, survey=survey, nky=3)
    
    # Test RHS construction
    ky = sim._quad_points[0].item()
    rhs = sim.getRHS(ky)
    
    assert isinstance(rhs, torch.Tensor)
    assert rhs.shape == (mock_mesh_2d.nC, 3)  # 3 sources
    assert rhs.dtype == torch.float64


def test_different_wavenumber_counts():
    """Test simulations with different numbers of wavenumbers."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    mesh = MockMesh2D()
    
    for nky in [3, 5, 7, 11]:
        sim = Simulation2DCellCentered(mesh, nky=nky)
        assert sim.nky == nky
        assert len(sim._quad_points) == nky
        assert len(sim._quad_weights) == nky


def test_inheritance_hierarchy():
    """Test that inheritance hierarchy is correct."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import (
        BaseDCSimulation2D,
        Simulation2DCellCentered,
        Simulation2DNodal
    )
    from simpegtorch.electromagnetics.resistivity.simulation import BaseDcSimulation
    
    # Test inheritance
    assert issubclass(Simulation2DCellCentered, BaseDCSimulation2D)
    assert issubclass(Simulation2DNodal, BaseDCSimulation2D)
    assert issubclass(BaseDCSimulation2D, BaseDcSimulation)
    
    # Test MRO (Method Resolution Order)
    assert BaseDCSimulation2D in Simulation2DCellCentered.__mro__
    assert BaseDCSimulation2D in Simulation2DNodal.__mro__
    assert BaseDcSimulation in BaseDCSimulation2D.__mro__


@pytest.mark.parametrize("bc_type", ["Neumann", "Robin"])
def test_boundary_conditions_cell_centered(mock_mesh_2d, bc_type):
    """Test different boundary conditions for cell-centered simulation."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    sim = Simulation2DCellCentered(mock_mesh_2d, bc_type=bc_type, nky=3)
    assert sim.bc_type == bc_type


@pytest.mark.parametrize("bc_type", ["Neumann", "Robin"])
def test_boundary_conditions_nodal(mock_mesh_2d, bc_type):
    """Test different boundary conditions for nodal simulation."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DNodal
    
    sim = Simulation2DNodal(mock_mesh_2d, bc_type=bc_type, nky=3)
    assert sim.bc_type == bc_type


def test_verbose_output(mock_mesh_2d, capsys):
    """Test verbose output during initialization."""
    from simpegtorch.electromagnetics.resistivity.simulation_2d import Simulation2DCellCentered
    
    # Test with verbose=True
    sim = Simulation2DCellCentered(mock_mesh_2d, nky=3, verbose=True)
    captured = capsys.readouterr()
    assert "trapezoidal" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__])