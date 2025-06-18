"""
Test DC resistivity nodal simulation with gradients and autodifferentiation

These tests verify the complete gradient flow through the nodal DC simulation:
1. Source discretization to mesh nodes
2. Edge inner product computation with inversion
3. System matrix assembly with Neumann boundary conditions
4. Linear system solve with TorchMatSolver
5. Receiver evaluation from nodal potentials
6. Gradient computation through automatic differentiation
"""

import torch
import pytest
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity.simulation import DCStaticSimulationNodal
from simpegtorch.electromagnetics.resistivity import sources, receivers, survey


def test_dc_simulation_nodal_fields_with_gradients():
    """Test complete nodal DC simulation with field computation and gradients."""

    # Create a simple 3D mesh with explicit cell sizes
    h = torch.ones(10, dtype=torch.float64)  # 10 cells of size 1.0 each
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create resistivity with gradients
    resistivity = torch.full(
        (mesh.n_cells,), 100.0, dtype=torch.float64, requires_grad=True
    )

    # Create receivers for measuring potential differences
    # Place receivers well within the mesh bounds (mesh is [0,10] x [0,10] x [0,10])
    rx_locations_m = torch.tensor(
        [
            [3.0, 3.0, 2.0],  # M electrode
            [4.0, 3.0, 2.0],  # Another M electrode
        ],
        dtype=torch.float64,
    )

    rx_locations_n = torch.tensor(
        [
            [3.0, 4.0, 2.0],  # N electrode
            [4.0, 4.0, 2.0],  # Another N electrode
        ],
        dtype=torch.float64,
    )

    rx = receivers.Dipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create dipole source within mesh bounds
    src_location_a = torch.tensor([2.0, 3.5, 2.0], dtype=torch.float64)  # A electrode
    src_location_b = torch.tensor([6.0, 3.5, 2.0], dtype=torch.float64)  # B electrode
    src = sources.Dipole(
        [rx], location_a=src_location_a, location_b=src_location_b, current=1.0
    )

    # Create survey
    surv = survey.Survey([src])

    # Create nodal DC simulation with survey
    sim = DCStaticSimulationNodal(mesh, survey=surv, bc_type="Neumann")

    # Compute fields - this tests the complete gradient flow through:
    # 1. Source discretization to mesh nodes
    # 2. Edge inner product with inversion
    # 3. System matrix assembly with Neumann BC handling
    # 4. Linear system solve with TorchMatSolver
    # 5. Receiver evaluation from nodal potentials
    fields = sim.fields(resistivity)

    # Verify field computation
    assert isinstance(fields, dict)
    assert src in fields
    field = fields[src]
    assert isinstance(field, torch.Tensor)
    assert field.shape == (mesh.n_nodes,)  # Nodal fields
    assert field.requires_grad  # Gradient tracking enabled

    # Test that we can compute gradients
    loss = torch.sum(field**2)
    loss.backward()

    # Verify gradients were computed
    assert resistivity.grad is not None
    assert resistivity.grad.shape == resistivity.shape
    assert not torch.allclose(resistivity.grad, torch.zeros_like(resistivity.grad))


def test_dc_simulation_nodal_jtvec():
    """Test Jacobian transpose vector product for nodal DC simulation."""

    # Create mesh and model
    h = torch.ones(8, dtype=torch.float64)  # Smaller mesh for efficiency
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    resistivity = torch.full(
        (mesh.n_cells,), 50.0, dtype=torch.float64, requires_grad=True
    )

    # Create simple dipole-dipole survey
    rx_locs_m = torch.tensor([[2.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    rx_locs_n = torch.tensor([[2.0, 3.0, 1.0], [3.0, 3.0, 1.0]], dtype=torch.float64)
    rx = receivers.Dipole(locations_m=rx_locs_m, locations_n=rx_locs_n)

    src_a = torch.tensor([1.5, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([5.5, 2.5, 1.0], dtype=torch.float64)
    src = sources.Dipole([rx], location_a=src_a, location_b=src_b, current=1.0)

    surv = survey.Survey([src])
    sim = DCStaticSimulationNodal(mesh, survey=surv, bc_type="Neumann")

    # Test Jtvec computation
    v = torch.ones(sim.n_data, dtype=torch.float64)
    jtvec_result = sim.Jtvec(resistivity, v)

    # Verify result properties
    assert isinstance(jtvec_result, torch.Tensor)
    assert jtvec_result.shape == resistivity.shape
    assert not torch.allclose(jtvec_result, torch.zeros_like(jtvec_result))

    # Test consistency with manual gradient computation
    resistivity_manual = torch.full(
        (mesh.n_cells,), 50.0, dtype=torch.float64, requires_grad=True
    )
    predicted_data = sim.dpred(resistivity_manual)
    loss = torch.sum(v * predicted_data)
    loss.backward()

    # Should be approximately equal (allowing for numerical differences)
    assert torch.allclose(jtvec_result, resistivity_manual.grad, rtol=1e-4, atol=1e-6)


def test_dc_simulation_nodal_multiple_sources():
    """Test nodal DC simulation with multiple sources."""

    # Create mesh
    h = torch.ones(6, dtype=torch.float64)  # Small mesh for multiple sources test
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    resistivity = torch.full(
        (mesh.n_cells,), 75.0, dtype=torch.float64, requires_grad=True
    )

    # Create multiple sources with different receiver configurations
    sources_list = []

    # Source 1: Dipole in one area
    rx1_m = torch.tensor([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5]], dtype=torch.float64)
    rx1_n = torch.tensor([[1.0, 2.0, 0.5], [2.0, 2.0, 0.5]], dtype=torch.float64)
    rx1 = receivers.Dipole(locations_m=rx1_m, locations_n=rx1_n)

    src1_a = torch.tensor([0.5, 1.5, 0.5], dtype=torch.float64)
    src1_b = torch.tensor([3.5, 1.5, 0.5], dtype=torch.float64)
    src1 = sources.Dipole([rx1], location_a=src1_a, location_b=src1_b, current=1.5)
    sources_list.append(src1)

    # Source 2: Dipole in another area
    rx2_m = torch.tensor([[3.0, 3.0, 1.0], [4.0, 3.0, 1.0]], dtype=torch.float64)
    rx2_n = torch.tensor([[3.0, 4.0, 1.0], [4.0, 4.0, 1.0]], dtype=torch.float64)
    rx2 = receivers.Dipole(locations_m=rx2_m, locations_n=rx2_n)

    src2_a = torch.tensor([2.5, 3.5, 1.0], dtype=torch.float64)
    src2_b = torch.tensor([5.5, 3.5, 1.0], dtype=torch.float64)
    src2 = sources.Dipole([rx2], location_a=src2_a, location_b=src2_b, current=0.8)
    sources_list.append(src2)

    # Create survey and simulation
    surv = survey.Survey(sources_list)
    sim = DCStaticSimulationNodal(mesh, survey=surv, bc_type="Neumann")

    # Test field computation for multiple sources
    fields = sim.fields(resistivity)

    # Verify fields structure
    assert isinstance(fields, dict)
    assert len(fields) == 2
    assert src1 in fields
    assert src2 in fields

    # Check individual field properties
    for src in sources_list:
        field = fields[src]
        assert isinstance(field, torch.Tensor)
        assert field.shape == (mesh.n_nodes,)
        assert field.requires_grad

    # Test data prediction
    predicted_data = sim.dpred(resistivity)
    expected_n_data = sum(src.receiver_list[0].nD for src in sources_list)
    assert predicted_data.shape == (expected_n_data,)
    assert predicted_data.requires_grad

    # Test gradient computation
    loss = torch.sum(predicted_data**2)
    loss.backward()

    assert resistivity.grad is not None
    assert not torch.allclose(resistivity.grad, torch.zeros_like(resistivity.grad))


def test_dc_simulation_nodal_apparent_resistivity():
    """Test nodal DC simulation apparent resistivity calculation."""

    # Create mesh with heterogeneous resistivity
    h = torch.ones(8, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create heterogeneous resistivity model (higher resistivity in center)
    resistivity = torch.full(
        (mesh.n_cells,), 50.0, dtype=torch.float64, requires_grad=True
    )

    # Add high resistivity anomaly in center
    cell_centers = mesh.cell_centers
    center_point = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
    distances = torch.norm(cell_centers - center_point, dim=1)
    anomaly_mask = distances < 2.0
    resistivity.data[anomaly_mask] = 200.0  # High resistivity anomaly

    # Create dipole-dipole survey configuration
    rx_locations_m = torch.tensor(
        [
            [2.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [4.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )

    rx_locations_n = torch.tensor(
        [
            [2.0, 3.0, 1.0],
            [3.0, 3.0, 1.0],
            [4.0, 3.0, 1.0],
        ],
        dtype=torch.float64,
    )

    rx = receivers.Dipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Source dipole
    src_a = torch.tensor([1.0, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([6.0, 2.5, 1.0], dtype=torch.float64)
    src = sources.Dipole([rx], location_a=src_a, location_b=src_b, current=1.0)

    surv = survey.Survey([src])
    sim = DCStaticSimulationNodal(mesh, survey=surv, bc_type="Neumann")

    # Compute predicted data (potential differences)
    predicted_data = sim.dpred(resistivity)

    # Verify data properties
    assert isinstance(predicted_data, torch.Tensor)
    assert predicted_data.shape == (rx.nD,)
    assert predicted_data.requires_grad

    # Test that heterogeneous model produces non-zero data
    assert not torch.allclose(predicted_data, torch.zeros_like(predicted_data))

    # Test apparent resistivity computation (simplified)
    # For dipole-dipole: rho_app = K * (V_MN / I)
    # where K is geometric factor (depends on electrode spacing)
    current = 1.0
    geometric_factor = 2 * torch.pi * 1.0  # Simplified K for unit spacing
    apparent_resistivity = geometric_factor * torch.abs(predicted_data) / current

    # Should be reasonable resistivity values
    assert torch.all(apparent_resistivity > 0)
    assert torch.all(apparent_resistivity < 1000)  # Reasonable upper bound

    # Test gradient computation through apparent resistivity
    loss = torch.sum(apparent_resistivity)
    loss.backward()

    assert resistivity.grad is not None
    assert not torch.allclose(resistivity.grad, torch.zeros_like(resistivity.grad))


def test_dc_simulation_nodal_vs_cell_centered_consistency():
    """Test that nodal and cell-centered simulations give consistent results for simple cases."""

    # Create identical mesh and resistivity model
    h = torch.ones(6, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    resistivity = torch.full((mesh.n_cells,), 100.0, dtype=torch.float64)

    # Create identical source and receiver setup
    rx_m = torch.tensor([[2.0, 2.0, 1.0]], dtype=torch.float64)
    rx_n = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float64)
    rx = receivers.Dipole(locations_m=rx_m, locations_n=rx_n)

    src_a = torch.tensor([1.5, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([4.5, 2.5, 1.0], dtype=torch.float64)
    src = sources.Dipole([rx], location_a=src_a, location_b=src_b, current=1.0)

    surv = survey.Survey([src])

    # Run nodal simulation
    sim_nodal = DCStaticSimulationNodal(mesh, survey=surv, bc_type="Neumann")
    data_nodal = sim_nodal.dpred(resistivity)

    # Import cell-centered simulation
    from simpegtorch.electromagnetics.resistivity.simulation import (
        DCStaticSimulationCellCentered,
    )

    # Run cell-centered simulation
    sim_cc = DCStaticSimulationCellCentered(mesh, survey=surv, bc_type="Dirichlet")
    data_cc = sim_cc.dpred(resistivity)

    # Both should produce similar results for homogeneous medium
    # (Allow for some difference due to different formulations and BCs)
    assert data_nodal.shape == data_cc.shape

    # They shouldn't be identical (different formulations), but should be same order of magnitude
    relative_diff = torch.abs(data_nodal - data_cc) / (torch.abs(data_nodal) + 1e-12)
    assert torch.all(
        relative_diff < 2.0
    )  # Allow up to 200% difference (different formulations)

    # Both should be non-zero for non-trivial geometry
    assert not torch.allclose(data_nodal, torch.zeros_like(data_nodal))
    assert not torch.allclose(data_cc, torch.zeros_like(data_cc))


if __name__ == "__main__":
    pytest.main([__file__])
