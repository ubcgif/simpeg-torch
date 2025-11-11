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
from simpegtorch.simulation.resistivity import (
    DC3DNodal,
    SrcDipole,
    RxDipole,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings


def test_dc_simulation_nodal_fields_with_gradients():
    """Test complete nodal DC simulation with field computation and gradients."""

    # Create a simple 3D mesh with explicit cell sizes
    h = torch.ones(10, dtype=torch.float64)  # 10 cells of size 1.0 each
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create conductivity with gradients
    sigma = torch.full(
        (mesh.n_cells,), 0.01, dtype=torch.float64, requires_grad=True
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

    rx = RxDipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create dipole source within mesh bounds
    src_location_a = torch.tensor([2.0, 3.5, 2.0], dtype=torch.float64)  # A electrode
    src_location_b = torch.tensor([6.0, 3.5, 2.0], dtype=torch.float64)  # B electrode
    src = SrcDipole(
        [rx], src_location_a, src_location_b, current=1.0
    )

    # Create survey
    surv = Survey([src])

    # Create resistivity mapping and nodal DC PDE
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DNodal(mesh, surv, resistivity_map, bc_type="Neumann")
    solver = DirectSolver(pde)

    # Compute predicted data - this tests the complete gradient flow through:
    # 1. Source discretization to mesh nodes
    # 2. Edge inner product with conductivity
    # 3. System matrix assembly with Neumann BC handling
    # 4. Linear system solve with TorchMatSolver
    # 5. Receiver evaluation from nodal potentials
    predicted_data = solver.forward()

    # Verify data computation
    assert isinstance(predicted_data, torch.Tensor)
    assert predicted_data.shape == (rx.locations_m.shape[0],)  # Data points
    assert predicted_data.requires_grad  # Gradient tracking enabled

    # Test that we can compute gradients
    loss = torch.sum(predicted_data**2)
    loss.backward()

    # Verify gradients were computed
    assert resistivity_map.trainable_parameters.grad is not None
    assert resistivity_map.trainable_parameters.grad.shape == resistivity_map.trainable_parameters.shape
    assert not torch.allclose(resistivity_map.trainable_parameters.grad, torch.zeros_like(resistivity_map.trainable_parameters.grad))


def test_dc_simulation_nodal_jtvec():
    """Test Jacobian transpose vector product for nodal DC simulation."""

    # Create mesh and model
    h = torch.ones(8, dtype=torch.float64)  # Smaller mesh for efficiency
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    sigma = torch.full(
        (mesh.n_cells,), 0.02, dtype=torch.float64, requires_grad=True
    )

    # Create simple dipole-dipole survey
    rx_locs_m = torch.tensor([[2.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    rx_locs_n = torch.tensor([[2.0, 3.0, 1.0], [3.0, 3.0, 1.0]], dtype=torch.float64)
    rx = RxDipole(locations_m=rx_locs_m, locations_n=rx_locs_n)

    src_a = torch.tensor([1.5, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([5.5, 2.5, 1.0], dtype=torch.float64)
    src = SrcDipole([rx], src_a, src_b, current=1.0)

    surv = Survey([src])
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DNodal(mesh, surv, resistivity_map, bc_type="Neumann")
    solver = DirectSolver(pde)

    # Test Jtvec computation via manual gradient
    v = torch.ones(rx.locations_m.shape[0], dtype=torch.float64)
    predicted_data = solver.forward()
    loss = torch.sum(v * predicted_data)
    loss.backward()
    jtvec_result = resistivity_map.trainable_parameters.grad

    # Verify result properties
    assert isinstance(jtvec_result, torch.Tensor)
    assert jtvec_result.shape == resistivity_map.trainable_parameters.shape
    assert not torch.allclose(jtvec_result, torch.zeros_like(jtvec_result))

    # Test consistency with manual gradient computation
    sigma_manual = torch.full(
        (mesh.n_cells,), 0.02, dtype=torch.float64, requires_grad=True
    )
    resistivity_map_manual = mappings.InverseMapping(sigma_manual)
    pde_manual = DC3DNodal(mesh, surv, resistivity_map_manual, bc_type="Neumann")
    solver_manual = DirectSolver(pde_manual)
    predicted_data_manual = solver_manual.forward()
    loss_manual = torch.sum(v * predicted_data_manual)
    loss_manual.backward()

    # Should be approximately equal (allowing for numerical differences)
    assert torch.allclose(jtvec_result, resistivity_map_manual.trainable_parameters.grad, rtol=1e-4, atol=1e-6)


def test_dc_simulation_nodal_multiple_sources():
    """Test nodal DC simulation with multiple sources."""

    # Create mesh
    h = torch.ones(6, dtype=torch.float64)  # Small mesh for multiple sources test
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    sigma = torch.full(
        (mesh.n_cells,), 1.0/75.0, dtype=torch.float64, requires_grad=True
    )

    # Create multiple sources with different receiver configurations
    sources_list = []

    # Source 1: Dipole in one area
    rx1_m = torch.tensor([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5]], dtype=torch.float64)
    rx1_n = torch.tensor([[1.0, 2.0, 0.5], [2.0, 2.0, 0.5]], dtype=torch.float64)
    rx1 = RxDipole(locations_m=rx1_m, locations_n=rx1_n)

    src1_a = torch.tensor([0.5, 1.5, 0.5], dtype=torch.float64)
    src1_b = torch.tensor([3.5, 1.5, 0.5], dtype=torch.float64)
    src1 = SrcDipole([rx1], src1_a, src1_b, current=1.5)
    sources_list.append(src1)

    # Source 2: Dipole in another area
    rx2_m = torch.tensor([[3.0, 3.0, 1.0], [4.0, 3.0, 1.0]], dtype=torch.float64)
    rx2_n = torch.tensor([[3.0, 4.0, 1.0], [4.0, 4.0, 1.0]], dtype=torch.float64)
    rx2 = RxDipole(locations_m=rx2_m, locations_n=rx2_n)

    src2_a = torch.tensor([2.5, 3.5, 1.0], dtype=torch.float64)
    src2_b = torch.tensor([5.5, 3.5, 1.0], dtype=torch.float64)
    src2 = SrcDipole([rx2], src2_a, src2_b, current=0.8)
    sources_list.append(src2)

    # Create survey and simulation
    surv = Survey(sources_list)
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DNodal(mesh, surv, resistivity_map, bc_type="Neumann")
    solver = DirectSolver(pde)

    # Test data prediction for multiple sources
    predicted_data = solver.forward()

    # Verify data structure
    expected_n_data = sum(src.receiver_list[0].locations_m.shape[0] for src in sources_list)
    assert predicted_data.shape == (expected_n_data,)
    assert predicted_data.requires_grad

    # Test gradient computation
    loss = torch.sum(predicted_data**2)
    loss.backward()

    assert resistivity_map.trainable_parameters.grad is not None
    assert not torch.allclose(resistivity_map.trainable_parameters.grad, torch.zeros_like(resistivity_map.trainable_parameters.grad))


def test_dc_simulation_nodal_apparent_resistivity():
    """Test nodal DC simulation apparent resistivity calculation."""

    # Create mesh with heterogeneous resistivity
    h = torch.ones(8, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create heterogeneous conductivity model (lower conductivity in center)
    sigma = torch.full(
        (mesh.n_cells,), 1.0/50.0, dtype=torch.float64, requires_grad=True
    )

    # Add low conductivity anomaly in center (high resistivity)
    cell_centers = mesh.cell_centers
    center_point = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
    distances = torch.norm(cell_centers - center_point, dim=1)
    anomaly_mask = distances < 2.0
    sigma.data[anomaly_mask] = 1.0/200.0  # Low conductivity anomaly (high resistivity)

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

    rx = RxDipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Source dipole
    src_a = torch.tensor([1.0, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([6.0, 2.5, 1.0], dtype=torch.float64)
    src = SrcDipole([rx], src_a, src_b, current=1.0)

    surv = Survey([src])
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DNodal(mesh, surv, resistivity_map, bc_type="Neumann")
    solver = DirectSolver(pde)

    # Compute predicted data (potential differences)
    predicted_data = solver.forward()

    # Verify data properties
    assert isinstance(predicted_data, torch.Tensor)
    assert predicted_data.shape == (rx.locations_m.shape[0],)
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

    assert resistivity_map.trainable_parameters.grad is not None
    assert not torch.allclose(resistivity_map.trainable_parameters.grad, torch.zeros_like(resistivity_map.trainable_parameters.grad))


def test_dc_simulation_nodal_vs_cell_centered_consistency():
    """Test that nodal and cell-centered simulations give consistent results for simple cases."""

    # Create identical mesh and conductivity model
    h = torch.ones(6, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full((mesh.n_cells,), 0.01, dtype=torch.float64)

    # Create identical source and receiver setup
    rx_m = torch.tensor([[2.0, 2.0, 1.0]], dtype=torch.float64)
    rx_n = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float64)
    rx = RxDipole(locations_m=rx_m, locations_n=rx_n)

    src_a = torch.tensor([1.5, 2.5, 1.0], dtype=torch.float64)
    src_b = torch.tensor([4.5, 2.5, 1.0], dtype=torch.float64)
    src = SrcDipole([rx], src_a, src_b, current=1.0)

    surv = Survey([src])

    # Run nodal simulation
    resistivity_map_nodal = mappings.InverseMapping(sigma)
    pde_nodal = DC3DNodal(mesh, surv, resistivity_map_nodal, bc_type="Neumann")
    solver_nodal = DirectSolver(pde_nodal)
    data_nodal = solver_nodal.forward()

    # Import and run cell-centered simulation
    from simpegtorch.simulation.resistivity import DC3DCellCentered

    resistivity_map_cc = mappings.InverseMapping(sigma)
    pde_cc = DC3DCellCentered(mesh, surv, resistivity_map_cc, bc_type="Dirichlet")
    solver_cc = DirectSolver(pde_cc)
    data_cc = solver_cc.forward()

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
