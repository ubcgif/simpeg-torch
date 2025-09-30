"""
Test complete DC simulation workflow with gradient computation using new source/receiver classes.
"""

import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.resistivity import (
    DC3DCellCentered,
    SrcDipole,
    SrcPole,
    RxDipole,
    RxPole,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings

torch.set_default_dtype(torch.float64)


def test_dc_simulation_fields_with_gradients():
    """Test complete DC simulation with field computation and gradients using new PDE architecture."""

    # Create a simple 3D mesh with explicit cell sizes
    h = torch.ones(10, dtype=torch.float64)  # 10 cells of size 1.0 each for more room
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create conductivity with gradients (note: using conductivity as base parameter)
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

    # Create resistivity mapping (resistivity = 1/sigma)
    resistivity_map = mappings.InverseMapping(sigma)

    # Create DC PDE and solver
    pde = DC3DCellCentered(mesh, surv, resistivity_map, bc_type="Dirichlet")
    solver = DirectSolver(pde)

    # Compute predicted data - this tests the complete gradient flow through:
    # 1. Source discretization to mesh
    # 2. Face inner product with inversion
    # 3. System matrix assembly (D @ MfRhoI @ G)
    # 4. Linear system solve with TorchMatSolver
    # 5. Receiver evaluation from fields
    predicted_data = solver.forward()

    # Compute a simple objective function based on data
    loss = torch.sum(predicted_data**2)

    # Compute gradients
    loss.backward()

    # Verify results
    assert predicted_data is not None, "Predicted data should be computed"
    assert predicted_data.shape[0] == rx.locations_m.shape[0], "Data should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"

    assert resistivity_map.trainable_parameters.grad is not None, "Gradients should be computed"
    assert torch.all(torch.isfinite(resistivity_map.trainable_parameters.grad)), "Gradients should be finite"
    assert torch.any(resistivity_map.trainable_parameters.grad != 0), "Some gradients should be non-zero"

    print("✅ Complete DC simulation test with new PDE architecture passed")
    print(f"Mesh cells: {mesh.n_cells}")
    print(f"Source: Dipole at A={src.location_a}, B={src.location_b}")
    print(f"Receivers: {rx.locations_m.shape[0]} dipole measurements")
    print(f"Data shape: {predicted_data.shape}")
    print(f"Data range: [{predicted_data.min():.6f}, {predicted_data.max():.6f}] V")
    print(f"Gradient mean: {resistivity_map.trainable_parameters.grad.mean():.2e}")
    print(f"Gradient std: {resistivity_map.trainable_parameters.grad.std():.2e}")


def test_dc_simulation_jtvec():
    """Test Jacobian transpose vector product functionality with new PDE architecture."""

    # Create mesh and model
    h = torch.ones(6, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full(
        (mesh.n_cells,), 0.02, dtype=torch.float64, requires_grad=True
    )

    # Create multiple receivers within mesh bounds [0,6] x [0,6] x [0,6]
    rx_locations_m = torch.tensor(
        [
            [2.0, 2.0, 0.5],
            [3.0, 2.0, 0.5],
            [4.0, 2.0, 0.5],
        ],
        dtype=torch.float64,
    )

    rx_locations_n = torch.tensor(
        [
            [2.0, 3.0, 0.5],
            [3.0, 3.0, 0.5],
            [4.0, 3.0, 0.5],
        ],
        dtype=torch.float64,
    )

    rx = RxDipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create source within mesh bounds
    src = SrcDipole(
        [rx],
        torch.tensor([1.0, 2.5, 0.5], dtype=torch.float64),
        torch.tensor([5.0, 2.5, 0.5], dtype=torch.float64),
        current=2.0,
    )

    # Create survey, mapping and solver
    surv = Survey([src])
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DCellCentered(mesh, surv, resistivity_map, bc_type="Dirichlet")
    solver = DirectSolver(pde)

    # Test Jtvec by computing gradients manually
    data_residuals = torch.randn(rx.locations_m.shape[0], dtype=torch.float64)
    
    # Forward pass
    predicted_data = solver.forward()
    
    # Compute loss using data residuals (Jtvec equivalent)
    loss = torch.sum(data_residuals * predicted_data)
    loss.backward()
    
    gradient = resistivity_map.trainable_parameters.grad

    # Verify results
    assert gradient is not None, "Gradients should be computed"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Jacobian transpose test with new PDE architecture passed")
    print(f"Data residuals shape: {data_residuals.shape}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_dc_simulation_multiple_sources():
    """Test simulation with multiple sources using new PDE architecture."""

    # Create mesh
    h = torch.ones(8, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full(
        (mesh.n_cells,), 1.0/75.0, dtype=torch.float64, requires_grad=True
    )

    # Create shared receivers
    rx_locations_m = torch.tensor(
        [
            [3.0, 3.0, 1.0],
            [4.0, 3.0, 1.0],
            [5.0, 3.0, 1.0],
        ],
        dtype=torch.float64,
    )

    rx_locations_n = torch.tensor(
        [
            [3.0, 4.0, 1.0],
            [4.0, 4.0, 1.0],
            [5.0, 4.0, 1.0],
        ],
        dtype=torch.float64,
    )

    rx = RxDipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create multiple sources
    src1 = SrcDipole(
        [rx],
        torch.tensor([1.0, 3.5, 1.0], dtype=torch.float64),
        torch.tensor([2.0, 3.5, 1.0], dtype=torch.float64),
    )

    src2 = SrcDipole(
        [rx],
        torch.tensor([6.0, 3.5, 1.0], dtype=torch.float64),
        torch.tensor([7.0, 3.5, 1.0], dtype=torch.float64),
    )

    # Test pole source too
    rx_pole = RxPole(
        locations=torch.tensor([[4.0, 3.5, 2.0]], dtype=torch.float64)
    )
    src3 = SrcPole(
        [rx_pole], torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64)
    )

    # Create survey with multiple sources
    surv = Survey([src1, src2, src3])
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DCellCentered(mesh, surv, resistivity_map, bc_type="Dirichlet")
    solver = DirectSolver(pde)

    # Test forward modeling
    predicted_data = solver.forward()

    # Test Jacobian transpose with multiple sources
    data_residuals = torch.randn_like(predicted_data)
    loss = torch.sum(data_residuals * predicted_data)
    loss.backward()
    gradient = resistivity_map.trainable_parameters.grad

    # Verify results
    expected_data_count = rx.locations_m.shape[0] * 2 + rx_pole.locations.shape[0]  # 2 dipole sources + 1 pole source
    assert (
        predicted_data.shape[0] == expected_data_count
    ), f"Expected {expected_data_count} data points"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Multiple sources test with new PDE architecture passed")
    print(f"Sources: {len(surv.source_list)} (2 dipole + 1 pole)")
    print(f"Total data points: {predicted_data.shape[0]}")
    print(f"Data range: [{predicted_data.min():.6f}, {predicted_data.max():.6f}] V")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_dc_simulation_apparent_resistivity():
    """Test apparent resistivity calculations with new PDE architecture."""

    # Create mesh
    h = torch.ones(10, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full((mesh.n_cells,), 0.01, dtype=torch.float64)  # 1/100 S/m

    # Create receivers for potential differences within mesh bounds [0,10] x [0,10] x [0,10]
    # Use a standard dipole-dipole configuration for proper geometric factors
    rx_locations_m = torch.tensor(
        [
            [3.0, 5.0, 1.0],  # M1 electrode
            [4.0, 5.0, 1.0],  # M2 electrode
        ],
        dtype=torch.float64,
    )

    rx_locations_n = torch.tensor(
        [
            [5.0, 5.0, 1.0],  # N1 electrode
            [6.0, 5.0, 1.0],  # N2 electrode
        ],
        dtype=torch.float64,
    )

    rx = RxDipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create source within mesh bounds - standard dipole-dipole array
    src = SrcDipole(
        [rx],
        torch.tensor([1.0, 5.0, 1.0], dtype=torch.float64),  # A electrode
        torch.tensor([2.0, 5.0, 1.0], dtype=torch.float64),  # B electrode
    )

    # Create survey and simulation
    surv = Survey([src])
    resistivity_map = mappings.InverseMapping(sigma)
    pde = DC3DCellCentered(mesh, surv, resistivity_map, bc_type="Dirichlet")
    solver = DirectSolver(pde)

    # Test potential difference calculation
    predicted_data = solver.forward()

    # Calculate apparent resistivity manually using simple geometric factor
    # For dipole-dipole: rho_app = K * (V_MN / I) where K is geometric factor
    current = 1.0  # Default current
    # Simplified geometric factor for dipole-dipole (depends on electrode spacing)
    electrode_spacing = 1.0  # Unit spacing
    geometric_factor = 2 * torch.pi * electrode_spacing  # Simplified K
    apparent_resistivity = geometric_factor * torch.abs(predicted_data) / current

    # Debug output
    print(f"Potential differences: {predicted_data}")
    print(f"Apparent resistivity values: {apparent_resistivity}")

    # Verify results
    assert torch.all(torch.isfinite(predicted_data))
    assert torch.all(torch.isfinite(apparent_resistivity))
    assert torch.all(apparent_resistivity > 0)

    print("✅ Apparent resistivity test with new PDE architecture passed")
    print(f"Geometric factor: {geometric_factor:.6f}")
    print(f"Potential difference range: [{predicted_data.min():.6e}, {predicted_data.max():.6e}] V")
    print(f"Apparent resistivity range: [{apparent_resistivity.min():.1f}, {apparent_resistivity.max():.1f}] Ω⋅m")


if __name__ == "__main__":
    print("🧪 Testing Full DC Simulation with New PDE Architecture")
    print("=" * 65)

    test_dc_simulation_fields_with_gradients()
    print()

    test_dc_simulation_jtvec()
    print()

    test_dc_simulation_multiple_sources()
    print()

    test_dc_simulation_apparent_resistivity()
    print()

    print("=" * 65)
    print("🎉 All DC simulation tests with new PDE architecture passed!")
