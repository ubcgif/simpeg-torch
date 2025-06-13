"""
Test complete DC simulation workflow with gradient computation using new source/receiver classes.
"""

import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity.simulation import (
    DCStaticSimulationCellCentered,
)
from simpegtorch.electromagnetics.resistivity import sources, receivers, survey

torch.set_default_dtype(torch.float64)


def test_dc_simulation_fields_with_gradients():
    """Test complete DC simulation with field computation and gradients using new source/receiver classes."""

    # Create a simple 3D mesh with explicit cell sizes
    h = torch.ones(10, dtype=torch.float64)  # 10 cells of size 1.0 each for more room
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

    # Create DC simulation with survey
    sim = DCStaticSimulationCellCentered(mesh, survey=surv)
    sim.setBC()

    # Compute fields - this tests the complete gradient flow through:
    # 1. Source discretization to mesh
    # 2. Face inner product with inversion
    # 3. System matrix assembly (D @ MfRhoI @ G)
    # 4. Linear system solve with TorchMatSolver
    # 5. Receiver evaluation from fields
    fields = sim.fields(resistivity)

    # Compute predicted data
    predicted_data = sim.dpred(resistivity)

    # Compute a simple objective function based on data
    loss = torch.sum(predicted_data**2)

    # Compute gradients
    loss.backward()

    # Verify results
    assert isinstance(
        fields, dict
    ), "Fields should be a dictionary for multiple sources"
    assert src in fields, "Fields should contain entry for source"

    src_fields = fields[src]
    assert src_fields is not None, "Fields should be computed"
    assert src_fields.shape[0] == mesh.n_cells, "Fields should have correct shape"
    assert torch.all(torch.isfinite(src_fields)), "Fields should be finite"

    assert predicted_data is not None, "Predicted data should be computed"
    assert predicted_data.shape[0] == rx.nD, "Data should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"

    assert resistivity.grad is not None, "Gradients should be computed"
    assert torch.all(torch.isfinite(resistivity.grad)), "Gradients should be finite"
    assert torch.any(resistivity.grad != 0), "Some gradients should be non-zero"

    print("✅ Complete DC simulation test with new source/receiver classes passed")
    print(f"Mesh cells: {mesh.n_cells}")
    print(f"Source: Dipole at A={src.location_a}, B={src.location_b}")
    print(f"Receivers: {rx.nD} dipole measurements")
    print(f"Fields shape: {src_fields.shape}")
    print(f"Fields range: [{src_fields.min():.6f}, {src_fields.max():.6f}]")
    print(f"Data shape: {predicted_data.shape}")
    print(f"Data range: [{predicted_data.min():.6f}, {predicted_data.max():.6f}] V")
    print(f"Gradient mean: {resistivity.grad.mean():.2e}")
    print(f"Gradient std: {resistivity.grad.std():.2e}")


def test_dc_simulation_jtvec():
    """Test Jtvec functionality with new source/receiver classes."""

    # Create mesh and model
    h = torch.ones(6, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    resistivity = torch.full(
        (mesh.n_cells,), 50.0, dtype=torch.float64, requires_grad=True
    )

    # Create multiple receivers within mesh bounds [0,6] x [0,6] x [0,4]
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

    rx = receivers.Dipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create source within mesh bounds
    src = sources.Dipole(
        [rx],
        location_a=torch.tensor([1.0, 2.5, 0.5], dtype=torch.float64),
        location_b=torch.tensor([5.0, 2.5, 0.5], dtype=torch.float64),
        current=2.0,
    )

    # Create survey and simulation
    surv = survey.Survey([src])
    sim = DCStaticSimulationCellCentered(mesh, survey=surv)
    sim.setBC()

    # Test Jtvec
    data_residuals = torch.randn(rx.nD, dtype=torch.float64)
    gradient = sim.Jtvec(resistivity, data_residuals)

    # Verify results
    assert gradient is not None, "Jtvec should return gradient"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Jtvec test with new source/receiver classes passed")
    print(f"Data residuals shape: {data_residuals.shape}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_dc_simulation_multiple_sources():
    """Test simulation with multiple sources using new classes."""

    # Create mesh
    h = torch.ones(8, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    resistivity = torch.full(
        (mesh.n_cells,), 75.0, dtype=torch.float64, requires_grad=True
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

    rx = receivers.Dipole(locations_m=rx_locations_m, locations_n=rx_locations_n)

    # Create multiple sources
    src1 = sources.Dipole(
        [rx],
        location_a=torch.tensor([1.0, 3.5, 1.0], dtype=torch.float64),
        location_b=torch.tensor([2.0, 3.5, 1.0], dtype=torch.float64),
    )

    src2 = sources.Dipole(
        [rx],
        location_a=torch.tensor([6.0, 3.5, 1.0], dtype=torch.float64),
        location_b=torch.tensor([7.0, 3.5, 1.0], dtype=torch.float64),
    )

    # Test pole source too
    rx_pole = receivers.Pole(
        locations=torch.tensor([[4.0, 3.5, 2.0]], dtype=torch.float64)
    )
    src3 = sources.Pole(
        [rx_pole], location=torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64)
    )

    # Create survey with multiple sources
    surv = survey.Survey([src1, src2, src3])
    sim = DCStaticSimulationCellCentered(mesh, survey=surv)
    sim.setBC()

    # Test forward modeling
    predicted_data = sim.dpred(resistivity)

    # Test Jtvec with multiple sources
    data_residuals = torch.randn_like(predicted_data)
    gradient = sim.Jtvec(resistivity, data_residuals)

    # Verify results
    expected_data_count = rx.nD * 2 + rx_pole.nD  # 2 dipole sources + 1 pole source
    assert (
        predicted_data.shape[0] == expected_data_count
    ), f"Expected {expected_data_count} data points"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Multiple sources test with new source/receiver classes passed")
    print(f"Sources: {surv.nSrc} (2 dipole + 1 pole)")
    print(f"Total data points: {surv.nD}")
    print(f"Data range: [{predicted_data.min():.6f}, {predicted_data.max():.6f}] V")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_dc_simulation_apparent_resistivity():
    """Test apparent resistivity calculations."""

    # Create mesh
    h = torch.ones(10, dtype=torch.float64)
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    resistivity = torch.full((mesh.n_cells,), 100.0, dtype=torch.float64)

    # Create receivers with apparent resistivity data type within mesh bounds [0,10] x [0,10] x [0,10]
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

    rx = receivers.Dipole(
        locations_m=rx_locations_m,
        locations_n=rx_locations_n,
        data_type="apparent_resistivity",
    )

    # Create source within mesh bounds - standard dipole-dipole array
    src = sources.Dipole(
        [rx],
        location_a=torch.tensor([1.0, 5.0, 1.0], dtype=torch.float64),  # A electrode
        location_b=torch.tensor([2.0, 5.0, 1.0], dtype=torch.float64),  # B electrode
    )

    # Create survey and set geometric factors
    surv = survey.Survey([src])
    geometric_factors = surv.set_geometric_factor(space_type="halfspace")

    # Create simulation
    sim = DCStaticSimulationCellCentered(mesh, survey=surv)
    sim.setBC()

    # Test apparent resistivity calculation
    apparent_resistivity = sim.dpred(resistivity)

    # Debug output
    print(f"Apparent resistivity values: {apparent_resistivity}")
    print(f"Geometric factors: {geometric_factors}")

    # Verify results
    assert torch.all(torch.isfinite(apparent_resistivity))

    print("✅ Apparent resistivity test with new source/receiver classes passed")
    print(
        f"Geometric factors range: [{geometric_factors.min():.6f}, {geometric_factors.max():.6f}]"
    )
    print(
        f"Apparent resistivity range: [{apparent_resistivity.min():.1f}, {apparent_resistivity.max():.1f}] Ω⋅m"
    )


if __name__ == "__main__":
    print("🧪 Testing Full DC Simulation with New Source/Receiver Classes")
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
    print("🎉 All DC simulation tests with new source/receiver classes passed!")
