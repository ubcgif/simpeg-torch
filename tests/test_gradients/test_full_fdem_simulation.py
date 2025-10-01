"""
Test complete FDEM simulation workflow with gradient computation using new PDE architecture.
"""

import torch
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM import (
    FDEM3DMagneticFluxDensity,
    MagneticDipole,
    RxPointMagneticFluxDensity,
    RxPointElectricField,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings

torch.set_default_dtype(torch.float64)


def test_fdem_simulation_fields_with_gradients():
    """Test complete FDEM simulation with field computation and gradients using new PDE architecture."""

    # Create a simple 3D mesh
    h = torch.ones(8, dtype=torch.float64) * 25.0  # 8 cells of 25m each
    mesh = TensorMesh([h, h, h], dtype=torch.float64)

    # Create conductivity with gradients
    sigma = torch.full(
        (mesh.n_cells,), 0.01, dtype=torch.float64, requires_grad=True
    )

    # Create receivers for measuring magnetic flux density
    # Place receivers within mesh bounds (mesh is [0,200] x [0,200] x [0,200])
    rx_locations = torch.tensor(
        [
            [75.0, 75.0, 50.0],
            [125.0, 75.0, 50.0],
            [75.0, 125.0, 50.0],
        ],
        dtype=torch.float64,
    )

    rx = RxPointMagneticFluxDensity(
        locations=rx_locations, orientation="z", component="real"
    )

    # Create magnetic dipole source
    src_location = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
    src_moment = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)  # Vertical dipole
    frequency = 1.0  # 1 Hz

    src = MagneticDipole([rx], frequency, src_location, moment=src_moment)

    # Create survey
    surv = Survey([src])

    # Create conductivity mapping
    sigma_map = mappings.BaseMapping(sigma)

    # Create FDEM PDE and solver
    pde = FDEM3DMagneticFluxDensity(mesh, surv, sigma_map)
    solver = DirectSolver(pde)

    # Compute predicted data
    predicted_data = solver.forward()

    # Compute a simple objective function based on data
    loss = torch.sum(torch.abs(predicted_data) ** 2)

    # Compute gradients
    loss.backward()

    # Verify results
    assert predicted_data is not None, "Predicted data should be computed"
    assert (
        predicted_data.shape[0] == rx.locations.shape[0]
    ), "Data should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"

    assert (
        sigma_map.trainable_parameters.grad is not None
    ), "Gradients should be computed"
    assert torch.all(
        torch.isfinite(sigma_map.trainable_parameters.grad)
    ), "Gradients should be finite"
    assert torch.any(
        sigma_map.trainable_parameters.grad != 0
    ), "Some gradients should be non-zero"

    print("✅ Complete FDEM simulation test with new PDE architecture passed")
    print(f"Mesh cells: {mesh.n_cells}")
    print(f"Source: Magnetic dipole at {src.location}, frequency {src.frequency} Hz")
    print(f"Receivers: {rx.locations.shape[0]} point measurements")
    print(f"Data shape: {predicted_data.shape}")
    print(f"Data range: [{predicted_data.min():.6e}, {predicted_data.max():.6e}] T")
    print(f"Gradient mean: {sigma_map.trainable_parameters.grad.mean():.2e}")
    print(f"Gradient std: {sigma_map.trainable_parameters.grad.std():.2e}")


def test_fdem_simulation_multiple_frequencies():
    """Test FDEM simulation with multiple frequencies."""

    # Create mesh and model
    h = torch.ones(6, dtype=torch.float64) * 30.0
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full((mesh.n_cells,), 0.02, dtype=torch.float64, requires_grad=True)

    # Create receivers within mesh bounds [0,180] x [0,180] x [0,180]
    rx_locations = torch.tensor(
        [
            [60.0, 60.0, 30.0],
            [90.0, 60.0, 30.0],
            [120.0, 60.0, 30.0],
        ],
        dtype=torch.float64,
    )

    rx_bz = RxPointMagneticFluxDensity(
        locations=rx_locations, orientation="z", component="real"
    )

    # Create sources at multiple frequencies
    src_location = torch.tensor([90.0, 90.0, 90.0], dtype=torch.float64)
    src_moment = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    frequencies = [0.5, 1.0, 2.0, 5.0]  # Multiple frequencies
    sources = []
    for freq in frequencies:
        src = MagneticDipole([rx_bz], freq, src_location, moment=src_moment)
        sources.append(src)

    # Create survey and simulation
    surv = Survey(sources)
    sigma_map = mappings.BaseMapping(sigma)
    pde = FDEM3DMagneticFluxDensity(mesh, surv, sigma_map)
    solver = DirectSolver(pde)

    # Test forward modeling
    predicted_data = solver.forward()

    # Test Jacobian transpose with multiple frequencies
    data_residuals = torch.randn_like(predicted_data)
    loss = torch.sum(data_residuals * predicted_data.real)
    loss.backward()
    gradient = sigma_map.trainable_parameters.grad

    # Verify results
    expected_data_count = rx_locations.shape[0] * len(frequencies)
    assert (
        predicted_data.shape[0] == expected_data_count
    ), f"Expected {expected_data_count} data points"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Multiple frequencies test with new PDE architecture passed")
    print(f"Frequencies: {surv.frequencies} Hz")
    print(f"Total data points: {predicted_data.shape[0]}")
    print(f"Data range: [{predicted_data.abs().min():.6e}, {predicted_data.abs().max():.6e}] T")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_fdem_simulation_multiple_sources():
    """Test simulation with multiple sources at same frequency using new PDE architecture."""

    # Create mesh
    h = torch.ones(8, dtype=torch.float64) * 20.0
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full(
        (mesh.n_cells,), 1.0 / 100.0, dtype=torch.float64, requires_grad=True
    )

    # Create receivers
    rx_locations = torch.tensor(
        [
            [60.0, 60.0, 40.0],
            [80.0, 60.0, 40.0],
            [100.0, 60.0, 40.0],
        ],
        dtype=torch.float64,
    )

    rx = RxPointMagneticFluxDensity(
        locations=rx_locations, orientation="z", component="real"
    )

    # Create multiple sources at same frequency
    frequency = 1.0
    src_moment = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    src1 = MagneticDipole(
        [rx], frequency, torch.tensor([40.0, 80.0, 80.0], dtype=torch.float64), moment=src_moment
    )

    src2 = MagneticDipole(
        [rx], frequency, torch.tensor([120.0, 80.0, 80.0], dtype=torch.float64), moment=src_moment
    )

    # Also test electric field receiver
    rx_e = RxPointElectricField(
        locations=torch.tensor([[80.0, 80.0, 60.0]], dtype=torch.float64),
        orientation="x",
        component="real",
    )

    src3 = MagneticDipole(
        [rx_e], frequency, torch.tensor([80.0, 40.0, 80.0], dtype=torch.float64), moment=src_moment
    )

    # Create survey with multiple sources
    surv = Survey([src1, src2, src3])
    sigma_map = mappings.BaseMapping(sigma)
    pde = FDEM3DMagneticFluxDensity(mesh, surv, sigma_map)
    solver = DirectSolver(pde)

    # Test forward modeling
    predicted_data = solver.forward()

    # Test Jacobian transpose with multiple sources
    data_residuals = torch.randn_like(predicted_data.real)
    loss = torch.sum(data_residuals * predicted_data.real)
    loss.backward()
    gradient = sigma_map.trainable_parameters.grad

    # Verify results
    expected_data_count = rx.locations.shape[0] * 2 + rx_e.locations.shape[
        0
    ]  # 2 B sources + 1 E source
    assert (
        predicted_data.shape[0] == expected_data_count
    ), f"Expected {expected_data_count} data points"
    assert gradient.shape[0] == mesh.n_cells, "Gradient should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Multiple sources test with new PDE architecture passed")
    print(f"Sources: {len(surv.source_list)} at frequency {frequency} Hz")
    print(f"Total data points: {predicted_data.shape[0]}")
    print(f"Data range: [{predicted_data.abs().min():.6e}, {predicted_data.abs().max():.6e}]")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


def test_fdem_simulation_with_permittivity():
    """Test FDEM simulation including displacement currents (permittivity)."""

    # Create mesh
    h = torch.ones(6, dtype=torch.float64) * 25.0
    mesh = TensorMesh([h, h, h], dtype=torch.float64)
    sigma = torch.full((mesh.n_cells,), 0.01, dtype=torch.float64, requires_grad=True)
    permittivity = torch.ones(mesh.n_cells, dtype=torch.float64) * 8.854e-12  # Vacuum

    # Create receiver
    rx_locations = torch.tensor(
        [
            [75.0, 75.0, 50.0],
            [100.0, 75.0, 50.0],
        ],
        dtype=torch.float64,
    )

    rx = RxPointMagneticFluxDensity(
        locations=rx_locations, orientation="z", component="real"
    )

    # Create source at higher frequency (where displacement current matters)
    src_location = torch.tensor([75.0, 75.0, 75.0], dtype=torch.float64)
    src_moment = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    frequency = 1e6  # 1 MHz - higher frequency

    src = MagneticDipole([rx], frequency, src_location, moment=src_moment)

    # Create survey and simulation with permittivity
    surv = Survey([src])
    sigma_map = mappings.BaseMapping(sigma)
    pde = FDEM3DMagneticFluxDensity(mesh, surv, sigma_map, permittivity=permittivity)
    solver = DirectSolver(pde)

    # Test forward modeling
    predicted_data = solver.forward()

    # Test gradients
    loss = torch.sum(torch.abs(predicted_data) ** 2)
    loss.backward()
    gradient = sigma_map.trainable_parameters.grad

    # Verify results
    assert predicted_data.shape[0] == rx.locations.shape[0], "Data should have correct shape"
    assert torch.all(torch.isfinite(predicted_data)), "Data should be finite"
    assert torch.all(torch.isfinite(gradient)), "Gradient should be finite"

    print("✅ Permittivity test with new PDE architecture passed")
    print(f"Frequency: {frequency:.2e} Hz")
    print(f"Data range: [{predicted_data.abs().min():.6e}, {predicted_data.abs().max():.6e}] T")
    print(f"Gradient range: [{gradient.min():.2e}, {gradient.max():.2e}]")


if __name__ == "__main__":
    print("🧪 Testing Full FDEM Simulation with New PDE Architecture")
    print("=" * 65)

    test_fdem_simulation_fields_with_gradients()
    print()

    test_fdem_simulation_multiple_frequencies()
    print()

    test_fdem_simulation_multiple_sources()
    print()

    test_fdem_simulation_with_permittivity()
    print()

    print("=" * 65)
    print("🎉 All FDEM simulation tests with new PDE architecture passed!")
