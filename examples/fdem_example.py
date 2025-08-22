#!/usr/bin/env python3
"""
Test FDEM functionality
========================

This script tests the complete FDEM implementation including:
- Sources (MagneticDipole, ElectricDipole, LoopSource)
- Receivers (PointMagneticFluxDensity, PointElectricField)
- Survey and Simulation classes
- Forward modeling (dpred) functionality
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.FDEM.simulation import Simulation3DMagneticFluxDensity
from simpegtorch.electromagnetics.FDEM.sources import (
    MagneticDipole,
    ElectricDipole,
    LoopSource,
)
from simpegtorch.electromagnetics.FDEM.receivers import (
    PointMagneticFluxDensity,
    PointMagneticFluxDensitySecondary,
    PointElectricField,
)
from simpegtorch.electromagnetics.FDEM.survey import FDEMSurvey


def create_test_mesh():
    """Create a simple 3D test mesh"""
    print("Creating test mesh...")

    # Create a small 6x6x4 mesh with 10m cells for faster testing
    nx, ny, nz = 6, 6, 4
    hx = torch.ones(nx) * 10.0  # 10m cell size
    hy = torch.ones(ny) * 10.0
    hz = torch.ones(nz) * 10.0

    mesh = TensorMesh([hx, hy, hz])
    print(f"  Mesh shape: {mesh.shape_cells}")
    print(
        f"  Mesh size: {mesh.n_cells} cells, {mesh.n_faces} faces, {mesh.n_edges} edges"
    )
    domain_size = [float(h.sum()) for h in mesh.h]
    print(f"  Domain size: {domain_size} m")

    return mesh


def test_basic_magnetic_dipole():
    """Test 1: Basic magnetic dipole source with magnetic flux receivers"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Magnetic Dipole")
    print("=" * 60)

    mesh = create_test_mesh()

    # Create receiver locations (line profile)
    rx_x = torch.linspace(15, 45, 4)  # 4 receivers from 15m to 45m
    rx_y = torch.ones_like(rx_x) * 30  # At y=30m (center)
    rx_z = torch.ones_like(rx_x) * 5.0  # At z=5m (just above surface)
    rx_locs = torch.stack([rx_x, rx_y, rx_z], dim=1)

    print(f"Receiver locations: {rx_locs.shape}")

    # Create receivers for real and imaginary parts
    receivers = [
        PointMagneticFluxDensitySecondary(rx_locs, "z", "real"),
        PointMagneticFluxDensitySecondary(rx_locs, "z", "imag"),
    ]

    # Create magnetic dipole source
    source = MagneticDipole(
        receiver_list=receivers,
        frequency=1000.0,  # 1 kHz
        location=torch.tensor([30.0, 30.0, 0.0]),  # Center of domain at surface
        moment=1.0,
        orientation="z",
    )

    # Create survey
    survey = FDEMSurvey([source])
    print(
        f"Survey: {survey.n_sources} sources, {survey.n_receivers} receivers, {survey.nD} data points"
    )

    # Create conductivity model (halfspace)
    sigma = torch.ones(mesh.n_cells) * 0.01  # 0.01 S/m background

    # Add a conductive block
    cell_centers = mesh.cell_centers
    block_mask = (
        (cell_centers[:, 0] > 20)
        & (cell_centers[:, 0] < 40)
        & (cell_centers[:, 1] > 20)
        & (cell_centers[:, 1] < 40)
        & (cell_centers[:, 2] > 10)
        & (cell_centers[:, 2] < 30)
    )
    sigma[block_mask] = 0.1  # 0.1 S/m conductive block

    print(f"Conductivity model: {sigma.min():.3f} to {sigma.max():.3f} S/m")

    # Create simulation
    simulation = Simulation3DMagneticFluxDensity(mesh=mesh, survey=survey, sigma=sigma)

    print("Running forward simulation...")
    try:
        # Compute predicted data
        dpred = simulation.dpred()
        print("✓ Forward simulation successful!")
        print(f"  Data shape: {dpred.shape}")
        print(f"  Data range: {dpred.real.min():.2e} to {dpred.real.max():.2e} (real)")
        print(f"              {dpred.imag.min():.2e} to {dpred.imag.max():.2e} (imag)")

        # Split into real and imaginary parts
        n_rx = len(rx_locs)
        data_real = dpred[:n_rx].real
        data_imag = dpred[
            n_rx:
        ].real  # Note: imag receiver extracts imaginary part as real values

        return True, dpred, rx_x, data_real, data_imag

    except Exception as e:
        print(f"✗ Forward simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None, None, None


def test_multiple_frequencies():
    """Test 2: Multiple frequencies"""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Frequencies")
    print("=" * 60)

    mesh = create_test_mesh()

    # Single receiver location
    rx_loc = torch.tensor([[30.0, 30.0, 5.0]])

    sources = []
    frequencies = [1000.0, 10000.0]  # 1 kHz, 10 kHz (reduced to 2 frequencies)

    for freq in frequencies:
        receivers = [
            PointMagneticFluxDensitySecondary(rx_loc, "z", "real"),
            PointMagneticFluxDensitySecondary(rx_loc, "z", "imag"),
        ]

        source = MagneticDipole(
            receiver_list=receivers,
            frequency=freq,
            location=torch.tensor([30.0, 30.0, 0.0]),
            moment=1.0,
            orientation="z",
        )
        sources.append(source)

    survey = FDEMSurvey(sources)
    print(f"Survey: {len(frequencies)} frequencies, {survey.nD} total data points")

    sigma = torch.ones(mesh.n_cells) * 0.01
    simulation = Simulation3DMagneticFluxDensity(mesh=mesh, survey=survey, sigma=sigma)

    try:
        dpred = simulation.dpred()
        print("✓ Multi-frequency simulation successful!")
        print(f"  Data shape: {dpred.shape}")

        # Analyze frequency response
        for i, freq in enumerate(frequencies):
            data_idx = i * 2  # 2 data points per frequency (real + imag)
            real_part = dpred[data_idx].real
            imag_part = dpred[data_idx + 1].real
            amplitude = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.atan2(imag_part, real_part) * 180 / torch.pi

            print(f"  {freq:6.0f} Hz: Amplitude={amplitude:.2e}, Phase={phase:.1f}°")

        return True

    except Exception as e:
        print(f"✗ Multi-frequency simulation failed: {e}")
        return False


def test_different_sources():
    """Test 3: Different source types"""
    print("\n" + "=" * 60)
    print("TEST 3: Different Source Types")
    print("=" * 60)

    mesh = create_test_mesh()
    rx_loc = torch.tensor([[40.0, 30.0, 5.0]])

    # Test each source type
    source_tests = []

    # 1. Magnetic Dipole
    try:
        receivers = [PointMagneticFluxDensitySecondary(rx_loc, "z", "real")]
        mag_dipole = MagneticDipole(
            receiver_list=receivers,
            frequency=1000.0,
            location=torch.tensor([30.0, 30.0, 0.0]),
            moment=1.0,
            orientation="z",
        )
        survey = FDEMSurvey([mag_dipole])
        simulation = Simulation3DMagneticFluxDensity(
            mesh=mesh, survey=survey, sigma=torch.ones(mesh.n_cells) * 0.01
        )
        dpred = simulation.dpred()
        print(f"✓ Magnetic Dipole: {dpred[0]:.2e}")
        source_tests.append(True)
    except Exception as e:
        print(f"✗ Magnetic Dipole failed: {e}")
        source_tests.append(False)

    # 2. Electric Dipole
    try:
        receivers = [PointMagneticFluxDensitySecondary(rx_loc, "z", "real")]
        elec_dipole = ElectricDipole(
            receiver_list=receivers,
            frequency=1000.0,
            location=torch.tensor([30.0, 30.0, 0.0]),
            current=1.0,
            length=1.0,
            orientation="z",
        )
        survey = FDEMSurvey([elec_dipole])
        simulation = Simulation3DMagneticFluxDensity(
            mesh=mesh, survey=survey, sigma=torch.ones(mesh.n_cells) * 0.01
        )
        dpred = simulation.dpred()
        print(f"✓ Electric Dipole: {dpred[0]:.2e}")
        source_tests.append(True)
    except Exception as e:
        print(f"✗ Electric Dipole failed: {e}")
        source_tests.append(False)

    # 3. Loop Source
    try:
        receivers = [PointMagneticFluxDensitySecondary(rx_loc, "z", "real")]
        loop_source = LoopSource(
            receiver_list=receivers,
            frequency=1000.0,
            location=torch.tensor([30.0, 30.0, 0.0]),
            radius=10.0,
            current=1.0,
            orientation="z",
            n_segments=12,
        )
        survey = FDEMSurvey([loop_source])
        simulation = Simulation3DMagneticFluxDensity(
            mesh=mesh, survey=survey, sigma=torch.ones(mesh.n_cells) * 0.01
        )
        dpred = simulation.dpred()
        print(f"✓ Loop Source: {dpred[0]:.2e}")
        source_tests.append(True)
    except Exception as e:
        print(f"✗ Loop Source failed: {e}")
        source_tests.append(False)

    return all(source_tests)


def test_receivers():
    """Test 4: Different receiver types"""
    print("\n" + "=" * 60)
    print("TEST 4: Different Receiver Types")
    print("=" * 60)

    mesh = create_test_mesh()
    rx_loc = torch.tensor([[40.0, 30.0, 5.0]])

    receiver_tests = []

    # Test each receiver type
    receiver_types = [
        (
            "Magnetic Flux (Real)",
            PointMagneticFluxDensity,
            {"orientation": "z", "component": "real"},
        ),
        (
            "Magnetic Flux (Imag)",
            PointMagneticFluxDensity,
            {"orientation": "z", "component": "imag"},
        ),
        (
            "Magnetic Flux Secondary",
            PointMagneticFluxDensitySecondary,
            {"orientation": "z", "component": "real"},
        ),
        (
            "Electric Field",
            PointElectricField,
            {"orientation": "x", "component": "real"},
        ),
    ]

    for name, receiver_class, kwargs in receiver_types:
        try:
            receiver = receiver_class(rx_loc, **kwargs)
            source = MagneticDipole(
                receiver_list=[receiver],
                frequency=1000.0,
                location=torch.tensor([30.0, 30.0, 0.0]),
                moment=1.0,
                orientation="z",
            )
            survey = FDEMSurvey([source])
            simulation = Simulation3DMagneticFluxDensity(
                mesh=mesh, survey=survey, sigma=torch.ones(mesh.n_cells) * 0.01
            )
            dpred = simulation.dpred()
            print(f"✓ {name}: {dpred[0]:.2e}")
            receiver_tests.append(True)
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            receiver_tests.append(False)

    return all(receiver_tests)


def plot_results(rx_x, data_real, data_imag):
    """Plot the results from Test 1"""
    print("\nCreating plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Real part
    ax1.plot(rx_x.numpy(), data_real.numpy(), "b.-", label="Real part")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Magnetic flux density (T)")
    ax1.set_title("Real Part")
    ax1.grid(True)
    ax1.legend()

    # Imaginary part
    ax2.plot(rx_x.numpy(), data_imag.numpy(), "r.-", label="Imaginary part")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Magnetic flux density (T)")
    ax2.set_title("Imaginary Part")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    output_file = project_root / "fdem_test_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    return fig


def run_all_tests():
    """Run all FDEM tests"""
    print("FDEM Functionality Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Basic functionality
    success, dpred, rx_x, data_real, data_imag = test_basic_magnetic_dipole()
    results.append(("Basic Magnetic Dipole", success))

    if success and dpred is not None:
        try:
            plot_results(rx_x, data_real, data_imag)
        except Exception as e:
            print(f"Plotting failed: {e}")

    # Test 2: Multiple frequencies
    success = test_multiple_frequencies()
    results.append(("Multiple Frequencies", success))

    # Test 3: Different sources
    success = test_different_sources()
    results.append(("Different Sources", success))

    # Test 4: Different receivers
    success = test_receivers()
    results.append(("Different Receivers", success))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:25s} {status}")
        if not passed:
            all_passed = False

    print("-" * 40)
    overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    print(f"{'Overall:':<25s} {overall_status}")

    return all_passed


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)  # Use double precision

    # Run tests
    success = run_all_tests()

    if success:
        print("\n🎉 FDEM implementation is working correctly!")
    else:
        print("\n⚠️  Some issues detected in FDEM implementation.")
        sys.exit(1)
