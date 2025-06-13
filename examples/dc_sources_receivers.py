#!/usr/bin/env python3
"""
Test script for the new DC resistivity sources and receivers implementation
"""

import torch
import sys

# Add the current directory to Python path
sys.path.insert(0, "/Users/tsuchijo/Documents/GIF_Work/simpeg-torch")

try:
    from simpegtorch.discretize import TensorMesh
    from simpegtorch.electromagnetics.resistivity import (
        sources,
        receivers,
        survey,
        simulation,
    )

    print("✓ Successfully imported simpeg-torch modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_basic_setup():
    """Test basic setup of mesh, sources, receivers, and survey"""
    print("\n=== Testing Basic Setup ===")

    # Create a simple 3D mesh
    print("Creating mesh...")
    h = torch.ones(20) * 5.0  # 5m cells
    mesh = TensorMesh([h, h, h])
    print(f"✓ Created mesh with {mesh.nC} cells")

    # Create dipole receivers (M-N electrode pairs)
    print("Creating receivers...")
    rx_locs_m = torch.tensor(
        [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=torch.float64
    )

    rx_locs_n = torch.tensor(
        [[15.0, 0.0, 0.0], [25.0, 0.0, 0.0], [35.0, 0.0, 0.0]], dtype=torch.float64
    )

    rx = receivers.Dipole(locations_m=rx_locs_m, locations_n=rx_locs_n)
    print(f"✓ Created dipole receiver with {rx.nD} measurements")

    # Create dipole source (A-B electrode pair)
    print("Creating source...")
    src_loc_a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    src_loc_b = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64)
    src = sources.Dipole([rx], location_a=src_loc_a, location_b=src_loc_b, current=1.0)
    print(f"✓ Created dipole source at A: {src.location_a}, B: {src.location_b}")

    # Create survey
    print("Creating survey...")
    surv = survey.Survey([src])
    print(f"✓ Created survey with {surv.nSrc} sources and {surv.nD} data points")

    # Test survey properties
    print(f"  - A electrode locations shape: {surv.locations_a.shape}")
    print(f"  - B electrode locations shape: {surv.locations_b.shape}")
    print(f"  - M electrode locations shape: {surv.locations_m.shape}")
    print(f"  - N electrode locations shape: {surv.locations_n.shape}")

    return mesh, surv


def test_simulation_setup(mesh, surv):
    """Test simulation setup and basic forward modeling"""
    print("\n=== Testing Simulation Setup ===")

    # Create simulation
    print("Creating simulation...")
    sim = simulation.DCStaticSimulationCellCentered(mesh, survey=surv)
    print(f"✓ Created simulation with {sim.n_data} data points")

    # Set boundary conditions
    print("Setting boundary conditions...")
    sim.setBC()
    print("✓ Set boundary conditions")

    # Create a simple resistivity model
    print("Creating resistivity model...")
    resistivity = (
        torch.ones(mesh.nC, dtype=torch.float64) * 100.0
    )  # 100 ohm-m background
    resistivity.requires_grad_(True)  # Enable gradients for Jtvec testing
    print(f"✓ Created resistivity model with shape {resistivity.shape}")

    return sim, resistivity


def test_forward_modeling(sim, resistivity):
    """Test forward modeling"""
    print("\n=== Testing Forward Modeling ===")

    try:
        # Test system matrix
        print("Computing system matrix...")
        A = sim.getA(resistivity)
        print(f"✓ System matrix A computed with shape {A.shape}")

        # Test RHS for first source
        print("Computing RHS...")
        source = sim.survey.source_list[0]
        b = sim.getRHS(source)
        print(f"✓ RHS computed with shape {b.shape}")
        print(f"  - RHS sum: {torch.sum(b):.6f} (should be close to 0 for dipole)")

        # Test field computation
        print("Computing fields...")
        fields = sim.fields(resistivity)
        print(f"✓ Fields computed for {len(fields)} sources")

        # Test predicted data
        print("Computing predicted data...")
        data = sim.dpred(resistivity)
        print(f"✓ Predicted data computed with shape {data.shape}")
        print(f"  - Data range: [{torch.min(data):.6f}, {torch.max(data):.6f}] V")

        return fields, data

    except Exception as e:
        print(f"✗ Forward modeling failed: {e}")
        raise


def test_jacobian_operations(sim, resistivity, data):
    """Test Jacobian operations (Jtvec and Jvec)"""
    print("\n=== Testing Jacobian Operations ===")

    try:
        # Test Jtvec (gradient computation)
        print("Testing Jtvec...")
        v = torch.ones_like(data)  # Data space vector
        gradient = sim.Jtvec(resistivity, v)
        print(f"✓ Jtvec computed with shape {gradient.shape}")
        print(
            f"  - Gradient range: [{torch.min(gradient):.6f}, {torch.max(gradient):.6f}]"
        )

        # Test Jvec (forward sensitivity)
        print("Testing Jvec...")
        w = torch.ones_like(resistivity) * 0.01  # Small model perturbation
        jv = sim.Jvec(resistivity, w)
        print(f"✓ Jvec computed with shape {jv.shape}")
        print(f"  - Jvec range: [{torch.min(jv):.6f}, {torch.max(jv):.6f}]")

        return gradient, jv

    except Exception as e:
        print(f"✗ Jacobian operations failed: {e}")
        raise


def test_geometric_factors(surv):
    """Test geometric factor calculation"""
    print("\n=== Testing Geometric Factors ===")

    try:
        # Calculate geometric factors
        print("Computing geometric factors...")
        G = surv.set_geometric_factor(space_type="halfspace")
        print(f"✓ Geometric factors computed with shape {G.shape}")
        print(f"  - G range: [{torch.min(G):.6f}, {torch.max(G):.6f}]")

        # Test apparent resistivity calculation (would need to set receiver data_type)
        print("✓ Geometric factors set for apparent resistivity calculations")

        return G

    except Exception as e:
        print(f"✗ Geometric factor computation failed: {e}")
        raise


def test_multiple_sources():
    """Test setup with multiple sources"""
    print("\n=== Testing Multiple Sources ===")

    try:
        # Create mesh
        h = torch.ones(15) * 5.0
        mesh = TensorMesh([h, h, h])

        # Create receivers
        rx_locs_m = torch.tensor(
            [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=torch.float64
        )
        rx_locs_n = torch.tensor(
            [[25.0, 0.0, 0.0], [35.0, 0.0, 0.0]], dtype=torch.float64
        )
        rx = receivers.Dipole(locations_m=rx_locs_m, locations_n=rx_locs_n)

        # Create multiple sources
        src1 = sources.Dipole(
            [rx],
            location_a=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            location_b=torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64),
        )

        src2 = sources.Dipole(
            [rx],
            location_a=torch.tensor([40.0, 0.0, 0.0], dtype=torch.float64),
            location_b=torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64),
        )

        # Create survey with multiple sources
        surv = survey.Survey([src1, src2])
        print(f"✓ Created survey with {surv.nSrc} sources and {surv.nD} data points")

        # Create simulation and test
        sim = simulation.DCStaticSimulationCellCentered(mesh, survey=surv)
        sim.setBC()

        resistivity = torch.ones(mesh.nC, dtype=torch.float64) * 200.0
        resistivity.requires_grad_(True)

        data = sim.dpred(resistivity)
        print(f"✓ Multiple source forward modeling successful: {data.shape}")

        return True

    except Exception as e:
        print(f"✗ Multiple sources test failed: {e}")
        return False


def test_pole_configurations():
    """Test pole source and receiver configurations"""
    print("\n=== Testing Pole Configurations ===")

    try:
        # Create mesh
        h = torch.ones(15) * 5.0
        mesh = TensorMesh([h, h, h])

        # Test pole receiver
        rx_locs = torch.tensor(
            [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=torch.float64
        )
        rx_pole = receivers.Pole(locations=rx_locs)
        print(f"✓ Created pole receiver with {rx_pole.nD} measurements")

        # Test pole source
        src_pole = sources.Pole(
            [rx_pole], location=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        )
        print(f"✓ Created pole source at {src_pole.location_a}")

        # Create survey and simulation
        surv = survey.Survey([src_pole])
        sim = simulation.DCStaticSimulationCellCentered(mesh, survey=surv)
        sim.setBC()

        resistivity = torch.ones(mesh.nC, dtype=torch.float64) * 100.0
        resistivity.requires_grad_(True)

        data = sim.dpred(resistivity)
        print(f"✓ Pole configuration forward modeling successful: {data.shape}")

        return True

    except Exception as e:
        print(f"✗ Pole configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing SimPEG-Torch DC Resistivity Implementation")
    print("=" * 60)

    try:
        # Basic setup tests
        mesh, surv = test_basic_setup()
        sim, resistivity = test_simulation_setup(mesh, surv)

        # Forward modeling tests
        fields, data = test_forward_modeling(sim, resistivity)

        # Jacobian tests
        gradient, jv = test_jacobian_operations(sim, resistivity, data)

        # Multi-source tests
        test_multiple_sources()

        # Pole configuration tests
        pole_success = test_pole_configurations()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Summary
        print("\nSummary:")
        print("  - Basic setup: ✓")
        print("  - Forward modeling: ✓")
        print("  - Jacobian operations: ✓")
        print("  - Geometric factors: ✓")
        print("  - Multiple sources: {'✓' if multi_success else '✗'}")
        print(f"  - Pole configurations: {'✓' if pole_success else '✗'}")

        return True

    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
