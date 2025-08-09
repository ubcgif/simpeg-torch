#!/usr/bin/env python3
"""
Detailed 3D DC Resistivity Profiling Benchmark
==============================================

This script profiles a single medium-sized 3D DC resistivity forward simulation
to understand where time is spent in both simpeg-torch and original SimPEG.
Uses cProfile and line_profiler to break down performance by function.
"""

import cProfile
import pstats
import io
import time
import psutil
import gc
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Import both SimPEG versions
# =============================================================================

print("Importing libraries...")

# simpeg-torch
try:
    from simpegtorch.discretize import TensorMesh as TorchTensorMesh
    from simpegtorch.electromagnetics.resistivity import (
        Simulation3DNodal as TorchSimulation3DNodal,
        Survey as TorchSurvey,
    )
    from simpegtorch.electromagnetics.utils import (
        generate_dcip_sources_line as torch_generate_sources,
    )
    from simpegtorch.utils import create_flat_topography

    HAVE_TORCH = True
    print("✅ simpeg-torch imported")
except ImportError as e:
    print(f"❌ Failed to import simpeg-torch: {e}")
    HAVE_TORCH = False

# Original SimPEG
try:
    from discretize import TensorMesh as OrigTensorMesh
    from simpeg.electromagnetics.static import resistivity as dc
    from simpeg import maps
    from simpeg.electromagnetics.static.utils.static_utils import (
        generate_dcip_sources_line as orig_generate_sources,
    )

    HAVE_ORIG = True
    print("✅ Original SimPEG imported")
except ImportError as e:
    print(f"❌ Failed to import original SimPEG: {e}")
    HAVE_ORIG = False

if not (HAVE_TORCH and HAVE_ORIG):
    print("❌ Cannot run benchmark without both implementations")
    exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Single medium-sized problem configuration
CONFIG = {
    "name": "medium_nodal_profile",
    "nx": 80,
    "ny": 80,
    "nz": 50,  # 16,000 cells
    "dx": 10.0,
    "dy": 10.0,
    "dz": 10.0,  # 25m cells
    "n_sources": 11,
    "n_receivers_per_source": 6,
    "model_type": "layered",  # More interesting than homogeneous
    "use_gpu": True,
}

print(
    f"Configuration: {CONFIG['nx']}×{CONFIG['ny']}×{CONFIG['nz']} = {CONFIG['nx']*CONFIG['ny']*CONFIG['nz']:,} cells"
)

# =============================================================================
# Setup Functions
# =============================================================================


def create_mesh_torch():
    """Create simpeg-torch mesh"""
    hx = torch.full((CONFIG["nx"],), CONFIG["dx"], dtype=torch.float64)
    hy = torch.full((CONFIG["ny"],), CONFIG["dy"], dtype=torch.float64)
    hz = torch.full((CONFIG["nz"],), CONFIG["dz"], dtype=torch.float64)

    origin = torch.tensor(
        [
            -CONFIG["nx"] * CONFIG["dx"] / 2,
            -CONFIG["ny"] * CONFIG["dy"] / 2,
            -CONFIG["nz"] * CONFIG["dz"],
        ],
        dtype=torch.float64,
    )

    device = torch.device(
        "cuda" if CONFIG["use_gpu"] and torch.cuda.is_available() else "cpu"
    )
    mesh = TorchTensorMesh([hx, hy, hz], origin=origin, device=device)
    return mesh


def create_mesh_orig():
    """Create original SimPEG mesh"""
    hx = np.full(CONFIG["nx"], CONFIG["dx"])
    hy = np.full(CONFIG["ny"], CONFIG["dy"])
    hz = np.full(CONFIG["nz"], CONFIG["dz"])

    origin = np.array(
        [
            -CONFIG["nx"] * CONFIG["dx"] / 2,
            -CONFIG["ny"] * CONFIG["dy"] / 2,
            -CONFIG["nz"] * CONFIG["dz"],
        ]
    )

    mesh = OrigTensorMesh([hx, hy, hz], origin=origin)
    return mesh


def create_layered_model_torch(mesh):
    """Create 3-layer conductivity model for simpeg-torch"""
    z_centers = mesh.cell_centers[:, 2]
    sigma = torch.full(
        (mesh.nC,), 0.01, dtype=torch.float64, device=mesh.device
    )  # 100 ohm-m background

    # Layer 1: 0 to -100m = 50 ohm-m (0.02 S/m)
    mask1 = z_centers > -100
    sigma[mask1] = 0.02

    # Layer 2: -100 to -300m = 200 ohm-m (0.005 S/m)
    mask2 = (z_centers <= -100) & (z_centers > -300)
    sigma[mask2] = 0.005

    # Layer 3: below -300m = 1000 ohm-m (0.001 S/m)
    mask3 = z_centers <= -300
    sigma[mask3] = 0.001

    return sigma


def create_layered_model_orig(mesh):
    """Create 3-layer conductivity model for original SimPEG"""
    z_centers = mesh.cell_centers[:, 2]
    sigma = np.full(mesh.nC, 0.01)  # 100 ohm-m background

    # Layer 1: 0 to -100m = 50 ohm-m (0.02 S/m)
    mask1 = z_centers > -100
    sigma[mask1] = 0.02

    # Layer 2: -100 to -300m = 200 ohm-m (0.005 S/m)
    mask2 = (z_centers <= -100) & (z_centers > -300)
    sigma[mask2] = 0.005

    # Layer 3: below -300m = 1000 ohm-m (0.001 S/m)
    mask3 = z_centers <= -300
    sigma[mask3] = 0.001

    return sigma


def create_survey_torch():
    """Create survey for simpeg-torch"""
    electrode_spacing = 50.0
    n_electrodes = CONFIG["n_sources"] + 3

    # Create topography
    x_extent = n_electrodes * electrode_spacing
    topo_xyz = create_flat_topography(
        x_extent=(-x_extent / 2, x_extent / 2),
        y_extent=(-100, 100),
        elevation=0.0,
        n_points_x=21,
        n_points_y=11,
    )

    # Generate sources
    end_locations = np.array(
        [
            -(n_electrodes - 1) * electrode_spacing / 2,
            (n_electrodes - 1) * electrode_spacing / 2,
            0.0,
            0.0,
        ]
    )

    source_list = torch_generate_sources(
        survey_type="dipole-dipole",
        data_type="volt",
        dimension_type="3D",
        end_points=end_locations,
        topo=topo_xyz,
        num_rx_per_src=CONFIG["n_receivers_per_source"],
        station_spacing=electrode_spacing,
    )

    source_list = source_list[: CONFIG["n_sources"]]
    survey = TorchSurvey(source_list)
    return survey


def create_survey_orig():
    """Create survey for original SimPEG"""
    electrode_spacing = 50.0
    n_electrodes = CONFIG["n_sources"] + 3

    # Create topography
    x_extent = n_electrodes * electrode_spacing
    x_topo = np.linspace(-x_extent / 2, x_extent / 2, 21)
    y_topo = np.linspace(-100, 100, 11)
    X, Y = np.meshgrid(x_topo, y_topo)
    Z = np.zeros_like(X)
    topo_xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Generate sources
    end_locations = np.array(
        [
            -(n_electrodes - 1) * electrode_spacing / 2,
            (n_electrodes - 1) * electrode_spacing / 2,
            0.0,
            0.0,
        ]
    )

    source_list = orig_generate_sources(
        survey_type="dipole-dipole",
        data_type="volt",
        dimension_type="3D",
        end_points=end_locations,
        topo=topo_xyz,
        num_rx_per_src=CONFIG["n_receivers_per_source"],
        station_spacing=electrode_spacing,
    )

    source_list = source_list[: CONFIG["n_sources"]]
    survey = dc.Survey(source_list)
    return survey


# =============================================================================
# Profiled Benchmark Functions
# =============================================================================


def run_torch_simulation():
    """Run simpeg-torch simulation with detailed timing"""
    print("\n🔥 Profiling simpeg-torch simulation...")

    # Set device
    device = torch.device(
        "cuda" if CONFIG["use_gpu"] and torch.cuda.is_available() else "cpu"
    )
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    # Create components with timing
    times = {}

    start = time.time()
    mesh = create_mesh_torch()
    times["mesh_creation"] = time.time() - start

    start = time.time()
    sigma = create_layered_model_torch(mesh)
    resistivity_model = 1.0 / sigma
    times["model_creation"] = time.time() - start

    start = time.time()
    survey = create_survey_torch()
    times["survey_creation"] = time.time() - start

    start = time.time()
    simulation = TorchSimulation3DNodal(mesh, survey=survey, sigma=sigma)
    times["simulation_setup"] = time.time() - start

    # Warm up run
    if torch.cuda.is_available() and CONFIG["use_gpu"]:
        torch.cuda.synchronize()
    _ = simulation.dpred(resistivity_model)

    # Profile the main computation
    print("  Running cProfile on forward simulation...")
    pr = cProfile.Profile()

    if torch.cuda.is_available() and CONFIG["use_gpu"]:
        torch.cuda.synchronize()

    start = time.time()
    pr.enable()
    dpred = simulation.dpred(resistivity_model)
    pr.disable()

    if torch.cuda.is_available() and CONFIG["use_gpu"]:
        torch.cuda.synchronize()

    times["forward_simulation"] = time.time() - start

    # Get profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions
    profile_output = s.getvalue()

    return {
        "times": times,
        "profile": profile_output,
        "data": dpred.detach().cpu().numpy(),
        "data_range": (dpred.min().item(), dpred.max().item()),
        "device": str(device),
        "total_time": sum(times.values()),
    }


def run_orig_simulation():
    """Run original SimPEG simulation with detailed timing"""
    print("\n📊 Profiling original SimPEG simulation...")

    # Create components with timing
    times = {}

    start = time.time()
    mesh = create_mesh_orig()
    times["mesh_creation"] = time.time() - start

    start = time.time()
    sigma = create_layered_model_orig(mesh)
    times["model_creation"] = time.time() - start

    start = time.time()
    survey = create_survey_orig()
    times["survey_creation"] = time.time() - start

    start = time.time()
    sigma_map = maps.IdentityMap(nP=mesh.nC)
    simulation = dc.Simulation3DNodal(mesh, survey=survey, sigmaMap=sigma_map)
    times["simulation_setup"] = time.time() - start

    # Profile the main computation
    print("  Running cProfile on forward simulation...")
    pr = cProfile.Profile()

    start = time.time()
    pr.enable()
    dpred = simulation.dpred(sigma)
    pr.disable()
    times["forward_simulation"] = time.time() - start

    # Get profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions
    profile_output = s.getvalue()

    return {
        "times": times,
        "profile": profile_output,
        "data": dpred,
        "data_range": (dpred.min(), dpred.max()),
        "total_time": sum(times.values()),
    }


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_profiles(torch_result, orig_result):
    """Analyze and compare profiling results"""
    print(f"\n{'='*80}")
    print("📈 DETAILED PROFILING ANALYSIS")
    print(f"{'='*80}")

    # Timing breakdown
    print("\n⏱️  TIMING BREAKDOWN:")
    print(f"{'Phase':<20} {'Torch (s)':<12} {'Original (s)':<12} {'Ratio':<8}")
    print("-" * 60)

    for phase in torch_result["times"].keys():
        torch_time = torch_result["times"][phase]
        orig_time = orig_result["times"].get(phase, 0)
        ratio = torch_time / orig_time if orig_time > 0 else float("inf")
        print(f"{phase:<20} {torch_time:<12.4f} {orig_time:<12.4f} {ratio:<8.2f}")

    total_torch = torch_result["total_time"]
    total_orig = orig_result["total_time"]
    overall_speedup = total_orig / total_torch if total_torch > 0 else 0

    print("-" * 60)
    print(
        f"{'TOTAL':<20} {total_torch:<12.4f} {total_orig:<12.4f} {overall_speedup:<8.2f}"
    )

    # Accuracy check
    torch_data = torch_result["data"]
    orig_data = orig_result["data"]
    accuracy_error = np.linalg.norm(torch_data - orig_data) / np.linalg.norm(orig_data)

    print("\n🎯 ACCURACY:")
    print(f"   Relative error: {accuracy_error:.2e}")
    print(
        f"   Torch data range: [{torch_result['data_range'][0]:.2e}, {torch_result['data_range'][1]:.2e}]"
    )
    print(
        f"   Original data range: [{orig_result['data_range'][0]:.2e}, {orig_result['data_range'][1]:.2e}]"
    )

    if accuracy_error < 1e-6:
        print("   ✅ ACCURACY TEST PASSED")
    else:
        print("   ⚠️  ACCURACY TEST FAILED - High error!")

    # Performance summary
    print("\n🚀 PERFORMANCE SUMMARY:")
    print(f"   Overall speedup: {overall_speedup:.2f}x")
    print(f"   Torch device: {torch_result.get('device', 'unknown')}")

    if overall_speedup > 1.0:
        print("   ✅ PERFORMANCE TEST PASSED - simpeg-torch is faster")
    else:
        print("   ⚠️  simpeg-torch is slower than original")

    return accuracy_error, overall_speedup


def save_detailed_profiles(torch_result, orig_result):
    """Save detailed profiling results to files"""
    # Save torch profile
    with open("torch_profile.txt", "w") as f:
        f.write("SIMPEG-TORCH DETAILED PROFILE\n")
        f.write("=" * 50 + "\n\n")
        f.write("TIMING BREAKDOWN:\n")
        for phase, time_val in torch_result["times"].items():
            f.write(f"{phase}: {time_val:.4f}s\n")
        f.write(f"\nTOTAL TIME: {torch_result['total_time']:.4f}s\n")
        f.write(f"DEVICE: {torch_result.get('device', 'unknown')}\n\n")
        f.write("CPROFILE OUTPUT:\n")
        f.write(torch_result["profile"])

    # Save original profile
    with open("original_profile.txt", "w") as f:
        f.write("ORIGINAL SIMPEG DETAILED PROFILE\n")
        f.write("=" * 50 + "\n\n")
        f.write("TIMING BREAKDOWN:\n")
        for phase, time_val in orig_result["times"].items():
            f.write(f"{phase}: {time_val:.4f}s\n")
        f.write(f"\nTOTAL TIME: {orig_result['total_time']:.4f}s\n\n")
        f.write("CPROFILE OUTPUT:\n")
        f.write(orig_result["profile"])

    print("\n💾 Detailed profiles saved:")
    print("   - torch_profile.txt")
    print("   - original_profile.txt")


def create_timing_plot(torch_result, orig_result):
    """Create timing comparison plot"""
    phases = list(torch_result["times"].keys())
    torch_times = [torch_result["times"][phase] for phase in phases]
    orig_times = [orig_result["times"].get(phase, 0) for phase in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Phase timing comparison
    x = np.arange(len(phases))
    width = 0.35

    ax1.bar(x - width / 2, torch_times, width, label="simpeg-torch", alpha=0.8)
    ax1.bar(x + width / 2, orig_times, width, label="Original SimPEG", alpha=0.8)
    ax1.set_xlabel("Simulation Phase")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Timing by Phase")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("_", "\n") for p in phases], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative timing
    torch_cumulative = np.cumsum([0] + torch_times)
    orig_cumulative = np.cumsum([0] + orig_times)

    phase_labels = ["start"] + phases
    ax2.plot(phase_labels, torch_cumulative, "o-", label="simpeg-torch", linewidth=2)
    ax2.plot(phase_labels, orig_cumulative, "s-", label="Original SimPEG", linewidth=2)
    ax2.set_xlabel("Simulation Phase")
    ax2.set_ylabel("Cumulative Time (seconds)")
    ax2.set_title("Cumulative Timing")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("timing_comparison.png", dpi=150, bbox_inches="tight")
    print("📊 Timing plot saved to: timing_comparison.png")
    plt.show()


# =============================================================================
# Main Function
# =============================================================================


def main():
    """Run the detailed profiling benchmark"""
    print("🔬 3D DC Resistivity Detailed Profiling Benchmark")
    print("=" * 60)

    # Hardware info
    print("💻 Hardware:")
    print(f"   CPU: {psutil.cpu_count()} cores")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )
    else:
        print("   GPU: Not available - using CPU only")

    print("\n📋 Problem Configuration:")
    print(
        f"   Mesh: {CONFIG['nx']}×{CONFIG['ny']}×{CONFIG['nz']} = {CONFIG['nx']*CONFIG['ny']*CONFIG['nz']:,} cells"
    )
    print(f"   Cell size: {CONFIG['dx']}m × {CONFIG['dy']}m × {CONFIG['dz']}m")
    print(
        f"   Survey: {CONFIG['n_sources']} sources, ~{CONFIG['n_sources']*CONFIG['n_receivers_per_source']} measurements"
    )
    print(f"   Model: {CONFIG['model_type']} conductivity")
    print(
        f"   GPU: {'Enabled' if CONFIG['use_gpu'] and torch.cuda.is_available() else 'Disabled'}"
    )

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run benchmarks
    print("\n🚀 Starting detailed profiling...")

    torch_result = run_torch_simulation()

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orig_result = run_orig_simulation()

    # Analyze results
    accuracy_error, speedup = analyze_profiles(torch_result, orig_result)

    # Save detailed results
    save_detailed_profiles(torch_result, orig_result)

    # Create visualizations
    create_timing_plot(torch_result, orig_result)

    # Final summary
    print("\n🎉 PROFILING COMPLETED!")
    print(f"   Accuracy: {accuracy_error:.2e} relative error")
    print(f"   Performance: {speedup:.2f}x speedup")
    print(f"   Device: {torch_result.get('device', 'unknown')}")
    print("\n📁 Output files:")
    print("   - torch_profile.txt (detailed function profiling)")
    print("   - original_profile.txt (detailed function profiling)")
    print("   - timing_comparison.png (timing visualization)")


if __name__ == "__main__":
    main()
