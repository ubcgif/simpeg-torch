"""
Simple DC resistivity test - reduced mesh size for faster computation.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# simpeg-torch imports
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity import (
    DCStaticSimulationNodal,
    SrcDipole,
    RxDipole,
    Survey,
)
from simpegtorch.utils import (
    get_indices_sphere,
    active_from_xyz,
    create_flat_topography,
    InjectActiveCells,
)

# Set PyTorch settings
torch.set_default_dtype(torch.float64)
print("Starting simple DC test...")

# =============================================================================
# Create Small Mesh
# =============================================================================

# Define small mesh parameters
dx = dy = dz = 100.0  # 100m cells
nx = ny = nz = 20  # 20x20x20 = 8000 cells total

# Create cell sizes
hx = torch.full((nx,), dx)
hy = torch.full((ny,), dy)
hz = torch.full((nz,), dz)

# Create mesh origin (center the mesh)
origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -1000.0])  # 1000m below surface

# Create the tensor mesh
mesh = TensorMesh([hx, hy, hz], origin=origin)
print(f"Small mesh: {mesh.nC} cells ({nx} x {ny} x {nz})")

# =============================================================================
# Define Active Cells and Model
# =============================================================================

# Create flat topography
topo_xyz = create_flat_topography(
    x_extent=(-1000, 1000),
    y_extent=(-1000, 1000),
    elevation=0.0,
    n_points_x=21,
    n_points_y=21,
)

# Get active cells
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = torch.sum(active_cells).item()

# Create mapping
air_resistivity = 1e8
active_mapping = InjectActiveCells(mesh, active_cells, valInactive=air_resistivity)

# Define simple model
background_resistivity = 100.0  # 100 Ohm-m
conductor_resistivity = 10.0  # 10 Ohm-m

# Create background model on active cells
active_model = torch.full((n_active,), background_resistivity, dtype=torch.float64)

# Add conductive sphere at center
sphere_center = [0.0, 0.0, -500.0]  # 500m deep
sphere_radius = 300.0  # 300m radius
active_centers = mesh.cell_centers[active_cells]
conductor_mask = get_indices_sphere(sphere_center, sphere_radius, active_centers)
active_model[conductor_mask] = conductor_resistivity

print(f"Active model range: {active_model.min():.1f} to {active_model.max():.1f} Ohm-m")

# Convert to conductivity for simulation
active_conductivity = 1.0 / active_model
full_conductivity = active_mapping * active_conductivity

print(
    f"Conductivity range: {full_conductivity.min():.2e} to {full_conductivity.max():.2e} S/m"
)

# =============================================================================
# Create Simple Survey
# =============================================================================

# Create a simple dipole-dipole survey
electrode_locs = []
for i in range(5):  # 5 electrodes along x-axis
    x = -400 + i * 200  # Every 200m from -400 to 400
    electrode_locs.append([x, 0.0, 0.0])

electrode_locs = torch.tensor(electrode_locs, dtype=torch.float64)

# Create one dipole source
src_loc = torch.stack([electrode_locs[0], electrode_locs[1]])  # A-B

# Create dipole receiver (M-N)
rx = RxDipole(
    locations_m=electrode_locs[2].unsqueeze(0),  # M electrode
    locations_n=electrode_locs[3].unsqueeze(0),  # N electrode
    data_type="volt",
)

# Create source
src = SrcDipole(
    receiver_list=[rx],
    location_a=electrode_locs[0],
    location_b=electrode_locs[1],
    data_type="volt",
)

# Create survey
survey = Survey([src])
print(f"Simple survey: {survey.nSrc} sources, {survey.nD} data points")

# =============================================================================
# Run Forward Simulation
# =============================================================================

try:
    print("\n=== CREATING NODAL DC SIMULATION ===")
    simulation = DCStaticSimulationNodal(mesh, survey=survey)
    print(f"Created nodal simulation with {simulation.bc_type} boundary conditions")

    print(f"Mesh info: nN={mesh.nN}, nE={mesh.nE}, nF={mesh.nF}, nC={mesh.nC}")

    # Run forward simulation
    print("\nRunning forward simulation...")
    dpred = simulation.dpred(full_conductivity)
    print(f"Predicted data: {dpred.item():.6e} V/A")

    # =============================================================================
    # Visualize Results
    # =============================================================================

    # Plot model slice
    print("\nPlotting model...")
    full_resistivity = active_mapping * active_model

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # XZ slice
    mesh.plot_slice(
        full_resistivity,
        normal="Y",
        slice_loc=0.0,
        ax=ax1,
        pcolor_opts={"cmap": "RdYlBu_r", "norm": LogNorm()},
    )
    ax1.set_title("Resistivity Model - XZ Slice (Y=0)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")

    # Plot survey layout
    ax2.scatter(electrode_locs[:, 0], electrode_locs[:, 1], c="red", s=100, marker="^")
    ax2.plot(
        [src_loc[0, 0], src_loc[1, 0]],
        [src_loc[0, 1], src_loc[1, 1]],
        "r-",
        linewidth=3,
        label="Source A-B",
    )
    # Plot receiver locations
    rx_m = rx.locations_m[0]  # M electrode
    rx_n = rx.locations_n[0]  # N electrode
    ax2.plot(
        [rx_m[0], rx_n[0]], [rx_m[1], rx_n[1]], "b-", linewidth=3, label="Receiver M-N"
    )
    ax2.set_title("Survey Layout")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend()
    ax2.grid(True)
    ax2.axis("equal")

    plt.tight_layout()
    plt.savefig(
        "/Users/tsuchijo/Documents/GIF_Work/simple_dc_test.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # Test apparent resistivity calculation
    print("\nTesting apparent resistivity calculation...")
    from simpegtorch.electromagnetics.utils import (
        apparent_resistivity_from_voltage,
        geometric_factor,
    )

    # Debug geometric factor
    G = geometric_factor(survey, space_type="half space")
    print(f"Geometric factor: {G.item():.6e}")
    print(f"Voltage: {dpred.item():.6e} V/A")

    # Calculate apparent resistivity manually first
    eps = 1e-10
    manual_rho = torch.abs(dpred * (1.0 / (G + eps)))
    print(f"Manual calculation: {manual_rho.item():.6f} Ohm-m")

    # Calculate apparent resistivity using function
    rho_app = apparent_resistivity_from_voltage(survey, dpred, space_type="half space")
    print(f"Function result: {rho_app.item():.6f} Ohm-m")

    print("\n=== SUCCESS! ===")
    print("✅ Forward simulation completed successfully")
    print(f"✅ Predicted voltage: {dpred.item():.6e} V/A")
    print(f"✅ Apparent resistivity: {rho_app.item():.1f} Ohm-m")
    print("✅ Model visualization created")

except Exception as e:
    print(f"\nError in simulation: {e}")
    import traceback

    traceback.print_exc()
