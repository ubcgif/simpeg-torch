"""
DC resistivity test with pseudosection plotting.

This example creates a more realistic dipole-dipole survey and plots
the apparent resistivity pseudosection.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# simpeg-torch imports
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity import (
    DCStaticSimulationNodal,
    Survey,
)
from simpegtorch.electromagnetics.utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    pseudo_locations,
    plot_pseudosection,
)
from simpegtorch.utils import (
    get_indices_sphere,
    active_from_xyz,
    create_flat_topography,
    InjectActiveCells,
)

# Set PyTorch settings
torch.set_default_dtype(torch.float64)
print("Starting DC test with pseudosection plotting...")

# =============================================================================
# Create Mesh
# =============================================================================

# Define mesh parameters
dx = dy = dz = 50.0  # 50m cells
nx = ny = nz = 30  # 30x30x30 = 27,000 cells total

# Create cell sizes
hx = torch.full((nx,), dx)
hy = torch.full((ny,), dy)
hz = torch.full((nz,), dz)

# Create mesh origin (center the mesh)
origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -750.0])  # 750m below surface

# Create the tensor mesh
mesh = TensorMesh([hx, hy, hz], origin=origin)
print(f"Mesh: {mesh.nC} cells ({nx} x {ny} x {nz})")

# =============================================================================
# Define Active Cells and Model
# =============================================================================

# Create flat topography
topo_xyz = create_flat_topography(
    x_extent=(-750, 750),
    y_extent=(-750, 750),
    elevation=0.0,
    n_points_x=31,
    n_points_y=31,
)

# Get active cells
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = torch.sum(active_cells).item()

# Create mapping
air_resistivity = 1e8
active_mapping = InjectActiveCells(mesh, active_cells, valInactive=air_resistivity)

# Define realistic model
background_resistivity = 100.0  # 100 Ohm-m background
conductor_resistivity = 10.0  # 10 Ohm-m conductive zone
resistor_resistivity = 1000.0  # 1000 Ohm-m resistive zone

# Create background model on active cells
active_model = torch.full((n_active,), background_resistivity, dtype=torch.float64)

# Add conductive zone at (-150, 0, -200) with radius 100m
conductor_center = [-150.0, 0.0, -200.0]
conductor_radius = 100.0
active_centers = mesh.cell_centers[active_cells]
conductor_mask = get_indices_sphere(conductor_center, conductor_radius, active_centers)
active_model[conductor_mask] = conductor_resistivity

# Add resistive zone at (150, 0, -200) with radius 100m
resistor_center = [150.0, 0.0, -200.0]
resistor_radius = 100.0
resistor_mask = get_indices_sphere(resistor_center, resistor_radius, active_centers)
active_model[resistor_mask] = resistor_resistivity

print(f"Active model range: {active_model.min():.1f} to {active_model.max():.1f} Ohm-m")

# Convert to conductivity for simulation
active_conductivity = 1.0 / active_model
full_conductivity = active_mapping * active_conductivity

print(
    f"Conductivity range: {full_conductivity.min():.2e} to {full_conductivity.max():.2e} S/m"
)

# =============================================================================
# Create Dipole-Dipole Survey
# =============================================================================

# Create a dipole-dipole survey line using the utility function
survey_type = "dipole-dipole"
data_type = "volt"
dimension_type = "3D"

# Survey line from -400m to 400m along x-axis
end_locations = torch.tensor([-400.0, 400.0, 0.0, 0.0])  # x1, x2, y1, y2
station_separation = 50.0  # 50m electrode spacing
num_rx_per_src = 8  # 8 receivers per source (n-spacings)

# Generate sources for the survey line
source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    topo_xyz,
    num_rx_per_src,
    station_separation,
)

# Create survey
survey = Survey(source_list)
print(f"Survey: {survey.nSrc} sources, {survey.nD} data points")

# =============================================================================
# Run Forward Simulation
# =============================================================================

try:
    print("\n=== RUNNING FORWARD SIMULATION ===")
    simulation = DCStaticSimulationNodal(mesh, survey=survey)
    print(f"Created nodal simulation with {simulation.bc_type} boundary conditions")

    print(f"Mesh info: nN={mesh.nN}, nE={mesh.nE}, nF={mesh.nF}, nC={mesh.nC}")

    # Run forward simulation
    print("Computing forward simulation...")
    dpred = simulation.dpred(full_conductivity)
    print(f"Predicted data shape: {dpred.shape}")
    print(f"Data range: {dpred.min():.6e} to {dpred.max():.6e} V/A")

    # =============================================================================
    # Compute Apparent Resistivities and Plot Pseudosection
    # =============================================================================

    # Calculate apparent resistivities
    print("\nCalculating apparent resistivities...")

    # Debug geometric factor calculation
    from simpegtorch.electromagnetics.utils.static_utils import geometric_factor

    G = geometric_factor(survey, space_type="half space")
    print(f"Geometric factor range: {G.min():.6e} to {G.max():.6e}")
    print(f"Voltage range: {dpred.min():.6e} to {dpred.max():.6e}")

    # Check for problematic values
    if torch.any(torch.isinf(G)) or torch.any(torch.isnan(G)):
        print("Warning: Geometric factor contains inf or nan values")
        print(f"Inf values: {torch.sum(torch.isinf(G))}")
        print(f"NaN values: {torch.sum(torch.isnan(G))}")

    apparent_resistivity = apparent_resistivity_from_voltage(
        survey, dpred, space_type="half space"
    )
    print(
        f"Apparent resistivity range: {apparent_resistivity.min():.1f} to {apparent_resistivity.max():.1f} Ohm-m"
    )

    # Create pseudosection plot
    print("\nCreating pseudosection plot...")

    # Get pseudo-locations for plotting
    pseudo_locs = pseudo_locations(survey)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Model cross-section
    ax1 = plt.subplot(2, 2, 1)
    full_resistivity = active_mapping * active_model
    mesh.plot_slice(
        full_resistivity,
        normal="Y",
        slice_loc=0.0,
        ax=ax1,
        pcolor_opts={"cmap": "RdYlBu_r", "norm": LogNorm(vmin=10, vmax=1000)},
    )
    ax1.set_title("True Resistivity Model - XZ Slice (Y=0)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")

    # Plot 2: Survey layout
    ax2 = plt.subplot(2, 2, 2)
    electrode_locs = survey.unique_electrode_locations
    if isinstance(electrode_locs, torch.Tensor):
        electrode_locs = electrode_locs.detach().cpu().numpy()

    ax2.scatter(electrode_locs[:, 0], electrode_locs[:, 1], c="red", s=20, marker="^")
    ax2.set_title("Survey Layout (Plan View)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.axis("equal")

    # Plot 3: Apparent resistivity pseudosection
    ax3 = plt.subplot(2, 1, 2)

    # Use the plot_pseudosection utility function
    plot_pseudosection(
        survey,
        dobs=apparent_resistivity,
        plot_type="contourf",
        ax=ax3,
        scale="log",
        cbar_label="Apparent Resistivity (Ohm-m)",
        contourf_opts={
            "cmap": "RdYlBu_r",
            "levels": 20,
        },
        data_locations=True,  # Show data points
    )
    ax3.set_title("Apparent Resistivity Pseudosection")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Elevation (m)")

    plt.tight_layout()
    plt.savefig(
        "/Users/tsuchijo/Documents/GIF_Work/dc_pseudosection_test.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # =============================================================================
    # Print Summary Statistics
    # =============================================================================

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print("Survey Configuration:")
    print(f"  - Type: {survey_type}")
    print(f"  - Line length: {800}m (x = -400 to +400)")
    print(f"  - Electrode spacing: {station_separation}m")
    print(f"  - Number of sources: {survey.nSrc}")
    print(f"  - Number of data: {survey.nD}")

    print("\nModel Properties:")
    print(f"  - Background: {background_resistivity} Ohm-m")
    print(f"  - Conductor: {conductor_resistivity} Ohm-m (at x=-150m)")
    print(f"  - Resistor: {resistor_resistivity} Ohm-m (at x=+150m)")

    print("\nSimulation Results:")
    print(f"  - Voltage range: {dpred.min():.2e} to {dpred.max():.2e} V/A")
    print(
        f"  - Apparent resistivity range: {apparent_resistivity.min():.1f} to {apparent_resistivity.max():.1f} Ohm-m"
    )

    print(
        "\n✅ SUCCESS: Complete DC resistivity forward simulation with pseudosection!"
    )

except Exception as e:
    print(f"\nError in simulation: {e}")
    import traceback

    traceback.print_exc()
