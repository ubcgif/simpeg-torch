"""
3D Forward Simulation with simpeg-torch

Recreation of the SimPEG 3D DC resistivity tutorial using simpeg-torch.
This example demonstrates the PyTorch-based implementation for automatic
differentiation and GPU acceleration.

Key differences from original SimPEG tutorial:
- Uses TensorMesh instead of TreeMesh
- Uses PyTorch tensors throughout
- Simplified topography (no complex elevation functions)
- Focus on core DC resistivity forward simulation
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
    pseudo_locations,
    apparent_resistivity_from_voltage,
    convert_survey_3d_to_2d_lines,
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Define Simple Flat Topography
# =============================================================================

# Create flat topography at z=0 using the utility function
topo_xyz = create_flat_topography(
    x_extent=(-2000, 2000),
    y_extent=(-2000, 2000),
    elevation=0.0,
    n_points_x=81,
    n_points_y=81,
)

print(f"Topography points: {topo_xyz.shape[0]}")

# =============================================================================
# Define the Survey using simpeg-torch utilities
# =============================================================================

# Define the parameters for each survey line (simplified from original)
survey_type = "dipole-dipole"
data_type = "volt"
dimension_type = "3D"

# Survey lines: 1 EW line and 2 NS lines (simplified from original 6 lines)
end_locations_list = [
    torch.tensor([-1000.0, 1000.0, 0.0, 0.0]),  # EW line
    torch.tensor([-300.0, -300.0, -1000.0, 1000.0]),  # NS line at x=-300
    torch.tensor([300.0, 300.0, -1000.0, 1000.0]),  # NS line at x=300
]

station_separation = 100.0
num_rx_per_src = 6  # Reduced from 8 for computational efficiency

# Generate source lists for each line
source_list = []
for end_locations in end_locations_list:
    line_sources = generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations,
        topo_xyz,
        num_rx_per_src,
        station_separation,
    )
    source_list.extend(line_sources)

# Create the survey
survey = Survey(source_list)
print(f"Survey contains {survey.nSrc} sources and {survey.nD} data points")

# Skip electrode draping for now - electrodes are already at correct positions
# survey = drape_electrodes_on_topography(survey, topo_xyz)
print("Electrodes are already positioned correctly for flat topography")

# =============================================================================
# Design a TensorMesh (instead of TreeMesh)
# =============================================================================

# Define mesh parameters
dx = 50.0  # Cell size in x
dy = 50.0  # Cell size in y
dz = 25.0  # Cell size in z

# Define mesh extent
nx = 80  # Number of cells in x
ny = 80  # Number of cells in y
nz = 40  # Number of cells in z

# Create cell sizes
hx = torch.full((nx,), dx)
hy = torch.full((ny,), dy)
hz = torch.full((nz,), dz)

# Create mesh origin (center the mesh)
origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -500.0])  # 500m below surface

# Create the tensor mesh
mesh = TensorMesh([hx, hy, hz], origin=origin)
print(f"Mesh: {mesh.nC} cells ({nx} x {ny} x {nz})")

# =============================================================================
# Define Active Cells using topography
# =============================================================================

# Use the topography to define active cells
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = torch.sum(active_cells).item()

# Create mapping from active cells to full mesh
# Inactive cells (air) get high resistivity
air_resistivity = 1e8  # Very high resistivity for air
active_mapping = InjectActiveCells(mesh, active_cells, valInactive=air_resistivity)

# =============================================================================
# Define the Model (on active cells only)
# =============================================================================

# Define resistivity values in Ohm-m (easier to interpret)
background_resistivity = 100.0  # 100 Ohm-m background
conductor_resistivity = 10.0  # 10 Ohm-m conductive sphere
resistor_resistivity = 1000.0  # 1000 Ohm-m resistive sphere

# Create background model on active cells only
active_model = torch.full((n_active,), background_resistivity, dtype=torch.float64)

# Add conductive sphere at (-300, 0, -100) with radius 150m
sphere_center_1 = [-300.0, 0.0, -100.0]
sphere_radius_1 = 150.0
active_centers = mesh.cell_centers[active_cells]
conductor_mask = get_indices_sphere(sphere_center_1, sphere_radius_1, active_centers)
active_model[conductor_mask] = conductor_resistivity

# Add resistive sphere at (300, 0, -100) with radius 150m
sphere_center_2 = [300.0, 0.0, -100.0]
sphere_radius_2 = 150.0
resistor_mask = get_indices_sphere(sphere_center_2, sphere_radius_2, active_centers)
active_model[resistor_mask] = resistor_resistivity

print(f"Active model range: {active_model.min():.1f} to {active_model.max():.1f} Ohm-m")

# Map to full mesh for visualization and simulation
full_resistivity_model = active_mapping * active_model
print(
    f"Full model range: {full_resistivity_model.min():.2e} to {full_resistivity_model.max():.2e} Ohm-m"
)

# =============================================================================
# Create the Forward Simulation
# =============================================================================

# Create the Forward Simulation using Nodal Formulation
try:
    print("\n=== CREATING NODAL DC SIMULATION ===")

    # Create nodal simulation (better for surface DC data)
    # Nodal formulation uses Neumann boundary conditions by default
    simulation = DCStaticSimulationNodal(mesh, survey=survey)
    print(f"Created nodal simulation with {simulation.bc_type} boundary conditions")

    # For nodal formulation, we work with resistivity values on mesh nodes
    # Create a nodal resistivity model similar to the cell-centered one
    node_centers = mesh.gridN  # Node locations
    nodal_resistivity = torch.full(
        (mesh.nN,), background_resistivity, dtype=torch.float64
    )

    # Add conductive sphere to nodes
    conductor_mask_nodal = get_indices_sphere(
        sphere_center_1, sphere_radius_1, node_centers
    )
    nodal_resistivity[conductor_mask_nodal] = conductor_resistivity

    # Add resistive sphere to nodes
    resistor_mask_nodal = get_indices_sphere(
        sphere_center_2, sphere_radius_2, node_centers
    )
    nodal_resistivity[resistor_mask_nodal] = resistor_resistivity

    # Set air nodes (above surface) to high resistivity
    air_mask_nodal = node_centers[:, 2] > 0.0
    nodal_resistivity[air_mask_nodal] = air_resistivity

    print(f"Nodal resistivity model shape: {nodal_resistivity.shape}")
    print(
        f"Nodal model range: {nodal_resistivity.min():.2e} to {nodal_resistivity.max():.2e} Ohm-m"
    )
    print(f"Mesh info: nN={mesh.nN}, nE={mesh.nE}, nF={mesh.nF}, nC={mesh.nC}")

    # For nodal DC, we need conductivity on mesh edges, not resistivity on nodes
    # Let's try using cell-centered conductivity instead
    print("\nTrying cell-centered approach instead...")

    # Convert active model (resistivity) to conductivity
    active_conductivity = 1.0 / active_model

    # Map to full mesh
    full_conductivity = active_mapping * active_conductivity

    print(f"Cell-centered conductivity shape: {full_conductivity.shape}")
    print(
        f"Cell-centered conductivity range: {full_conductivity.min():.2e} to {full_conductivity.max():.2e} S/m"
    )

    # Predict data using nodal formulation with cell-centered conductivity
    print("\nRunning nodal forward simulation with cell-centered model...")
    dpred = simulation.dpred(full_conductivity)
    print(f"Predicted data shape: {dpred.shape}")
    print(f"Data range: {dpred.min():.6e} to {dpred.max():.6e} V/A")

    # Convert to apparent resistivity
    apparent_resistivity = apparent_resistivity_from_voltage(survey, dpred)
    apparent_conductivity = 1.0 / apparent_resistivity
    print(
        f"Apparent conductivity range: {apparent_conductivity.min():.2e} to {apparent_conductivity.max():.2e} S/m"
    )

    # =============================================================================
    # Plot Results
    # =============================================================================

    # Plot the survey layout
    unique_locations = survey.unique_electrode_locations
    if isinstance(unique_locations, torch.Tensor):
        unique_locations = unique_locations.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot electrode locations
    ax1.scatter(unique_locations[:, 0], unique_locations[:, 1], s=8, c="r")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Electrode Locations")
    ax1.grid(True)
    ax1.axis("equal")

    # Plot pseudo-locations for EW line
    pseudo_locs = pseudo_locations(survey)
    if isinstance(pseudo_locs, torch.Tensor):
        pseudo_locs = pseudo_locs.detach().cpu().numpy()

    # Extract EW line data (assuming first line is EW)
    n_lines = len(end_locations_list)
    n_data_per_line = survey.nD // n_lines
    ew_indices = slice(0, n_data_per_line)

    ax2.scatter(pseudo_locs[ew_indices, 0], pseudo_locs[ew_indices, 2], s=8, c="r")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Pseudo-locations (EW Line)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(
        "/Users/tsuchijo/Documents/GIF_Work/simpeg_torch_survey_layout.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # Plot model (cross-section through y=0)
    print("\nPlotting model cross-section...")
    try:
        # Use the full resistivity model we created with active mapping
        # Plot XZ slice through y=0
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        mesh.plot_slice(
            full_resistivity_model,
            normal="Y",
            slice_loc=0.0,
            ax=ax,
            pcolor_opts={"cmap": "RdYlBu_r", "norm": LogNorm()},
            grid=True,
        )
        ax.set_title("Resistivity Model - XZ Slice (Y=0)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        plt.savefig(
            "/Users/tsuchijo/Documents/GIF_Work/simpeg_torch_model_slice.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    except Exception as e:
        print(f"Error plotting model slice: {e}")

    # Plot pseudosection for EW line
    try:
        # Create line IDs
        lineID = torch.hstack(
            [torch.full((n_data_per_line,), i + 1) for i in range(n_lines)]
        )

        # Convert to 2D surveys
        survey_2d_list, index_list = convert_survey_3d_to_2d_lines(
            survey, lineID, data_type="volt", output_indexing=True
        )

        # Get data for first line (EW)
        ew_survey = survey_2d_list[0]
        ew_data = apparent_conductivity[index_list[0]]

        # Plot pseudosection
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        plot_pseudosection(
            ew_survey,
            dobs=ew_data,
            plot_type="contourf",
            ax=ax,
            scale="log",
            cbar_label="S/m",
            contourf_opts={
                "cmap": "RdYlBu_r",
                "levels": 20,
            },
            data_locations=True,  # Show data points
        )
        ax.set_title("Apparent Conductivity - EW Line")
        plt.savefig(
            "/Users/tsuchijo/Documents/GIF_Work/simpeg_torch_pseudosection.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    except Exception as e:
        print(f"Error plotting pseudosection: {e}")

except Exception as e:
    print(f"Error in forward simulation: {e}")
    import traceback

    traceback.print_exc()
