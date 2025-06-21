"""
Simple DC resistivity example with pseudosection plotting.

Creates a realistic dipole-dipole survey and plots the apparent resistivity pseudosection.
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
from simpegtorch.electromagnetics.utils import (
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
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
print("Creating DC pseudosection example...")

# =============================================================================
# Create Mesh
# =============================================================================

# Define mesh parameters - smaller for faster computation
dx = dy = dz = 25.0  # 25m cells
nx, ny, nz = 40, 20, 20  # 40x20x20 = 16,000 cells

# Create cell sizes
hx = torch.full((nx,), dx)
hy = torch.full((ny,), dy)
hz = torch.full((nz,), dz)

# Create mesh origin
origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -250.0])  # 250m below surface

# Create the tensor mesh
mesh = TensorMesh([hx, hy, hz], origin=origin, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Mesh: {mesh.nC} cells ({nx} x {ny} x {nz})")

# =============================================================================
# Define Model
# =============================================================================

# Create flat topography
topo_xyz = create_flat_topography(
    x_extent=(-500, 500),
    y_extent=(-250, 250),
    elevation=0.0,
    n_points_x=21,
    n_points_y=11,
)

# Get active cells
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = torch.sum(active_cells).item()

# Create mapping
air_resistivity = 1e8
active_mapping = InjectActiveCells(mesh, active_cells, valInactive=air_resistivity)

# Define 3-layer model
background_resistivity = 100.0  # 100 Ohm-m
conductor_resistivity = 10.0  # 10 Ohm-m conductive body
resistor_resistivity = 1000.0  # 1000 Ohm-m resistive body

# Create background model
active_model = torch.full((n_active,), background_resistivity, dtype=torch.float64)

# Add conductive body at (-150, 0, -100) with radius 75m
conductor_center = [-150.0, 0.0, -100.0]
conductor_radius = 75.0
active_centers = mesh.cell_centers[active_cells]
conductor_mask = get_indices_sphere(conductor_center, conductor_radius, active_centers)
active_model[conductor_mask] = conductor_resistivity

# Add resistive body at (150, 0, -100) with radius 75m
resistor_center = [150.0, 0.0, -100.0]
resistor_radius = 75.0
resistor_mask = get_indices_sphere(resistor_center, resistor_radius, active_centers)
active_model[resistor_mask] = resistor_resistivity

print(f"Active model range: {active_model.min():.1f} to {active_model.max():.1f} Ohm-m")

# Convert to conductivity for simulation
active_conductivity = 1.0 / active_model
full_conductivity = active_mapping * active_conductivity

# =============================================================================
# Create Manual Dipole-Dipole Survey
# =============================================================================

# Create a dipole-dipole survey manually for better control
electrode_spacing = 25.0  # 25m spacing
n_electrodes = 17  # 17 electrodes along line
electrodes_x = (
    torch.arange(n_electrodes) * electrode_spacing
    - (n_electrodes - 1) * electrode_spacing / 2
)
electrodes_y = torch.zeros(n_electrodes)
electrodes_z = torch.zeros(n_electrodes)

electrode_locations = torch.stack([electrodes_x, electrodes_y, electrodes_z], dim=1)
print(
    f"Survey line: {n_electrodes} electrodes from {electrodes_x[0]:.0f}m to {electrodes_x[-1]:.0f}m"
)

# Create dipole-dipole survey
sources = []
n_spacings = 6  # Number of receiver spacings per source

for i in range(n_electrodes - 3):  # Need at least 4 electrodes for dipole-dipole
    # Source dipole: electrodes i and i+1
    src_a = electrode_locations[i]
    src_b = electrode_locations[i + 1]

    receivers = []
    for n in range(1, n_spacings + 1):  # n = 1, 2, 3, ...
        # Receiver dipole: electrodes at spacing n from source
        rx_idx_m = i + 1 + n
        rx_idx_n = i + 2 + n

        if rx_idx_n < n_electrodes:  # Check bounds
            rx_m = electrode_locations[rx_idx_m].unsqueeze(0)
            rx_n = electrode_locations[rx_idx_n].unsqueeze(0)

            # Create receiver
            rx = RxDipole(locations_m=rx_m, locations_n=rx_n, data_type="volt")
            receivers.append(rx)

    if receivers:  # Only create source if there are receivers
        # Create source
        src = SrcDipole(
            receiver_list=receivers,
            location_a=src_a,
            location_b=src_b,
            data_type="volt",
        )
        sources.append(src)

# Create survey
survey = Survey(sources)
print(f"Survey: {survey.nSrc} sources, {survey.nD} data points")

# =============================================================================
# Run Forward Simulation
# =============================================================================

try:
    print("\n=== RUNNING FORWARD SIMULATION ===")
    simulation = DCStaticSimulationNodal(mesh, survey=survey)

    # Run forward simulation
    print("Computing forward data...")
    dpred = simulation.dpred(active_mapping * active_model)
    print(f"Predicted data shape: {dpred.shape}")
    print(f"Data range: {dpred.min():.2e} to {dpred.max():.2e} V/A")

    # Calculate apparent resistivities
    from simpegtorch.electromagnetics.utils.static_utils import geometric_factor

    G = geometric_factor(survey, space_type="half space")
    print(f"Geometric factor range: {G.min():.2e} to {G.max():.2e}")

    # Check for issues
    if torch.any(torch.isinf(G)) or torch.any(torch.isnan(G)):
        print("Warning: Geometric factor has inf/nan values")
        print(f"Inf count: {torch.sum(torch.isinf(G))}")
        print(f"NaN count: {torch.sum(torch.isnan(G))}")
        print(f"Zero count: {torch.sum(G == 0)}")

    apparent_resistivity = apparent_resistivity_from_voltage(
        survey, dpred, space_type="half space"
    )
    print(
        f"Apparent resistivity range: {apparent_resistivity.min():.6f} to {apparent_resistivity.max():.6f} Ohm-m"
    )

    # =============================================================================
    # Create Plots
    # =============================================================================

    print("\nCreating plots...")

    # Create figure
    fig = plt.figure(figsize=(15, 8))

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
    ax1.set_title("True Resistivity Model")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")

    # Plot 2: Survey layout
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(
        electrode_locations[:, 0].cpu().detach(), electrode_locations[:, 1].cpu().detach(), c="red", s=30, marker="^"
    )
    ax2.set_title("Survey Layout")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.axis("equal")

    # Plot 3: Apparent resistivity pseudosection
    ax3 = plt.subplot(2, 1, 2)

    # Get pseudo-locations for plotting
    pseudo_locs = pseudo_locations(survey)

    # Plot pseudosection using the plot_pseudosection utility
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
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "../dc_pseudosection_simple.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # =============================================================================
    # Summary
    # =============================================================================

    print("\n" + "=" * 60)
    print("PSEUDOSECTION EXAMPLE SUMMARY")
    print("=" * 60)
    print("Survey Configuration:")
    print(f"  - {n_electrodes} electrodes with {electrode_spacing}m spacing")
    print(f"  - Dipole-dipole array with {n_spacings} spacings")
    print(f"  - {survey.nSrc} sources, {survey.nD} data points")

    print("\nModel:")
    print(f"  - Background: {background_resistivity} Ohm-m")
    print(f"  - Conductor: {conductor_resistivity} Ohm-m at x={conductor_center[0]}m")
    print(f"  - Resistor: {resistor_resistivity} Ohm-m at x={resistor_center[0]}m")

    print("\nResults:")
    print(f"  - Voltage range: {dpred.min():.2e} to {dpred.max():.2e} V/A")
    print(
        f"  - Apparent resistivity: {apparent_resistivity.min():.6f} to {apparent_resistivity.max():.6f} Ohm-m"
    )

    print("\n✅ SUCCESS: DC pseudosection example completed!")

except Exception as e:
    print(f"\nError: {e}")
    import traceback

    traceback.print_exc()
