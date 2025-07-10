import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn as nn

# simpeg-torch imports
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity import (
    Simulation3DNodal,
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
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# Define mesh parameters - smaller for faster computation
dx = dy = dz = 25.0  # 25m cells
nx, ny, nz = 40, 20, 20  # 40x20x20 = 16,000 cells

# Create cell sizes
hx = torch.full((nx,), dx)
hy = torch.full((ny,), dy)
hz = torch.full((nz,), dz)

origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -500.0])  # 500m below surface

mesh = TensorMesh(
    [hx, hy, hz],
    origin=origin,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Create flat topography
topo_xyz = create_flat_topography(
    x_extent=(-1000, 1000),
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

# =============================================================================
# Create Manual Dipole-Dipole Survey
# =============================================================================

# Create a dipole-dipole survey manually for better control
electrode_spacing = 50.0  # 25m spacing
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

# =============================================================================
# Generate reference data using the true model
# =============================================================================

# Create full model for forward simulation
true_model = active_mapping.forward(active_model)

# Create simulation and compute reference data
sim = Simulation3DNodal(mesh, survey=survey)
dpred_reference = sim.dpred(true_model)

# Ensure gradients are turned off for dpred_reference
dpred_reference = dpred_reference.detach()

print(f"Survey: {len(sources)} sources, {survey.nD} total data points")
print(
    f"Reference data range: {dpred_reference.min():.2e} to {dpred_reference.max():.2e} V"
)

# =============================================================================
# Set up inversion with PyTorch module
# =============================================================================


class ForwardDCSim(nn.Module):
    def __init__(self, mesh, survey, starting_model, active_mapping):
        super().__init__()
        # Model parameters to optimize
        self.inferred_model = nn.Parameter(starting_model.clone(), requires_grad=True)
        self.starting_model = nn.Parameter(starting_model.clone(), requires_grad=False)
        self.simulation = Simulation3DNodal(mesh, survey=survey)
        self.active_mapping = active_mapping
        self.alpha = 0.0  # Regularization parameter

    def forward(self, dpred_reference):
        # Map active model to full mesh
        full_model = self.active_mapping.forward(self.inferred_model)

        # Compute predicted data
        dpred = self.simulation.dpred(full_model)

        # Data misfit
        loss_data = torch.nn.functional.mse_loss(dpred, dpred_reference)

        # Model regularization (smoothness)
        loss_model = torch.nn.functional.mse_loss(
            self.inferred_model, self.starting_model
        )

        return loss_data + self.alpha * loss_model, loss_data, loss_model


# =============================================================================
# Run inversion with Adam optimizer
# =============================================================================

# Create starting model (homogeneous background)
starting_model = torch.full((n_active,), 100.0, dtype=torch.float64)

# Create forward simulation module
forward_sim = ForwardDCSim(mesh, survey, starting_model, active_mapping)

# Set up Adam optimizer
optimizer = torch.optim.Adam(forward_sim.parameters(), lr=1.0)

# Inversion parameters
n_iterations = 200
loss_history = []
data_loss_history = []
model_loss_history = []

print(f"\nStarting inversion with {n_iterations} iterations...")
print("Iter | Total Loss | Data Loss | Model Loss | Grad Norm | Model Norm")
print("-" * 75)

# Plot the reference data and starting model
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot reference data
ax1 = axes[0]
ax1.plot(dpred_reference.cpu(), "b-", label="Reference Data")
ax1.set_title("Reference Data")
ax1.set_xlabel("Data Index")
ax1.set_ylabel("Voltage (V)")
ax1.legend()
ax1.grid(True)

dpred_prior = sim.dpred(active_mapping.forward(starting_model))

# Plot starting model
ax2 = axes[1]
ax2.plot(dpred_prior.detach().cpu(), "r-", label="Starting Model")
ax2.set_title("Prior data distributions")
ax2.set_xlabel("Data Index")
ax2.set_ylabel("Voltage (V)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

for iteration in range(n_iterations):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    total_loss, data_loss, model_loss = forward_sim(dpred_reference)

    # Backward pass
    total_loss.backward()

    # Update parameters
    optimizer.step()

    # Store loss values
    loss_history.append(total_loss.item())
    data_loss_history.append(data_loss.item())
    model_loss_history.append(model_loss.item())

    # Print progress
    if iteration % 20 == 0 or iteration == n_iterations - 1:
        # Check gradient norms
        grad_norm = torch.norm(forward_sim.inferred_model.grad).item()
        model_norm = torch.norm(forward_sim.inferred_model).item()
        print(
            f"{iteration:4d} | {total_loss.item():.2e} | {data_loss.item():.2e} | {model_loss.item():.2e} | grad_norm: {grad_norm:.2e} | model_norm: {model_norm:.2e}"
        )

print("\nInversion completed!")
print(f"Final data loss: {data_loss_history[-1]:.2e}")
print(f"Final model loss: {model_loss_history[-1]:.2e}")

# =============================================================================
# Plot results
# =============================================================================

# Get final inverted model
inverted_model = forward_sim.inferred_model.detach().clone()

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Loss history
ax1 = axes[0, 0]
iterations = range(len(loss_history))
ax1.semilogy(iterations, loss_history, "b-", label="Total Loss")
ax1.semilogy(iterations, data_loss_history, "r-", label="Data Loss")
ax1.semilogy(iterations, model_loss_history, "g-", label="Model Loss")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.set_title("Inversion Loss History")
ax1.legend()
ax1.grid(True)

# Plot 2: True model cross-section
ax2 = axes[0, 1]
# Create 2D slice at y=0
y_slice_idx = ny // 2
true_model_2d = true_model.reshape(nx, ny, nz)[:, y_slice_idx, :]
x_coords = mesh.nodes_x[:-1] + mesh.h[0] / 2
z_coords = mesh.nodes_z[:-1] + mesh.h[2] / 2
X, Z = torch.meshgrid(x_coords, z_coords, indexing="ij")

im2 = ax2.pcolormesh(
    X.cpu(),
    Z.cpu(),
    true_model_2d.cpu(),
    norm=LogNorm(vmin=10, vmax=1000),
    cmap="viridis",
)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")
ax2.set_title("True Model (y=0 slice)")
ax2.set_aspect("equal")
plt.colorbar(im2, ax=ax2, label="Resistivity (Ω⋅m)")

# Plot 3: Inverted model cross-section
ax3 = axes[1, 0]
inverted_full = active_mapping.forward(inverted_model)
inverted_model_2d = inverted_full.reshape(nx, ny, nz)[:, y_slice_idx, :]

im3 = ax3.pcolormesh(
    X.cpu(),
    Z.cpu(),
    inverted_model_2d.cpu(),
    norm=LogNorm(vmin=10, vmax=1000),
    cmap="viridis",
)
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Z (m)")
ax3.set_title("Inverted Model (y=0 slice)")
ax3.set_aspect("equal")
plt.colorbar(im3, ax=ax3, label="Resistivity (Ω⋅m)")

# Plot 4: Anomaly profile through Y-slices
ax4 = axes[1, 1]
# Create anomaly profile showing most anomalous value for each Y slice
background_rho = 100.0  # Background resistivity

# For true model
true_full = true_model.reshape(nx, ny, nz)
anomaly_profile_true = []
for y_idx in range(ny):
    y_slice = true_full[:, y_idx, :]
    # Find most anomalous value (furthest from background)
    deviation = torch.abs(y_slice - background_rho)
    max_deviation_idx = torch.argmax(deviation)
    most_anomalous_value = y_slice.flatten()[max_deviation_idx]
    anomaly_profile_true.append(most_anomalous_value.item())

# For inverted model
inverted_full = active_mapping.forward(inverted_model)
inverted_full_reshaped = inverted_full.reshape(nx, ny, nz)
anomaly_profile_inv = []
for y_idx in range(ny):
    y_slice = inverted_full_reshaped[:, y_idx, :]
    # Find most anomalous value (furthest from background)
    deviation = torch.abs(y_slice - background_rho)
    max_deviation_idx = torch.argmax(deviation)
    most_anomalous_value = y_slice.flatten()[max_deviation_idx]
    anomaly_profile_inv.append(most_anomalous_value.item())

# Y coordinates for plotting
y_centers = mesh.cell_centers_y.cpu().numpy()

# Plot anomaly profiles
ax4.plot(y_centers, anomaly_profile_true, "b-", linewidth=2, label="True Model")
ax4.plot(y_centers, anomaly_profile_inv, "r--", linewidth=2, label="Inverted Model")
ax4.axhline(y=background_rho, color="k", linestyle=":", alpha=0.5, label="Background")
ax4.set_xlabel("Y (m)")
ax4.set_ylabel("Most Anomalous Resistivity (Ω⋅m)")
ax4.set_title("Anomaly Profile Through Y-Slices")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Data fit comparison
ax5 = axes[1, 2]
final_dpred = forward_sim.simulation.dpred(active_mapping.forward(inverted_model))
ax5.loglog(
    torch.abs(dpred_reference).cpu(),
    torch.abs(final_dpred.detach()).cpu(),
    "bo",
    alpha=0.6,
)
data_min = min(torch.abs(dpred_reference).min(), torch.abs(final_dpred.detach()).min())
data_max = max(torch.abs(dpred_reference).max(), torch.abs(final_dpred.detach()).max())
ax5.loglog([data_min, data_max], [data_min, data_max], "r--", label="Perfect fit")
ax5.set_xlabel("True Data (V)")
ax5.set_ylabel("Predicted Data (V)")
ax5.set_title("Data Fit")
ax5.legend()
ax5.grid(True)

# Plot 6: Model comparison scatter plot
ax6 = axes[0, 2]
ax6.scatter(
    active_model.detach().cpu().numpy(),
    inverted_model.detach().cpu().numpy(),
    alpha=0.5,
)
ax6.plot([10, 1000], [10, 1000], "r--", label="Perfect fit")
ax6.set_xlabel("True Resistivity (Ω⋅m)")
ax6.set_ylabel("Inverted Resistivity (Ω⋅m)")
ax6.set_title("Model Comparison")
ax6.legend()
ax6.grid(True)
ax6.set_xscale("log")
ax6.set_yscale("log")

plt.tight_layout()
plt.show()

# =============================================================================
# Print summary statistics
# =============================================================================

print("INVERSION SUMMARY")
print("=" * 60)
print(f"Number of active cells: {n_active}")
print(f"Number of data points: {survey.nD}")
print(f"Number of iterations: {n_iterations}")
print(
    f"Final data RMS: {torch.sqrt(torch.mean((dpred_reference - final_dpred.detach())**2)):.2e}"
)
print(f"Data fit improvement: {data_loss_history[0]/data_loss_history[-1]:.1f}x")

# Model statistics
true_active = active_model
print(f"\nTrue model range: {true_active.min():.1f} - {true_active.max():.1f} Ω⋅m")
print(
    f"Inverted model range: {inverted_model.min():.1f} - {inverted_model.max():.1f} Ω⋅m"
)
print(
    f"Model RMS error: {torch.sqrt(torch.mean((true_active - inverted_model)**2)):.1f} Ω⋅m"
)
