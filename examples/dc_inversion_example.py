"""
DC Resistivity Inversion Example using New PDE Framework

This example demonstrates a complete inversion workflow with the new
framework that uses automatic differentiation (no wrapper needed):
1. Create synthetic "true" model
2. Generate synthetic data with noise
3. Set up inversion with regularization (direct solver interface)
4. Run inversion with beta cooling
5. Visualize results
"""

import torch
import matplotlib.pyplot as plt
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.resistivity import (
    DC3DCellCentered,
    SrcDipole,
    RxDipole,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings
from simpegtorch.inversion import (
    BaseInversion,
    BaseInvProblem,
    BetaSchedule,
    TargetMisfit,
    BetaEstimate_ByEig,
)
from simpegtorch.data_misfit import L2DataMisfit
from simpegtorch.regularization import Smallness, SmoothnessFirstOrder

# Set default dtype
torch.set_default_dtype(torch.float64)


# ============================================================================
# Combined Regularization (Tikhonov-style)
# ============================================================================
class TikhonovRegularization(torch.nn.Module):
    """
    Combined Tikhonov regularization: smallness + smoothness terms.

    φ_m = α_s||m - m_ref||² + α_x||∇_x m||² + α_y||∇_y m||² + α_z||∇_z m||²
    """

    def __init__(
        self,
        mesh,
        mapping,
        alpha_s=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        reference_model=None,
    ):
        super().__init__()
        self.mesh = mesh
        self.mapping = mapping

        # Smallness term
        self.smallness = Smallness(
            mesh, mapping=mapping, alpha=alpha_s, reference_model=reference_model
        )

        # Smoothness terms
        self.smoothness_x = SmoothnessFirstOrder(
            mesh, mapping=mapping, orientation="x", alpha=alpha_x
        )
        self.smoothness_y = SmoothnessFirstOrder(
            mesh, mapping=mapping, orientation="y", alpha=alpha_y
        )
        self.smoothness_z = SmoothnessFirstOrder(
            mesh, mapping=mapping, orientation="z", alpha=alpha_z
        )

    def forward(self):
        """Compute combined regularization"""
        phi = self.smallness()
        phi = phi + self.smoothness_x()
        phi = phi + self.smoothness_y()
        phi = phi + self.smoothness_z()
        return phi


print("=" * 70)
print("DC Resistivity Inversion with New PDE Framework")
print("=" * 70)

# ============================================================================
# 1. Create mesh
# ============================================================================
print("\n[1/6] Creating mesh...")
hx = torch.ones(20) * 10.0  # 20 cells of 10m
hy = torch.ones(20) * 10.0
hz = torch.ones(10) * 10.0

# 200m x 200m x 100m mesh
origin = torch.tensor([0.0, 0.0, -100.0])  # Start 100m below surface
mesh = TensorMesh([hx, hy, hz], origin=origin)

print(
    f"Mesh: {mesh.n_cells} cells ({mesh.shape_cells[0]}x{mesh.shape_cells[1]}x{mesh.shape_cells[2]})"
)

# ============================================================================
# 2. Create survey
# ============================================================================
print("\n[2/6] Creating survey...")

# Create a dipole-dipole survey with multiple measurements
n_data_points = 15
x_locs = torch.linspace(20.0, 180.0, n_data_points)
y_center = 100.0

# M electrodes (negative voltage terminal)
M_locs = torch.stack(
    [x_locs, torch.full_like(x_locs, y_center), torch.zeros_like(x_locs)], dim=1
)

# N electrodes (positive voltage terminal) - 20m spacing
N_locs = torch.stack(
    [
        x_locs + 20.0,
        torch.full_like(x_locs, y_center),
        torch.zeros_like(x_locs),
    ],
    dim=1,
)

rx = RxDipole(locations_m=M_locs, locations_n=N_locs)

# Dipole source (A-B electrodes) - 40m dipole at center
src = SrcDipole(
    [rx],
    torch.tensor([60.0, 100.0, 0.0]),  # A electrode
    torch.tensor([140.0, 100.0, 0.0]),  # B electrode
    current=1.0,
)

survey = Survey([src])
print(f"Survey: {survey.nD} data points")

# ============================================================================
# 3. Create true model and generate synthetic data
# ============================================================================
print("\n[3/6] Generating synthetic data...")

# True model: background of 100 Ohm-m with a conductive block (10 Ohm-m)
sigma_background = 0.01  # 1/100 S/m = 100 Ohm-m
sigma_true = torch.ones(mesh.n_cells) * sigma_background

# Add a conductive anomaly: centered block
# Find cells in the anomaly region (x: 80-120m, y: 80-120m, z: -60 to -40m)
cell_centers = mesh.cell_centers
anomaly_mask = (
    (cell_centers[:, 0] > 80.0)
    & (cell_centers[:, 0] < 120.0)
    & (cell_centers[:, 1] > 80.0)
    & (cell_centers[:, 1] < 120.0)
    & (cell_centers[:, 2] > -60.0)
    & (cell_centers[:, 2] < -40.0)
)
sigma_true[anomaly_mask] = 10.0  # 10 Ohm-m anomaly

# Create mapping and PDE for forward modeling
sigma_map_true = mappings.BaseMapping(sigma_true)
pde_true = DC3DCellCentered(mesh, survey, sigma_map_true, bc_type="Dirichlet")
solver_true = DirectSolver(pde_true)

# Generate clean data
with torch.no_grad():
    data_clean = solver_true.forward()

# Add noise (5% relative + 1e-6 V floor)
torch.manual_seed(42)  # For reproducibility
noise_level = 0.05
noise_floor = 1e-6
noise = noise_level * torch.abs(data_clean) + noise_floor
noise = noise * torch.randn_like(data_clean)
data_obs = data_clean + noise

print(f"Data range: [{data_obs.min():.3e}, {data_obs.max():.3e}] V")
print(f"SNR: ~{1/noise_level:.0f}:1")

# ============================================================================
# 4. Set up inversion
# ============================================================================
print("\n[4/6] Setting up inversion...")

# Starting model: homogeneous half-space (best guess)
# Use LogMapping to ensure conductivity stays positive during optimization
log_sigma_start = torch.log(torch.ones(mesh.n_cells) * sigma_background)
log_sigma_start.requires_grad = True
sigma_map_inv = mappings.LogMapping(log_sigma_start)

# Create PDE and solver for inversion (no wrapper needed!)
pde_inv = DC3DCellCentered(mesh, survey, sigma_map_inv, bc_type="Dirichlet")
solver_inv = DirectSolver(pde_inv)

# Data misfit with uncertainty weighting
uncertainties = noise_level * torch.abs(data_obs) + noise_floor
dmisfit = L2DataMisfit(solver_inv, data_obs, weights=1.0 / uncertainties)

# Regularization (Tikhonov smooth inversion)
# Note: reference model is in log-space since we're using LogMapping
alpha_s = 1e-4  # Smallness weight
alpha_x = 1.0  # x-derivative weight
alpha_y = 1.0  # y-derivative weight
alpha_z = 1.0  # z-derivative weight

log_sigma_ref = torch.log(torch.ones(mesh.n_cells) * sigma_background)
reg = TikhonovRegularization(
    mesh,
    mapping=sigma_map_inv,
    alpha_s=alpha_s,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    alpha_z=alpha_z,
    reference_model=log_sigma_ref,
)

# Optimizer (Adam works well for geophysical inversions)
optimizer = torch.optim.LBFGS(
    [sigma_map_inv.trainable_parameters],
    lr=1.0,
    max_iter=20,
    line_search_fn="strong_wolfe",
)

# Inverse problem
inv_prob = BaseInvProblem(
    dmisfit, reg, optimizer, beta=1.0, max_iter=50  # Will be estimated
)

# Directives for inversion
directives = [
    BetaEstimate_ByEig(beta0_ratio=1.0),  # Estimate initial beta
    BetaSchedule(cooling_factor=2.0, cooling_rate=3),  # Cool beta every 3 iterations
    TargetMisfit(chi_factor=0.1),  # Stop when misfit ~ # of data points
]

inversion = BaseInversion(inv_prob, directives=directives, device="cpu")

# Display starting model in conductivity space (not log-space)
with torch.no_grad():
    sigma_start_display = sigma_map_inv.forward()
print(
    f"Starting model: {sigma_start_display.mean():.4f} ± {sigma_start_display.std():.4f} S/m"
)
print(
    f"True anomaly contrast: {sigma_true[anomaly_mask].mean() / sigma_background:.1f}x"
)

# ============================================================================
# 5. Run inversion
# ============================================================================
print("\n[5/6] Running inversion...")
print("-" * 70)

# Run inversion (initial model already set in mapping)
recovered_mapping = inversion.run()

# Get recovered model from mapping
with torch.no_grad():
    sigma_recovered = recovered_mapping.forward()

print("-" * 70)
print(f"\nInversion completed in {inversion.iteration} iterations")
print(
    f"Final misfit: φ_d = {inversion.phi_d_history[-1]:.2e} (target: {dmisfit.n_data:.2e})"
)
print(
    f"Recovered model: {sigma_recovered.mean():.4f} ± {sigma_recovered.std():.4f} S/m"
)

# ============================================================================
# 6. Plot results
# ============================================================================
print("\n[6/6] Plotting results...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Convergence curve
ax = axes[0, 0]
ax.semilogy(inversion.phi_d_history, "b-o", label="Data misfit")
ax.semilogy(inversion.phi_m_history, "r-s", label="Model norm")
ax.axhline(dmisfit.n_data, color="k", linestyle="--", label="Target")
ax.set_xlabel("Iteration")
ax.set_ylabel("Misfit")
ax.set_title("Convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Beta schedule
ax = axes[0, 1]
ax.semilogy(inversion.beta_history, "g-d")
ax.set_xlabel("Iteration")
ax.set_ylabel("Beta (trade-off parameter)")
ax.set_title("Beta Cooling Schedule")
ax.grid(True, alpha=0.3)

# Plot 3: Data fit
ax = axes[0, 2]
ax.plot(data_obs.numpy(), data_clean.numpy(), "b.", alpha=0.5, label="Observed")
with torch.no_grad():
    pred_final = solver_inv.forward()
ax.plot(data_obs.numpy(), pred_final.numpy(), "r.", alpha=0.5, label="Predicted")
lims = [data_obs.min().item(), data_obs.max().item()]
ax.plot(lims, lims, "k--", alpha=0.3)
ax.set_xlabel("Observed Data (V)")
ax.set_ylabel("Predicted Data (V)")
ax.set_title("Data Fit")
ax.legend()
ax.grid(True, alpha=0.3)

# Extract middle slice (y = 100m) for visualization
y_slice_idx = mesh.shape_cells[1] // 2
x_coords = mesh.cell_centers_x
z_coords = mesh.cell_centers_z

# Reshape models to grid
true_grid = sigma_true.reshape(mesh.shape_cells)[:, y_slice_idx, :]
recovered_grid = torch.tensor(sigma_recovered).reshape(mesh.shape_cells)[
    :, y_slice_idx, :
]

# Convert to resistivity for plotting (with small epsilon for safety)
epsilon = 1e-10
rho_true = 1.0 / (true_grid.numpy() + epsilon)
rho_recovered = 1.0 / (recovered_grid.numpy() + epsilon)

# Plot 4: True model (resistivity)
ax = axes[1, 0]
im = ax.pcolormesh(
    x_coords.numpy(),
    z_coords.numpy(),
    rho_true.T,
    cmap="jet_r",
    vmin=10,
    vmax=100,
    shading="auto",
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("True Resistivity (Ω⋅m)")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

# Plot 5: Recovered model (resistivity)
ax = axes[1, 1]
im = ax.pcolormesh(
    x_coords.numpy(),
    z_coords.numpy(),
    rho_recovered.T,
    cmap="jet_r",
    vmin=10,
    vmax=100,
    shading="auto",
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Recovered Resistivity (Ω⋅m)")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

# Plot 6: Difference
ax = axes[1, 2]
diff = rho_recovered - rho_true
im = ax.pcolormesh(
    x_coords.numpy(),
    z_coords.numpy(),
    diff.T,
    cmap="RdBu_r",
    vmin=-20,
    vmax=20,
    shading="auto",
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Difference (Ω⋅m)")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("dc_inversion_results.png", dpi=150, bbox_inches="tight")
print("Results saved to: dc_inversion_results.png")
plt.show()

print("\n" + "=" * 70)
print("Inversion complete!")
print("=" * 70)
