"""
FDEM Inversion Example using New PDE Framework

This example demonstrates frequency-domain electromagnetic (FDEM) inversion:
1. Create synthetic model with conductive anomaly
2. Generate synthetic data at multiple frequencies
3. Set up and run inversion
4. Visualize results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM import (
    FDEM3DMagneticFluxDensity,
    MagneticDipole,
    RxPointMagneticFluxDensity,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, SimulationWrapper, mappings
from simpegtorch.inversion import (
    BaseInversion,
    BaseInvProblem,
    BetaSchedule,
    TargetMisfit,
    BetaEstimate_ByEig,
)
from simpegtorch.data_misfit import L2DataMisfit
from simpegtorch.regularization import TikhonovRegularization

# Set default dtype
torch.set_default_dtype(torch.float64)

print("=" * 70)
print("FDEM Inversion with New PDE Framework")
print("=" * 70)

# ============================================================================
# 1. Create mesh
# ============================================================================
print("\n[1/5] Creating mesh...")
h = torch.ones(12) * 20.0  # 12 cells of 20m = 240m domain
mesh = TensorMesh([h, h, h], origin=torch.tensor([0.0, 0.0, 0.0]))

print(f"Mesh: {mesh.n_cells} cells ({mesh.shape_cells[0]}x{mesh.shape_cells[1]}x{mesh.shape_cells[2]})")

# ============================================================================
# 2. Create survey with multiple frequencies
# ============================================================================
print("\n[2/5] Creating survey...")

# Receiver locations - measure vertical magnetic field at surface
n_rx = 9
x_rx = torch.linspace(40.0, 200.0, n_rx)
rx_locations = torch.stack(
    [x_rx, torch.full_like(x_rx, 120.0), torch.zeros_like(x_rx)], dim=1
)

rx = RxPointMagneticFluxDensity(
    locations=rx_locations, orientation="z", component="real"
)

# Source - vertical magnetic dipole
src_location = torch.tensor([120.0, 120.0, 60.0])
src_moment = torch.tensor([0.0, 0.0, 1.0])

# Multiple frequencies for better resolution
frequencies = [1.0, 10.0, 100.0]  # Hz
sources = []
for freq in frequencies:
    src = MagneticDipole([rx], freq, src_location, moment=src_moment)
    sources.append(src)

survey = Survey(sources)
print(f"Survey: {len(frequencies)} frequencies, {survey.nD} total data points")
print(f"Frequencies: {frequencies} Hz")

# ============================================================================
# 3. Create true model and generate synthetic data
# ============================================================================
print("\n[3/5] Generating synthetic data...")

# True model: background + conductive block
sigma_background = 0.01  # 100 Ohm-m
sigma_true = torch.ones(mesh.n_cells) * sigma_background

# Conductive anomaly (10 Ohm-m)
cell_centers = mesh.cell_centers
anomaly_mask = (
    (cell_centers[:, 0] > 80.0)
    & (cell_centers[:, 0] < 160.0)
    & (cell_centers[:, 1] > 80.0)
    & (cell_centers[:, 1] < 160.0)
    & (cell_centers[:, 2] > 40.0)
    & (cell_centers[:, 2] < 100.0)
)
sigma_true[anomaly_mask] = 0.1  # 10 Ohm-m

# Generate synthetic data
sigma_map_true = mappings.BaseMapping(sigma_true)
pde_true = FDEM3DMagneticFluxDensity(mesh, survey, sigma_map_true)
solver_true = DirectSolver(pde_true)

with torch.no_grad():
    data_clean = solver_true.forward()

# Add noise (3% relative)
torch.manual_seed(42)
noise_level = 0.03
noise = noise_level * torch.abs(data_clean) * torch.randn_like(data_clean)
data_obs = data_clean + noise

print(f"Data range: [{data_obs.abs().min():.3e}, {data_obs.abs().max():.3e}] T")

# ============================================================================
# 4. Set up inversion
# ============================================================================
print("\n[4/5] Setting up inversion...")

# Starting model: homogeneous
sigma_start = torch.ones(mesh.n_cells, requires_grad=True) * sigma_background
sigma_map_inv = mappings.BaseMapping(sigma_start)

# Create simulation for inversion
pde_inv = FDEM3DMagneticFluxDensity(mesh, survey, sigma_map_inv)
solver_inv = DirectSolver(pde_inv)
simulation = SimulationWrapper(pde_inv, solver_inv)

# Data misfit (using real part only for simplicity)
# Note: For real inversions, you'd use both real and imaginary parts
data_obs_real = data_obs.real
uncertainties = noise_level * torch.abs(data_obs_real)
dmisfit = L2DataMisfit(
    simulation,
    data_obs_real,
    weights=1.0 / torch.clamp(uncertainties, min=1e-12),
)

# Regularization
reg = TikhonovRegularization(
    mesh,
    alpha_s=1e-4,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    reference_model=sigma_background * torch.ones(mesh.n_cells),
)

# Optimizer
optimizer = torch.optim.Adam([sigma_map_inv.trainable_parameters], lr=0.02)

# Inverse problem
inv_prob = BaseInvProblem(dmisfit, reg, optimizer, beta=1.0, max_iter=30)

# Directives
directives = [
    BetaEstimate_ByEig(beta0_ratio=1.0),
    BetaSchedule(cooling_factor=2.0, cooling_rate=3),
    TargetMisfit(chi_factor=1.0),
]

inversion = BaseInversion(inv_prob, directives=directives)

# ============================================================================
# 5. Run inversion
# ============================================================================
print("\n[5/5] Running inversion...")
print("-" * 70)

sigma_recovered = inversion.run(sigma_start)

print("-" * 70)
print(f"\nInversion completed in {inversion.iteration} iterations")
print(f"Final misfit: φ_d = {inversion.phi_d_history[-1]:.2e}")
print(f"Recovered model: {sigma_recovered.mean():.4f} ± {sigma_recovered.std():.4f} S/m")

# ============================================================================
# Plot results
# ============================================================================
print("\nPlotting results...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Convergence
ax = axes[0, 0]
ax.semilogy(inversion.phi_d_history, "b-o", label="Data misfit", markersize=4)
ax.semilogy(inversion.phi_m_history, "r-s", label="Model norm", markersize=4)
ax.axhline(dmisfit.n_data, color="k", linestyle="--", alpha=0.5, label="Target")
ax.set_xlabel("Iteration")
ax.set_ylabel("Misfit")
ax.set_title("Convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Data fit for each frequency
ax = axes[0, 1]
pred_final = simulation.dpred().real
n_rx_per_freq = len(rx_locations)
for i, freq in enumerate(frequencies):
    start_idx = i * n_rx_per_freq
    end_idx = start_idx + n_rx_per_freq
    obs_freq = data_obs_real[start_idx:end_idx].numpy()
    pred_freq = pred_final[start_idx:end_idx].detach().numpy()
    ax.plot(x_rx.numpy(), obs_freq, "o", label=f"{freq} Hz (obs)", alpha=0.6)
    ax.plot(x_rx.numpy(), pred_freq, "-", label=f"{freq} Hz (pred)")
ax.set_xlabel("X position (m)")
ax.set_ylabel("B_z (T)")
ax.set_title("Data Fit vs Position")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Extract middle slice for visualization
y_slice_idx = mesh.shape_cells[1] // 2
x_coords = mesh.cell_centers_x
z_coords = mesh.cell_centers_z

true_grid = sigma_true.reshape(mesh.shape_cells)[:, y_slice_idx, :]
recovered_grid = (
    torch.tensor(sigma_recovered).reshape(mesh.shape_cells)[:, y_slice_idx, :]
)

# Convert to resistivity
rho_true = 1.0 / true_grid.numpy()
rho_recovered = 1.0 / recovered_grid.numpy()

# Plot 3: True model
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
plt.colorbar(im, ax=ax, label="ρ (Ω⋅m)")

# Plot 4: Recovered model
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
plt.colorbar(im, ax=ax, label="ρ (Ω⋅m)")

plt.tight_layout()
plt.savefig("fdem_inversion_results.png", dpi=150, bbox_inches="tight")
print("Results saved to: fdem_inversion_results.png")
plt.show()

print("\n" + "=" * 70)
print("FDEM Inversion complete!")
print("=" * 70)
