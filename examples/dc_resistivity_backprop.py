"""
3D DC Resistivity Inversion Example using PyTorch Backpropagation

This example demonstrates:
1. Creating a 3D tensor mesh
2. Setting up a forward model with synthetic data
3. Running an inversion using backpropagation to recover resistivity
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from simpegtorch.discretize import TensorMesh
from simpegtorch.electromagnetics.resistivity.simulation import (
    DCStaticSimulationCellCentered,
)


class Source:
    """Simple point source for DC resistivity."""

    def __init__(self, cell, current):
        self.cell = cell
        self.current = current


class Receiver:
    """Simple point receiver for DC resistivity."""

    def __init__(self, cell):
        self.cell = cell


def create_3d_mesh():
    """Create a 3D tensor mesh for the resistivity problem."""
    # Define cell sizes
    hx = [(10.0, 5), (5.0, 10), (10.0, 5)]  # x-direction: coarse-fine-coarse
    hy = [(10.0, 5), (5.0, 10), (10.0, 5)]  # y-direction: coarse-fine-coarse
    hz = [(10.0, 3), (5.0, 8), (10.0, 3)]  # z-direction: coarse-fine-coarse

    # Create mesh
    mesh = TensorMesh([hx, hy, hz], origin="CCC")

    print("Mesh created:")
    print(f"  Cells: {mesh.nC}")
    print(f"  Nodes: {mesh.nN}")
    print(f"  Faces: {mesh.nF}")
    print(f"  Edges: {mesh.nE}")

    return mesh


def create_true_resistivity_model(mesh):
    """Create a synthetic true resistivity model with anomalies."""
    # Background resistivity (ohm-m)
    rho_background = 100.0
    rho_true = torch.full((mesh.nC,), rho_background, dtype=torch.float64)

    # Add a conductive anomaly (lower resistivity)
    cc = mesh.cell_centers
    anomaly_center = torch.tensor([0.0, 0.0, -50.0], dtype=torch.float64)
    radius = 40.0

    # Find cells within the anomaly
    distances = torch.norm(cc - anomaly_center, dim=1)
    anomaly_mask = distances <= radius
    rho_true[anomaly_mask] = 20.0  # Conductive anomaly

    # Add a resistive anomaly
    anomaly_center2 = torch.tensor([60.0, 60.0, -30.0], dtype=torch.float64)
    distances2 = torch.norm(cc - anomaly_center2, dim=1)
    anomaly_mask2 = distances2 <= 30.0
    rho_true[anomaly_mask2] = 500.0  # Resistive anomaly

    return rho_true


def setup_sources_and_receivers(mesh):
    """Setup source and receiver configurations."""
    sources = []
    receivers = []

    # Get surface cells (z closest to 0)
    cc = mesh.cell_centers
    surface_mask = torch.abs(cc[:, 2]) < 20.0  # Near surface
    surface_indices = torch.where(surface_mask)[0]

    if len(surface_indices) == 0:
        # Fallback: use cells closest to surface
        surface_indices = torch.argsort(torch.abs(cc[:, 2]))[:20]

    # Create current sources (bipole)
    source_positive = Source(surface_indices[0].item(), 1.0)  # +1 A
    source_negative = Source(surface_indices[-1].item(), -1.0)  # -1 A
    sources = [source_positive, source_negative]

    # Create receivers at surface
    for i in range(0, min(10, len(surface_indices)), 2):
        receiver = Receiver(surface_indices[i].item())
        receivers.append(receiver)

    print(f"Created {len(sources)} sources and {len(receivers)} receivers")
    return sources, receivers


def add_noise(data, noise_level=0.05):
    """Add Gaussian noise to synthetic data."""
    noise = torch.randn_like(data) * noise_level * torch.std(data)
    return data + noise


def plot_3d_models(mesh, rho_true, rho_recovered, losses):
    """Create 3D plots of true and recovered models."""
    # Convert resistivity to conductivity (S/m)
    cond_true = 1.0 / rho_true
    cond_recovered = 1.0 / rho_recovered

    # Create comprehensive plots
    plt.figure(figsize=(20, 15))

    # 1. Convergence plot
    plt.subplot(3, 4, 1)
    plt.semilogy(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Inversion Convergence")
    plt.grid(True)

    # 2. Crossplot
    plt.subplot(3, 4, 2)
    plt.scatter(rho_true.numpy(), rho_recovered.numpy(), alpha=0.6, s=20)
    plt.plot([rho_true.min(), rho_true.max()], [rho_true.min(), rho_true.max()], "r--")
    plt.xlabel("True Resistivity (Ω⋅m)")
    plt.ylabel("Recovered Resistivity (Ω⋅m)")
    plt.title("True vs Recovered")
    plt.grid(True)

    # 3-5. True conductivity model slices
    ax3 = plt.subplot(3, 4, 3)
    mesh.plot_slice(
        cond_true,
        normal="Z",
        ind=mesh.shape_cells[2] // 2,
        ax=ax3,
        pcolor_opts={"cmap": "viridis"},
    )
    ax3.set_title("True Conductivity\n(XY slice)")

    ax4 = plt.subplot(3, 4, 4)
    mesh.plot_slice(
        cond_true,
        normal="Y",
        ind=mesh.shape_cells[1] // 2,
        ax=ax4,
        pcolor_opts={"cmap": "viridis"},
    )
    ax4.set_title("True Conductivity\n(XZ slice)")

    # 6-8. Recovered conductivity model slices
    ax5 = plt.subplot(3, 4, 7)
    mesh.plot_slice(
        cond_recovered,
        normal="Z",
        ind=mesh.shape_cells[2] // 2,
        ax=ax5,
        pcolor_opts={"cmap": "viridis"},
    )
    ax5.set_title("Recovered Conductivity\n(XY slice)")

    ax6 = plt.subplot(3, 4, 8)
    mesh.plot_slice(
        cond_recovered,
        normal="Y",
        ind=mesh.shape_cells[1] // 2,
        ax=ax6,
        pcolor_opts={"cmap": "viridis"},
    )
    ax6.set_title("Recovered Conductivity\n(XZ slice)")

    # 9-10. Difference (residual) plots
    cond_diff = cond_recovered - cond_true

    ax7 = plt.subplot(3, 4, 11)
    mesh.plot_slice(
        cond_diff,
        normal="Z",
        ind=mesh.shape_cells[2] // 2,
        ax=ax7,
        pcolor_opts={"cmap": "RdBu_r"},
    )
    ax7.set_title("Conductivity Difference\n(XY slice)")

    ax8 = plt.subplot(3, 4, 12)
    mesh.plot_slice(
        cond_diff,
        normal="Y",
        ind=mesh.shape_cells[1] // 2,
        ax=ax8,
        pcolor_opts={"cmap": "RdBu_r"},
    )
    ax8.set_title("Conductivity Difference\n(XZ slice)")

    # Statistics subplot
    ax9 = plt.subplot(3, 4, 5)
    ax9.axis("off")
    stats_text = f"""Model Statistics:

True Conductivity:
  Min: {cond_true.min():.4f} S/m
  Max: {cond_true.max():.4f} S/m
  Mean: {cond_true.mean():.4f} S/m

Recovered Conductivity:
  Min: {cond_recovered.min():.4f} S/m
  Max: {cond_recovered.max():.4f} S/m
  Mean: {cond_recovered.mean():.4f} S/m

RMS Error: {torch.sqrt(torch.mean(cond_diff**2)):.6f} S/m
Data Fit: {(1 - torch.var(cond_diff)/torch.var(cond_true))*100:.1f}%"""
    ax9.text(
        0.05,
        0.95,
        stats_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Histogram comparison
    plt.subplot(3, 4, 6)
    plt.hist(cond_true.numpy(), bins=30, alpha=0.5, label="True", density=True)
    plt.hist(
        cond_recovered.numpy(), bins=30, alpha=0.5, label="Recovered", density=True
    )
    plt.xlabel("Conductivity (S/m)")
    plt.ylabel("Density")
    plt.title("Conductivity Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Additional slice views
    ax11 = plt.subplot(3, 4, 9)
    mesh.plot_slice(
        cond_true,
        normal="X",
        ind=mesh.shape_cells[0] // 2,
        ax=ax11,
        pcolor_opts={"cmap": "viridis"},
    )
    ax11.set_title("True Conductivity\n(YZ slice)")

    ax12 = plt.subplot(3, 4, 10)
    mesh.plot_slice(
        cond_recovered,
        normal="X",
        ind=mesh.shape_cells[0] // 2,
        ax=ax12,
        pcolor_opts={"cmap": "viridis"},
    )
    ax12.set_title("Recovered Conductivity\n(YZ slice)")

    plt.tight_layout()
    # plt.savefig('dc_resistivity_3d_inversion_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Additional 3D visualization using plot_image for multiple slices
    print("\nCreating additional 3D slice visualizations...")

    # True model 3D slices
    fig2, ax_true = plt.subplots(figsize=(15, 5))
    mesh.plot_image(cond_true, ax=ax_true, pcolor_opts={"cmap": "viridis"})
    plt.suptitle("True Conductivity Model - All Slices", fontsize=16)
    # plt.savefig('true_conductivity_3d_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Recovered model 3D slices
    fig3, ax_rec = plt.subplots(figsize=(15, 5))
    mesh.plot_image(cond_recovered, ax=ax_rec, pcolor_opts={"cmap": "viridis"})
    plt.suptitle("Recovered Conductivity Model - All Slices", fontsize=16)
    # plt.savefig('recovered_conductivity_3d_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("3D plots saved:")
    print("  - dc_resistivity_3d_inversion_results.png (comprehensive view)")
    print("  - true_conductivity_3d_slices.png (true model slices)")
    print("  - recovered_conductivity_3d_slices.png (recovered model slices)")


def forward_simulation_example():
    """Example 1: Forward simulation with true model."""
    print("\n=== Forward Simulation Example ===")

    # Create mesh and true model
    mesh = create_3d_mesh()
    rho_true = create_true_resistivity_model(mesh)

    # Setup sources and receivers
    sources, receivers = setup_sources_and_receivers(mesh)

    # Create simulation
    simulation = DCStaticSimulationCellCentered(mesh, sources, receivers)
    simulation.setBC()

    # Compute forward data
    print("Computing forward simulation...")
    fields = simulation.fields(rho_true)

    # Extract data at receiver locations
    data_true = fields[torch.tensor([r.cell for r in receivers])]

    print("Forward simulation completed")
    print(f"Field range: [{fields.min().item():.3f}, {fields.max().item():.3f}]")
    print(f"Data range: [{data_true.min().item():.3f}, {data_true.max().item():.3f}]")

    return mesh, rho_true, sources, receivers, data_true


def inversion_example():
    """Example 2: Inversion using backpropagation."""
    print("\n=== Inversion Example ===")

    # Get forward simulation results
    mesh, rho_true, sources, receivers, data_true = forward_simulation_example()

    # Add noise to synthetic data
    data_obs = add_noise(data_true, noise_level=0.05)
    print("Added 5% noise to synthetic data")

    # Initialize starting resistivity model (homogeneous)
    rho_init = torch.full((mesh.nC,), 100.0, dtype=torch.float64, requires_grad=True)

    # Create simulation for inversion
    simulation = DCStaticSimulationCellCentered(mesh, sources, receivers)
    simulation.setBC()

    # Setup optimization
    optimizer = optim.Adam([rho_init], lr=10.0)

    # Inversion parameters
    n_iterations = 50
    losses = []

    print(f"Starting inversion with {n_iterations} iterations...")

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Forward simulation
        fields_pred = simulation.fields(rho_init)
        data_pred = fields_pred[torch.tensor([r.cell for r in receivers])]

        # L2 data misfit loss
        data_misfit = torch.sum((data_pred - data_obs) ** 2)

        # Optional: Add smoothness regularization
        alpha_smooth = 0.01
        smoothness = torch.sum(torch.diff(rho_init) ** 2)

        # Total loss
        loss = data_misfit + alpha_smooth * smoothness

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Ensure positive resistivity
        with torch.no_grad():
            rho_init.clamp_(min=1.0)

        losses.append(loss.item())

        if i % 10 == 0:
            print(
                f"  Iteration {i:3d}: Loss = {loss.item():.6f}, "
                f"Data misfit = {data_misfit.item():.6f}"
            )

    print(f"Inversion completed after {n_iterations} iterations")
    print(f"Final loss: {losses[-1]:.6f}")

    # Compare results
    rho_recovered = rho_init.detach()

    print("\nModel comparison:")
    print(
        f"True model - min: {rho_true.min():.1f}, max: {rho_true.max():.1f}, mean: {rho_true.mean():.1f}"
    )
    print(
        f"Recovered  - min: {rho_recovered.min():.1f}, max: {rho_recovered.max():.1f}, mean: {rho_recovered.mean():.1f}"
    )

    # Create 3D plots using mesh plotting capabilities
    plot_3d_models(mesh, rho_true, rho_recovered, losses)

    return mesh, rho_true, rho_recovered, losses


if __name__ == "__main__":
    print("DC Resistivity 3D Inversion Example")
    print("=" * 40)

    # Run forward simulation example
    mesh, rho_true, sources, receivers, data_true = forward_simulation_example()

    # Run inversion example
    mesh, rho_true, rho_recovered, losses = inversion_example()

    print("\nExample completed successfully!")
    print("3D visualization results saved to multiple PNG files")
