"""
PyTorch Inversion Example using 3D DC Resistivity Survey

This example demonstrates how to use the PyTorch-based inversion framework
with a realistic 3D DC resistivity survey setup.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# SimPEG-Torch imports
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


# Import mapping classes
from simpegtorch.maps import LogMapping


# Import our PyTorch inversion framework
from simpegtorch.inversion import (
    BaseInversion,
    BaseInvProblem,
    BetaSchedule,
    TargetMisfit,
    BetaEstimate_ByEig,
)
from simpegtorch.regularization import WeightedLeastSquares
from simpegtorch.data_misfit import L2DataMisfit

# Set PyTorch settings
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def create_dcr_survey():
    """Create 3D DC resistivity survey and synthetic model"""

    print("Setting up 3D DC resistivity survey...")

    # Define mesh parameters - reasonable size for inversion
    dx = dy = dz = 25.0  # 25m cells
    nx, ny, nz = 40, 20, 20  # 40x20x20 = 16,000 cells

    # Create cell sizes
    hx = torch.full((nx,), dx)
    hy = torch.full((ny,), dy)
    hz = torch.full((nz,), dz)

    origin = torch.tensor([-nx * dx / 2, -ny * dy / 2, -dz * nz])  # 500m below surface

    mesh = TensorMesh([hx, hy, hz], origin=origin, device=device)

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

    print(f"Mesh: {mesh.nC} total cells, {n_active} active cells")

    # Create 3-layer synthetic model
    background_resistivity = 100.0  # 100 Ohm-m
    conductor_resistivity = 10.0  # 10 Ohm-m conductive body
    resistor_resistivity = 1000.0  # 1000 Ohm-m resistive body

    # Create background model
    true_active_model = torch.full(
        (n_active,), background_resistivity, dtype=torch.float64
    )

    # Add conductive body at (-150, 0, -100) with radius 75m
    conductor_center = [-150.0, 0.0, -100.0]
    conductor_radius = 75.0
    active_centers = mesh.cell_centers[active_cells]
    conductor_mask = get_indices_sphere(
        conductor_center, conductor_radius, active_centers
    )
    true_active_model[conductor_mask] = conductor_resistivity

    # Add resistive body at (150, 0, -100) with radius 75m
    resistor_center = [150.0, 0.0, -100.0]
    resistor_radius = 75.0
    resistor_mask = get_indices_sphere(resistor_center, resistor_radius, active_centers)
    true_active_model[resistor_mask] = resistor_resistivity

    print(
        f"True model: {true_active_model.min():.1f} - {true_active_model.max():.1f} Ω⋅m"
    )

    # Create dipole-dipole survey
    electrode_spacing = 25.0
    n_electrodes = 31
    electrodes_x = (
        torch.arange(n_electrodes) * electrode_spacing
        - (n_electrodes - 1) * electrode_spacing / 2
    )
    electrodes_y = torch.zeros(n_electrodes)
    electrodes_z = torch.zeros(n_electrodes)

    electrode_locations = torch.stack([electrodes_x, electrodes_y, electrodes_z], dim=1)

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

    print(f"Survey: {len(sources)} sources, {survey.nD} total data points")
    print(f"Survey line: {electrodes_x[0]:.0f}m to {electrodes_x[-1]:.0f}m")

    return mesh, survey, active_mapping, true_active_model


def generate_synthetic_data(
    mesh, survey, active_mapping, true_active_model, noise_level=0.05
):
    """Generate synthetic observed data with noise"""

    print(f"\nGenerating synthetic data with {noise_level*100:.1f}% noise...")

    # Create simulation
    sim = Simulation3DNodal(mesh, survey=survey)

    # Generate clean data
    true_full_model = active_mapping.forward(true_active_model)
    d_clean = sim.dpred(true_full_model).detach()

    # Add noise
    noise_std = noise_level * torch.abs(d_clean)
    noise = torch.normal(0.0, noise_std)
    d_obs = d_clean + noise

    # Data uncertainties for weighting
    uncertainties = noise_std + 1e-6  # Small floor to avoid division by zero

    print(f"Data range: {d_obs.min():.2e} to {d_obs.max():.2e} V")
    print(
        f"SNR range: {(torch.abs(d_clean) / noise_std).min():.1f} to {(torch.abs(d_clean) / noise_std).max():.1f}"
    )

    return d_obs, uncertainties, d_clean


def run_torch_dcr_inversion():
    """Run PyTorch-based DC resistivity inversion"""

    print("=" * 70)
    print("PyTorch DC Resistivity Inversion Example")
    print("=" * 70)
    print(f"Using device: {device}")

    # Create survey and model
    mesh, survey, active_mapping, true_active_model = create_dcr_survey()

    # Generate synthetic data
    d_obs, uncertainties, d_clean = generate_synthetic_data(
        mesh, survey, active_mapping, true_active_model, noise_level=0.03
    )

    # Create simulation directly (no wrapper needed!)
    simulation = Simulation3DNodal(mesh, survey=survey)

    # Create log resistivity mapping with PyTorch autograd
    log_resistivity_mapping = LogMapping(active_mapping)

    # Data misfit with log resistivity mapping
    dmisfit = L2DataMisfit(
        simulation=simulation,
        data=d_obs,
        weights=1.0 / uncertainties,  # Weight by inverse of uncertainties
        mapping=log_resistivity_mapping,  # Log resistivity mapping with autograd!
        device=device,
    )

    # Regularization - focus on smoothness for DC resistivity
    reg = WeightedLeastSquares(
        mesh=mesh,
        alpha_s=1e-4,  # Small smallness weight
        alpha_x=1.0,  # X-smoothness weight
        alpha_y=1.0,  # Y-smoothness weight
        alpha_z=1.0,  # Z-smoothness weight
        alpha_xx=0.0,  # No second-order terms
        alpha_yy=0.0,
        alpha_zz=0.0,
        reference_model_in_smooth=False,
        device=device,
    )

    # Inverse problem with Adam optimizer
    inv_prob = BaseInvProblem(
        dmisfit=dmisfit,
        reg=reg,
        optimizer_class="Adam",
        optimizer_kwargs={"lr": 0.1},  # Moderate learning rate for log resistivity
        beta=1.0,
        max_iter=200,
    )

    # Directives for inversion control
    beta_est = BetaEstimate_ByEig(beta0_ratio=1e-2)
    beta_schedule = BetaSchedule(cooling_factor=1.2, cooling_rate=20)  # Gentle cooling
    target_misfit = TargetMisfit(chi_factor=0.1)

    directives = [beta_est, beta_schedule, target_misfit]

    # Inversion
    inv = BaseInversion(inv_prob=inv_prob, directives=directives, device=device)

    # Starting model in log space - homogeneous background
    starting_resistivity = 100.0  # Ω⋅m
    m0_log = torch.full(
        (len(true_active_model),), np.log(starting_resistivity), dtype=torch.float64
    )

    print("\nStarting inversion...")
    print(
        f"Starting model: homogeneous {starting_resistivity:.1f} Ω⋅m (log = {m0_log[0]:.2f})"
    )
    print(f"Target data points: {len(d_obs)}")

    # Run inversion (works in log space)
    m_recovered_log = inv.run(m0_log)

    # Convert recovered model back to linear resistivity space for interpretation
    m_recovered = torch.exp(torch.tensor(m_recovered_log))

    # Results summary
    print("\nInversion Results:")
    print(f"Final beta: {inv_prob.beta:.2e}")
    print(f"Final data misfit: {inv.phi_d_history[-1]:.2e}")
    print(f"Final regularization: {inv.phi_m_history[-1]:.2e}")
    print(f"Data RMS: {np.sqrt(inv.phi_d_history[-1] / len(d_obs)):.4f}")
    print(f"Model range: {m_recovered.min():.1f} - {m_recovered.max():.1f} Ω⋅m")

    # Compute final data fit using log mapping
    with torch.no_grad():
        m_log_torch = torch.tensor(m_recovered_log, device=device, dtype=torch.float64)
        d_pred = simulation.dpred(log_resistivity_mapping.forward(m_log_torch))
        rms_error = torch.sqrt(torch.mean((d_obs - d_pred) ** 2)).item()
        print(f"Final data RMS error: {rms_error:.2e} V")
        print(
            f"Log resistivity range: {m_recovered_log.min():.2f} to {m_recovered_log.max():.2f}"
        )
        print(
            f"Linear resistivity range: {m_recovered.min():.1f} to {m_recovered.max():.1f} Ω⋅m"
        )

    # Plot results
    plot_dcr_results(
        mesh,
        active_mapping,
        true_active_model,
        m_recovered,
        d_obs,
        d_clean,
        d_pred.detach(),
        inv,
    )

    return mesh, active_mapping, true_active_model, m_recovered, inv


def plot_dcr_results(
    mesh, active_mapping, m_true, m_recovered, d_obs, d_clean, d_pred, inv
):
    """Plot DC resistivity inversion results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Model comparison - get cell centers for active cells
    active_cells = active_mapping.active_cells
    active_centers = mesh.cell_centers[active_cells]

    # Convert z-coordinates to depth (0 = ground surface, positive = depth)
    # Assuming z=0 is ground surface in the mesh
    x_coords = active_centers[:, 0].cpu().numpy()
    z_coords = active_centers[:, 2].cpu().numpy()
    depth_coords = -z_coords  # Convert z to depth (flip sign)

    # Create slice near y=0 for 2D visualization
    y_mask = torch.abs(active_centers[:, 1]) < 25.0  # Within 25m of y=0

    # Filter coordinates and models for the slice
    x_slice = x_coords[y_mask]
    depth_slice = depth_coords[y_mask]
    m_true_slice = m_true[y_mask].cpu().numpy()
    m_recovered_slice = m_recovered[y_mask].cpu().numpy()

    # Create regular grid for heatmap interpolation
    x_min, x_max = x_slice.min(), x_slice.max()
    depth_min, depth_max = depth_slice.min(), depth_slice.max()

    # Create grid
    xi = np.linspace(x_min, x_max, 100)
    di = np.linspace(depth_min, depth_max, 80)
    Xi, Di = np.meshgrid(xi, di)

    # Interpolate data onto regular grid
    from scipy.interpolate import griddata

    # True model heatmap
    true_interp = griddata(
        (x_slice, depth_slice),
        m_true_slice,
        (Xi, Di),
        method="linear",
        fill_value=np.nan,
    )
    im0 = axes[0, 0].imshow(
        true_interp,
        extent=[x_min, x_max, depth_max, depth_min],
        aspect="auto",
        cmap="RdYlBu_r",
        norm=LogNorm(vmin=10, vmax=1000),
    )
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Depth (m)")
    axes[0, 0].set_title("True Model (Y≈0)")

    # Add electrode positions on surface
    electrode_spacing = 50.0
    n_electrodes = 17
    electrode_x = (
        np.arange(n_electrodes) * electrode_spacing
        - (n_electrodes - 1) * electrode_spacing / 2
    )
    axes[0, 0].plot(
        electrode_x, np.zeros_like(electrode_x), "kv", markersize=6, label="Electrodes"
    )

    # Add depth reference lines
    for depth in [50, 100, 200, 300]:
        if depth <= depth_max:
            axes[0, 0].axhline(
                y=depth, color="white", linestyle="--", alpha=0.5, linewidth=0.5
            )

    plt.colorbar(im0, ax=axes[0, 0], label="Resistivity (Ω⋅m)", shrink=0.8)

    # Recovered model heatmap
    recovered_interp = griddata(
        (x_slice, depth_slice),
        m_recovered_slice,
        (Xi, Di),
        method="linear",
        fill_value=np.nan,
    )
    im1 = axes[0, 1].imshow(
        recovered_interp,
        extent=[x_min, x_max, depth_max, depth_min],
        aspect="auto",
        cmap="RdYlBu_r",
        norm=LogNorm(vmin=10, vmax=1000),
    )
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Depth (m)")
    axes[0, 1].set_title("Recovered Model (Y≈0)")

    # Add electrode positions on surface
    axes[0, 1].plot(
        electrode_x, np.zeros_like(electrode_x), "kv", markersize=6, label="Electrodes"
    )

    # Add depth reference lines
    for depth in [50, 100, 200, 300]:
        if depth <= depth_max:
            axes[0, 1].axhline(
                y=depth, color="white", linestyle="--", alpha=0.5, linewidth=0.5
            )

    plt.colorbar(im1, ax=axes[0, 1], label="Resistivity (Ω⋅m)", shrink=0.8)

    # Model histograms
    axes[0, 2].hist(
        m_true.cpu().numpy(), bins=30, alpha=0.7, label="True model", density=True
    )
    axes[0, 2].hist(
        m_recovered.cpu().numpy(),
        bins=30,
        alpha=0.7,
        label="Recovered model",
        density=True,
    )
    axes[0, 2].set_xlabel("Resistivity (Ω⋅m)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Model Histograms")
    axes[0, 2].legend()
    axes[0, 2].set_xscale("log")

    # Data fit
    data_indices = np.arange(len(d_obs))
    axes[1, 0].plot(
        data_indices,
        d_clean.cpu().numpy(),
        "k-",
        linewidth=1,
        label="True data",
        alpha=0.8,
    )
    axes[1, 0].plot(
        data_indices,
        d_obs.cpu().numpy(),
        "bo",
        markersize=2,
        label="Observed data",
        alpha=0.6,
    )
    axes[1, 0].plot(
        data_indices, d_pred.cpu().numpy(), "r-", linewidth=1, label="Predicted data"
    )
    axes[1, 0].set_xlabel("Data index")
    axes[1, 0].set_ylabel("Voltage (V)")
    axes[1, 0].set_title("Data Fit")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Convergence curves
    iterations = np.arange(len(inv.phi_d_history))
    axes[1, 1].semilogy(iterations, inv.phi_d_history, "b-", label="Data misfit")
    axes[1, 1].semilogy(
        iterations,
        np.array(inv.phi_m_history) * np.array(inv.beta_history),
        "r-",
        label="β × Regularization",
    )
    axes[1, 1].semilogy(iterations, inv.phi_history, "k-", label="Total objective")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Objective function")
    axes[1, 1].set_title("Convergence")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Beta evolution
    axes[1, 2].semilogy(iterations, inv.beta_history, "g-", linewidth=2)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("β (trade-off parameter)")
    axes[1, 2].set_title("Beta Cooling Schedule")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("torch_dcr_inversion_results.png", dpi=150, bbox_inches="tight")
    print("\nResults saved to: torch_dcr_inversion_results.png")
    plt.show()


if __name__ == "__main__":
    # Run the DC resistivity inversion example
    try:
        mesh, active_mapping, m_true, m_recovered, inv = run_torch_dcr_inversion()

        print("\n" + "=" * 70)
        print("SUCCESS: DC Resistivity Inversion with Log Mapping!")
        print("PyTorch autograd handles log transformation automatically!")
        print("=" * 70)

        # Compute some final statistics
        model_rms = np.sqrt(
            np.mean((m_true.cpu().numpy() - m_recovered.cpu().numpy()) ** 2)
        )
        print(f"Final model RMS error: {model_rms:.1f} Ω⋅m")

        # Correlation coefficient
        correlation = np.corrcoef(m_true.cpu().numpy(), m_recovered.cpu().numpy())[0, 1]
        print(f"Model correlation coefficient: {correlation:.3f}")

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback

        traceback.print_exc()
