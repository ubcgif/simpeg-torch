import torch
import unittest
import numpy as np

# Set default dtype to float64 for numerical precision
torch.set_default_dtype(torch.float64)

from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.resistivity import (
    DC2DCellCentered,
    SrcDipole,
    RxDipole,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings
from simpegtorch.discretize.utils import ndgrid

# Import original simpeg for comparison
from simpeg.electromagnetics import resistivity as dc
from simpeg import maps
from discretize import TensorMesh as OriginalTensorMesh


class DC2DBasicTest(unittest.TestCase):
    """Basic tests for 2.5D DC resistivity simulation."""

    def setUp(self):
        """Set up a simple 2D mesh and survey."""
        # Create a simple 2D mesh
        npad = 5
        cs = 12.5
        hx_spec = [(cs, npad, -1.3), (cs, 30), (cs, npad, 1.3)]
        hy_spec = [(cs, npad, -1.3), (cs, 15)]

        # Create original simpeg mesh first to get exact dimensions
        self.mesh_orig = OriginalTensorMesh([hx_spec, hy_spec], x0="CN")

        # Create torch mesh using the exact h arrays from original mesh
        # to avoid discrepancies in mesh generation
        hx = torch.from_numpy(self.mesh_orig.h[0]).float()
        hy = torch.from_numpy(self.mesh_orig.h[1]).float()
        origin_x = float(self.mesh_orig.origin[0])
        origin_y = float(self.mesh_orig.origin[1])

        mesh = TensorMesh([hx, hy], origin=torch.tensor([origin_x, origin_y]))
        self.mesh_torch = mesh

        # Verify they have the same number of cells
        assert (
            mesh.nC == self.mesh_orig.nC
        ), f"Mesh size mismatch: torch={mesh.nC}, orig={self.mesh_orig.nC}"

        # Uniform halfspace conductivity
        sigma_halfspace = 1e-2
        self.sigma = torch.ones(mesh.nC) * sigma_halfspace

        # Create mappings
        self.sigma_map = mappings.BaseMapping(self.sigma)
        self.resistivity_map = mappings.InverseMapping(self.sigma)

        # Create a simple dipole-dipole survey based on actual mesh extent
        # Get the cell centers that are within a reasonable range
        x_min = float(mesh.cell_centers_x.min())
        x_max = float(mesh.cell_centers_x.max())
        x_range = x_max - x_min
        # Use central 60% of mesh
        x_center = (x_min + x_max) / 2
        x_extent = x_range * 0.3

        x = mesh.cell_centers_x[
            (mesh.cell_centers_x > x_center - x_extent)
            & (mesh.cell_centers_x < x_center + x_extent)
        ]
        M = ndgrid(x, [0.0])
        N = ndgrid(x + 25.0, [0.0])

        rx = RxDipole(
            locations_m=M,
            locations_n=N,
        )

        loc_a = torch.tensor([-50.0, 0.0])
        loc_b = torch.tensor([50.0, 0.0])

        src = SrcDipole([rx], loc_a, loc_b, current=1.0)
        self.survey_torch = Survey([src])

        # Create original simpeg survey
        rx_orig = dc.receivers.Dipole(M.numpy(), N.numpy())
        src_orig = dc.sources.Dipole([rx_orig], loc_a.numpy(), loc_b.numpy())
        self.survey_orig = dc.Survey([src_orig])

    def test_instantiation(self):
        """Test that DC2DCellCentered can be instantiated."""
        pde = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map,
            bc_type="Dirichlet",
        )

        # Check quadrature points were created
        self.assertEqual(len(pde._quad_points), pde.nky)
        self.assertEqual(len(pde._quad_weights), pde.nky)

        # Check that quadrature points are positive
        self.assertTrue(torch.all(pde._quad_points > 0))

        # Check that sum of weights is positive (individual weights can be negative)
        self.assertTrue(pde._quad_weights.sum() > 0)

        print(f"Created 2.5D DC simulation with {pde.nky} wavenumber points")
        print(
            f"Quadrature points range: [{pde._quad_points.min():.4f}, {pde._quad_points.max():.4f}]"
        )
        print(f"Sum of weights: {pde._quad_weights.sum():.4f}")

    def test_system_matrices_shape(self):
        """Test that system matrices have correct shape."""
        pde = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map,
            bc_type="Dirichlet",
            nky=5,
        )

        A = pde.get_system_matrices()

        # Should have shape [n_ky, nC, nC]
        expected_shape = (pde.nky, self.mesh_torch.nC, self.mesh_torch.nC)
        self.assertEqual(A.shape, expected_shape)

        print(f"System matrices shape: {A.shape}")

    def test_rhs_tensors_shape(self):
        """Test that RHS tensors have correct shape."""
        pde = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map,
            bc_type="Dirichlet",
        )

        rhs = pde.get_rhs_tensors()

        # Should have shape [n_sources, nC]
        n_sources = len(self.survey_torch.source_list)
        expected_shape = (n_sources, self.mesh_torch.nC)
        self.assertEqual(rhs.shape, expected_shape)

        print(f"RHS shape: {rhs.shape}")

    def test_forward_simulation(self):
        """Test that forward simulation runs without error."""
        pde = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map,
            bc_type="Dirichlet",
            nky=7,
            verbose=True,
        )

        solver = DirectSolver(pde)
        data = solver.forward()

        # Check data shape
        expected_n_data = sum(
            rx.nD for src in self.survey_torch.source_list for rx in src.receiver_list
        )
        self.assertEqual(data.shape[0], expected_n_data)

        # Check that data is not all zeros or NaN
        self.assertFalse(torch.all(data == 0))
        self.assertFalse(torch.any(torch.isnan(data)))

        print(f"Forward simulation produced {len(data)} data points")
        print(f"Data range: [{data.min():.6e}, {data.max():.6e}]")

    def test_compare_with_original_simpeg(self, tolerance=0.15):
        """
        Compare results with original SimPEG 2.5D implementation.

        Note: Using Dirichlet BC for simplicity since Robin BC not yet implemented.
        Higher tolerance expected since BC handling differs.
        """
        # Run torch implementation with Dirichlet BC
        pde_torch = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,  # Use trapezoidal for reproducibility
            verbose=True,
        )

        solver_torch = DirectSolver(pde_torch)
        data_torch = solver_torch.forward()

        # Run original simpeg with Dirichlet BC
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,
        )

        data_orig = simulation_orig.dpred(self.sigma.numpy())

        # Compare
        data_torch_np = data_torch.detach().numpy()

        relative_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print("\n2.5D Comparison Results:")
        print(f"Relative error: {relative_error:.6f}")
        print(
            f"SimPEG-torch data range: [{data_torch_np.min():.6e}, {data_torch_np.max():.6e}]"
        )
        print(
            f"Original SimPEG data range: [{data_orig.min():.6e}, {data_orig.max():.6e}]"
        )

        # Print first few values for debugging
        print("\nFirst 5 data comparisons:")
        for i in range(min(5, len(data_torch_np))):
            print(
                f"  [{i}] Torch: {data_torch_np[i]:.6e}, SimPEG: {data_orig[i]:.6e}, Diff: {abs(data_torch_np[i] - data_orig[i]):.6e}"
            )

        self.assertLess(
            relative_error,
            tolerance,
            f"Results differ by {relative_error:.2%}, which exceeds tolerance of {tolerance:.2%}",
        )


if __name__ == "__main__":
    unittest.main()
