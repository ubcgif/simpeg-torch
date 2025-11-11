import torch
import unittest
import numpy as np

# Set default dtype to float64 for numerical precision
torch.set_default_dtype(torch.float64)

from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.resistivity import (
    DC2DCellCentered,
    SrcPole,
    SrcDipole,
    RxDipole,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver, mappings

# Import original simpeg for comparison
from simpeg.electromagnetics import resistivity as dc
from simpeg import maps
from discretize import TensorMesh as OriginalTensorMesh


class DC2DComparisonTest(unittest.TestCase):
    """
    1-to-1 comparison tests between SimPEG-torch 2.5D and original SimPEG 2.5D.

    This test ensures identical setup between both implementations to verify
    that the PyTorch version produces the same results as the original.
    """

    def setUp(self):
        """Set up identical meshes and surveys for both implementations."""
        # Use simple uniform mesh to avoid mesh generation discrepancies
        npad = 7
        cs = 25.0  # Cell size
        ncore = 20  # Number of core cells

        # Create padding cells
        hx = [(cs, npad, -1.3), (cs, ncore), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 10)]

        # Create original SimPEG mesh first (this is the reference)
        self.mesh_orig = OriginalTensorMesh([hx, hy], x0="CN")

        print("\nOriginal mesh:")
        print(f"  nC: {self.mesh_orig.nC}")
        print(
            f"  nCx: {self.mesh_orig.shape_cells[0]}, nCy: {self.mesh_orig.shape_cells[1]}"
        )
        x_min, x_max = (
            self.mesh_orig.nodes[:, 0].min(),
            self.mesh_orig.nodes[:, 0].max(),
        )
        y_min, y_max = (
            self.mesh_orig.nodes[:, 1].min(),
            self.mesh_orig.nodes[:, 1].max(),
        )
        print(f"  x extent: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  y extent: [{y_min:.2f}, {y_max:.2f}]")

        # Create torch mesh using exact same h arrays and origin
        hx_torch = torch.from_numpy(self.mesh_orig.h[0]).double()
        hy_torch = torch.from_numpy(self.mesh_orig.h[1]).double()
        origin_torch = torch.from_numpy(self.mesh_orig.origin).double()

        self.mesh_torch = TensorMesh([hx_torch, hy_torch], origin=origin_torch)

        print("\nTorch mesh:")
        print(f"  nC: {self.mesh_torch.nC}")
        print(
            f"  nCx: {self.mesh_torch.shape_cells[0]}, nCy: {self.mesh_torch.shape_cells[1]}"
        )

        # Verify meshes match
        assert self.mesh_torch.nC == self.mesh_orig.nC, "Mesh cell count mismatch"

        # Uniform halfspace conductivity
        sigma_halfspace = 1e-2
        self.sigma_np = np.ones(self.mesh_orig.nC) * sigma_halfspace
        self.sigma_torch = torch.from_numpy(self.sigma_np).double()

        # Create mappings for torch
        self.sigma_map_torch = mappings.BaseMapping(self.sigma_torch)
        self.resistivity_map_torch = mappings.InverseMapping(self.sigma_torch)

        # Create mapping for original simpeg
        self.sigma_map_orig = maps.IdentityMap(nP=self.mesh_orig.nC)

    def _create_pole_dipole_survey(self):
        """Create a simple pole-dipole survey for testing."""
        # Source location (pole at origin)
        a_loc = np.array([0.0, 0.0])

        # Receiver locations (dipole receivers along x-axis)
        # Make sure they're well within the mesh bounds
        x_rx = np.arange(-100, 125, 25.0)

        # M and N locations for dipole receivers (25m spacing)
        M_np = np.c_[x_rx, np.zeros(len(x_rx))]
        N_np = np.c_[x_rx + 25.0, np.zeros(len(x_rx))]

        print("\nSurvey setup:")
        print(f"  Source location: {a_loc}")
        print(f"  Number of receivers: {len(x_rx)}")
        print(
            f"  M locations x-range: [{M_np[:, 0].min():.1f}, {M_np[:, 0].max():.1f}]"
        )
        print(
            f"  N locations x-range: [{N_np[:, 0].min():.1f}, {N_np[:, 0].max():.1f}]"
        )

        # Create torch survey
        M_torch = torch.from_numpy(M_np).double()
        N_torch = torch.from_numpy(N_np).double()
        a_loc_torch = torch.from_numpy(a_loc).double()

        rx_torch = RxDipole(M_torch, N_torch)
        src_torch = SrcPole([rx_torch], a_loc_torch)
        self.survey_torch = Survey([src_torch])

        # Create original simpeg survey
        rx_orig = dc.receivers.Dipole(M_np, N_np)
        src_orig = dc.sources.Pole([rx_orig], a_loc)
        self.survey_orig = dc.Survey([src_orig])

        print(f"  Survey nD: torch={self.survey_torch.nD}, orig={self.survey_orig.nD}")

    def _create_dipole_dipole_survey(self):
        """Create a dipole-dipole survey for testing."""
        # Source locations
        a_loc = np.array([-50.0, 0.0])
        b_loc = np.array([-25.0, 0.0])

        # Receiver locations
        x_rx = np.arange(25.0, 150.0, 25.0)
        M_np = np.c_[x_rx, np.zeros(len(x_rx))]
        N_np = np.c_[x_rx + 25.0, np.zeros(len(x_rx))]

        print("\nSurvey setup (dipole-dipole):")
        print(f"  Source A: {a_loc}, B: {b_loc}")
        print(f"  Number of receivers: {len(x_rx)}")

        # Create torch survey
        M_torch = torch.from_numpy(M_np).double()
        N_torch = torch.from_numpy(N_np).double()
        a_loc_torch = torch.from_numpy(a_loc).double()
        b_loc_torch = torch.from_numpy(b_loc).double()

        rx_torch = RxDipole(M_torch, N_torch)
        src_torch = SrcDipole([rx_torch], a_loc_torch, b_loc_torch)
        self.survey_torch = Survey([src_torch])

        # Create original simpeg survey
        rx_orig = dc.receivers.Dipole(M_np, N_np)
        src_orig = dc.sources.Dipole([rx_orig], a_loc, b_loc)
        self.survey_orig = dc.Survey([src_orig])

    def test_pole_dipole_dirichlet(self, tolerance=0.05):
        """
        Test pole-dipole survey with Dirichlet BC.

        This is the simplest case - homogeneous Dirichlet BC is natural
        for both implementations.
        """
        self._create_pole_dipole_survey()

        # Create torch simulation
        pde_torch = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map_torch,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,  # Use trapezoidal for reproducibility
            verbose=True,
        )

        solver_torch = DirectSolver(pde_torch)
        data_torch = solver_torch.forward()

        # Create original simpeg simulation
        sim_orig = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=self.sigma_map_orig,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,
        )

        data_orig = sim_orig.dpred(self.sigma_np)

        # Compare
        data_torch_np = data_torch.detach().cpu().numpy()

        print(f"\n{'='*70}")
        print("POLE-DIPOLE DIRICHLET BC COMPARISON")
        print(f"{'='*70}")
        print(f"Number of data points: {len(data_torch_np)}")
        print("\nData statistics:")
        print(
            f"  SimPEG-torch: min={data_torch_np.min():.6e}, max={data_torch_np.max():.6e}"
        )
        print(f"  Original:     min={data_orig.min():.6e}, max={data_orig.max():.6e}")

        # Compute errors
        abs_diff = np.abs(data_torch_np - data_orig)
        rel_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print("\nError metrics:")
        print(f"  Relative error (norm): {rel_error:.6e}")
        print(f"  Max absolute diff: {abs_diff.max():.6e}")
        print(f"  Mean absolute diff: {abs_diff.mean():.6e}")

        # Print detailed comparison
        print("\nDetailed comparison (first 10 data points):")
        print(
            f"{'Index':<8} {'SimPEG-torch':<15} {'Original':<15} {'Abs Diff':<15} {'Rel Diff (%)':<15}"
        )
        print(f"{'-'*75}")
        for i in range(min(10, len(data_torch_np))):
            rel_diff_pct = (
                100 * abs_diff[i] / abs(data_orig[i]) if data_orig[i] != 0 else 0
            )
            print(
                f"{i:<8} {data_torch_np[i]:<15.6e} {data_orig[i]:<15.6e} "
                f"{abs_diff[i]:<15.6e} {rel_diff_pct:<15.2f}"
            )

        # Assert tolerance
        self.assertLess(
            rel_error,
            tolerance,
            f"Relative error {rel_error:.2%} exceeds tolerance {tolerance:.2%}",
        )

        # Also check point-wise
        np.testing.assert_allclose(
            data_torch_np,
            data_orig,
            rtol=tolerance,
            err_msg="Point-wise comparison failed",
        )

    def test_dipole_dipole_dirichlet(self, tolerance=0.05):
        """
        Test dipole-dipole survey with Dirichlet BC.
        """
        self._create_dipole_dipole_survey()

        # Create torch simulation
        pde_torch = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map_torch,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,
            verbose=True,
        )

        solver_torch = DirectSolver(pde_torch)
        data_torch = solver_torch.forward()

        # Create original simpeg simulation
        sim_orig = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=self.sigma_map_orig,
            bc_type="Dirichlet",
            nky=11,
            do_trap=True,
        )

        data_orig = sim_orig.dpred(self.sigma_np)

        # Compare
        data_torch_np = data_torch.detach().cpu().numpy()

        print(f"\n{'='*70}")
        print("DIPOLE-DIPOLE DIRICHLET BC COMPARISON")
        print(f"{'='*70}")
        print(f"Number of data points: {len(data_torch_np)}")
        print("\nData statistics:")
        print(
            f"  SimPEG-torch: min={data_torch_np.min():.6e}, max={data_torch_np.max():.6e}"
        )
        print(f"  Original:     min={data_orig.min():.6e}, max={data_orig.max():.6e}")

        # Compute errors
        abs_diff = np.abs(data_torch_np - data_orig)
        rel_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print("\nError metrics:")
        print(f"  Relative error (norm): {rel_error:.6e}")
        print(f"  Max absolute diff: {abs_diff.max():.6e}")
        print(f"  Mean absolute diff: {abs_diff.mean():.6e}")

        # Print detailed comparison
        print("\nDetailed comparison (all data points):")
        print(
            f"{'Index':<8} {'SimPEG-torch':<15} {'Original':<15} {'Abs Diff':<15} {'Rel Diff (%)':<15}"
        )
        print(f"{'-'*75}")
        for i in range(len(data_torch_np)):
            rel_diff_pct = (
                100 * abs_diff[i] / abs(data_orig[i]) if data_orig[i] != 0 else 0
            )
            print(
                f"{i:<8} {data_torch_np[i]:<15.6e} {data_orig[i]:<15.6e} "
                f"{abs_diff[i]:<15.6e} {rel_diff_pct:<15.2f}"
            )

        # Assert tolerance
        self.assertLess(
            rel_error,
            tolerance,
            f"Relative error {rel_error:.2%} exceeds tolerance {tolerance:.2%}",
        )

        # Also check point-wise
        np.testing.assert_allclose(
            data_torch_np,
            data_orig,
            rtol=tolerance,
            err_msg="Point-wise comparison failed",
        )

    def test_pole_dipole_neumann(self, tolerance=0.05):
        """
        Test pole-dipole survey with Neumann BC.

        Note: Neumann BC requires matrix conditioning which may introduce
        small numerical differences.
        """
        self._create_pole_dipole_survey()

        # Create torch simulation
        pde_torch = DC2DCellCentered(
            self.mesh_torch,
            self.survey_torch,
            self.resistivity_map_torch,
            bc_type="Neumann",
            nky=11,
            do_trap=True,
            verbose=True,
        )

        solver_torch = DirectSolver(pde_torch)
        data_torch = solver_torch.forward()

        # Create original simpeg simulation
        sim_orig = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=self.sigma_map_orig,
            bc_type="Neumann",
            nky=11,
            do_trap=True,
        )

        data_orig = sim_orig.dpred(self.sigma_np)

        # Compare
        data_torch_np = data_torch.detach().cpu().numpy()

        print(f"\n{'='*70}")
        print("POLE-DIPOLE NEUMANN BC COMPARISON")
        print(f"{'='*70}")
        print(f"Number of data points: {len(data_torch_np)}")
        print("\nData statistics:")
        print(
            f"  SimPEG-torch: min={data_torch_np.min():.6e}, max={data_torch_np.max():.6e}"
        )
        print(f"  Original:     min={data_orig.min():.6e}, max={data_orig.max():.6e}")

        # Compute errors
        abs_diff = np.abs(data_torch_np - data_orig)
        rel_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print("\nError metrics:")
        print(f"  Relative error (norm): {rel_error:.6e}")
        print(f"  Max absolute diff: {abs_diff.max():.6e}")
        print(f"  Mean absolute diff: {abs_diff.mean():.6e}")

        # Print detailed comparison
        print("\nDetailed comparison (first 10 data points):")
        print(
            f"{'Index':<8} {'SimPEG-torch':<15} {'Original':<15} {'Abs Diff':<15} {'Rel Diff (%)':<15}"
        )
        print(f"{'-'*75}")
        for i in range(min(10, len(data_torch_np))):
            rel_diff_pct = (
                100 * abs_diff[i] / abs(data_orig[i]) if data_orig[i] != 0 else 0
            )
            print(
                f"{i:<8} {data_torch_np[i]:<15.6e} {data_orig[i]:<15.6e} "
                f"{abs_diff[i]:<15.6e} {rel_diff_pct:<15.2f}"
            )

        # Assert tolerance
        self.assertLess(
            rel_error,
            tolerance,
            f"Relative error {rel_error:.2%} exceeds tolerance {tolerance:.2%}",
        )

        # Also check point-wise
        np.testing.assert_allclose(
            data_torch_np,
            data_orig,
            rtol=tolerance,
            err_msg="Point-wise comparison failed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
