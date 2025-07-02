import torch

from simpegtorch.discretize import TensorMesh
import unittest
from simpegtorch.electromagnetics.resistivity import (
    DCStaticSimulationNodal,
    DCStaticSimulationCellCentered,
    SrcDipole,
    RxDipole,
    Survey,
)

from simpegtorch.discretize.utils import (
    ndgrid,
)

## Import the same functions from original simpeg
from simpeg.electromagnetics import resistivity as dc
from discretize import TensorMesh as OriginalTensorMesh
from simpeg import maps
import numpy as np


class DCSolutionTest(unittest.TestCase):

    def setUp(self):
        # Create a tensor mesh
        hx = torch.ones(10) * 25
        hy = torch.ones(10) * 25
        hz = torch.ones(10) * 25

        # 250m x 250m x 250m mesh

        # Origin at (-125, -125, -250) to center the mesh
        origin = torch.tensor([-125.0, -125.0, -250.0])

        mesh = TensorMesh(
            [hx, hy, hz],
            origin=origin,
        )

        sigma = torch.ones(mesh.nC) * 1e-2  # Uniform conductivity
        self.sigma = sigma

        # Set up survey parameters for numeric solution
        x = mesh.cell_centers_x[
            (mesh.cell_centers_x > -75.0) & (mesh.cell_centers_x < 75.0)
        ]
        y = mesh.cell_centers_y[
            (mesh.cell_centers_y > -75.0) & (mesh.cell_centers_y < 75.0)
        ]

        M = ndgrid(x - 25.0, y, [0.0])
        N = ndgrid(x + 25.0, y, [0.0])

        # create a dipole dipole survey
        rx = RxDipole(
            locations_m=M,
            locations_n=N,
        )

        loc_a = torch.tensor([-25.0, 0.0, 0.0])
        loc_b = torch.tensor([25.0, 0.0, 0.0])

        src = SrcDipole(
            [rx],
            loc_a,  # location of A
            loc_b,  # location of B
            current=1.0,  # current in Amperes
        )

        survey = Survey([src])

        self.survey_torch = survey
        self.mesh_torch = mesh

        # Create the Simpeg equivalent survey and mesh
        mesh_orig = OriginalTensorMesh(
            [hx.numpy(), hy.numpy(), hz.numpy()],
            origin=origin.numpy(),
        )

        self.mesh_orig = mesh_orig
        rx_orig = dc.receivers.Dipole(
            locations_m=M.numpy(),
            locations_n=N.numpy(),
        )

        src_orig = dc.sources.Dipole(
            [rx_orig],
            loc_a.numpy(),  # location of A
            loc_b.numpy(),  # location of B
            current=1.0,  # current in Amperes
        )

        survey_orig = dc.Survey([src_orig])
        self.survey_orig = survey_orig

    def test_dc_nodal_fields(self, tolerance=0.1):
        # Create Nodal Neumann sim in both simpegtorch and simpeg
        self.setUp()
        simulation_torch = DCStaticSimulationNodal(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Neumann",
        )

        # Create conductivity mapping for original SimPEG
        # Use IdentityMap since we're directly parameterizing conductivity
        # This ensures proper model parameterization and derivatives
        # Alternative mappings could be used for log-conductivity, etc.
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)

        simulation_orig = dc.Simulation3DNodal(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Neumann",
        )

        # Run forward simulation
        # For simpeg-torch: convert sigma (conductivity) to resistivity
        # since dpred expects resistivity model
        resistivity_model = 1.0 / self.sigma
        data_torch = simulation_torch.dpred(resistivity_model)

        # For original SimPEG: use sigma as the model parameter with mapping
        sigma_model = self.sigma.numpy()
        data_orig = simulation_orig.dpred(sigma_model)

        # Compare results
        data_torch_np = data_torch.detach().numpy()

        # Compute relative error
        relative_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print(
            f"Relative error between simpeg-torch and original SimPEG: {relative_error:.6f}"
        )
        print(
            f"SimPEG-torch data range: [{data_torch_np.min():.6e}, {data_torch_np.max():.6e}]"
        )
        print(
            f"Original SimPEG data range: [{data_orig.min():.6e}, {data_orig.max():.6e}]"
        )

        # Assert that the results are close (within tolerance)
        self.assertLess(
            relative_error,
            tolerance,
            f"Results differ by {relative_error:.2%}, which exceeds tolerance of {tolerance:.2%}",
        )

        # Also check individual values are close
        np.testing.assert_allclose(
            data_torch_np,
            data_orig,
            rtol=tolerance,
            err_msg="Individual data values differ beyond tolerance",
        )

    def test_dc_cell_centered_fields(self, tolerance=0.1):
        # Create Cell-Centered Dirichlet sim in both simpegtorch and simpeg
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )

        # Create conductivity mapping for original SimPEG
        # Use IdentityMap since we're directly parameterizing conductivity
        # This ensures proper model parameterization and derivatives
        # Alternative mappings could be used for log-conductivity, etc.
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)

        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        # Run forward simulation
        # For simpeg-torch: convert sigma (conductivity) to resistivity
        # since dpred expects resistivity model
        resistivity_model = 1.0 / self.sigma
        data_torch = simulation_torch.dpred(resistivity_model)

        # For original SimPEG: use sigma as the model parameter with mapping
        sigma_model = self.sigma.numpy()
        data_orig = simulation_orig.dpred(sigma_model)

        # Compare results
        data_torch_np = data_torch.detach().numpy()

        # Compute relative error
        relative_error = np.linalg.norm(data_torch_np - data_orig) / np.linalg.norm(
            data_orig
        )

        print(
            f"Cell-centered relative error between simpeg-torch and original SimPEG: {relative_error:.6f}"
        )
        print(
            f"SimPEG-torch data range: [{data_torch_np.min():.6e}, {data_torch_np.max():.6e}]"
        )
        print(
            f"Original SimPEG data range: [{data_orig.min():.6e}, {data_orig.max():.6e}]"
        )

        # Print all values for comparison
        for i, (val_torch, val_orig) in enumerate(zip(data_torch_np, data_orig)):
            print(
                f"Index {i}: SimPEG-torch = {val_torch:.6e}, Original SimPEG = {val_orig:.6e}, Difference = {abs(val_torch - val_orig):.6e}"
            )

        # Assert that the results are close (within tolerance)
        self.assertLess(
            relative_error,
            tolerance,
            f"Cell-centered results differ by {relative_error:.2%}, which exceeds tolerance of {tolerance:.2%}",
        )

        # Also check individual values are close
        np.testing.assert_allclose(
            data_torch_np,
            data_orig,
            rtol=tolerance,
            err_msg="Cell-centered individual data values differ beyond tolerance",
        )
