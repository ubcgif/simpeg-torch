import torch
from simpegtorch.torchmatsolver.utils import torch_tensor_to_sp
import scipy.sparse as sparse
import unittest

from simpegtorch.discretize import TensorMesh
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

    def test_compare_A_matrices(self, tolerance=1e-8):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        resistivity_model = 1.0 / self.sigma

        D_torch = simulation_torch.Div
        G_torch = simulation_torch.Grad
        MfRhoI_torch = simulation_torch.mesh.get_face_inner_product(
            resistivity_model, invert_matrix=True
        )

        D_orig = simulation_orig.Div
        G_orig = simulation_orig.Grad
        MfRhoI_orig = simulation_orig.mesh.get_face_inner_product(
            1.0 / self.sigma.numpy(), invert_matrix=True
        )

        print(f"\nTorch D: {D_torch.dtype}, {D_torch.shape}")
        print(f"Orig D: {D_orig.dtype}, {D_orig.shape}")
        print(f"Torch G: {G_torch.dtype}, {G_torch.shape}")
        print(f"Orig G: {G_orig.dtype}, {G_orig.shape}")
        print(f"Torch MfRhoI: {MfRhoI_torch.dtype}, {MfRhoI_torch.shape}")
        print(f"Orig MfRhoI: {MfRhoI_orig.dtype}, {MfRhoI_orig.shape}")

        A_torch = D_torch @ MfRhoI_torch @ G_torch
        A_orig = D_orig @ MfRhoI_orig @ G_orig

        A_torch_dense = A_torch.to_dense().numpy()
        A_orig_dense = A_orig.toarray()

        print("\nComparing A Matrices...")
        np.testing.assert_allclose(
            A_torch_dense,
            A_orig_dense,
            rtol=tolerance,
            atol=tolerance,
            err_msg="A matrices are not equal.",
        )
        print("A Matrices are equal.")

    def test_compare_RHS_vectors(self, tolerance=1e-8):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        # Get RHS for simpeg-torch
        b_torch = simulation_torch.getRHS(simulation_torch.survey.source_list[0])

        # Get RHS for simpeg-old
        b_orig = simulation_orig.getRHS().flatten()

        print("\nComparing RHS Vectors...")
        np.testing.assert_allclose(
            b_torch.detach().numpy(),
            b_orig,
            rtol=tolerance,
            atol=tolerance,
            err_msg="RHS vectors are not equal.",
        )
        print("RHS Vectors are equal.")

    def test_compare_potential_fields(self, tolerance=1e-8):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        resistivity_model = 1.0 / self.sigma

        # Get A and b from simpeg-torch
        A_torch = simulation_torch.getA(resistivity_model)
        b_torch = simulation_torch.getRHS(simulation_torch.survey.source_list[0])

        # Get A and b from simpeg-old
        A_orig = simulation_orig.getA(resistivity_model.numpy())
        b_orig = simulation_orig.getRHS().flatten()

        # Solve using scipy.sparse.linalg.spsolve for both
        x_torch = sparse.linalg.spsolve(torch_tensor_to_sp(A_torch), b_torch.numpy())
        x_orig = sparse.linalg.spsolve(A_orig, b_orig)

        print("\nComparing Potential Fields (Solver Output)...")
        np.testing.assert_allclose(
            x_torch,
            x_orig,
            rtol=tolerance,
            atol=tolerance,
            err_msg="Potential fields (solver output) are not equal.",
        )
        print("Potential Fields (Solver Output) are equal.")

    def test_compare_receiver_projections(self, tolerance=1e-7):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        resistivity_model = 1.0 / self.sigma

        # Get potential fields
        fields_torch = simulation_torch.fields(resistivity_model)
        fields_orig = simulation_orig.fields(self.sigma.numpy())

        # Project fields to receivers
        data_torch = simulation_torch.dpred(resistivity_model, f=fields_torch)
        data_orig = simulation_orig.dpred(self.sigma.numpy(), f=fields_orig)

        print("\nComparing Receiver Projections...")
        np.testing.assert_allclose(
            data_torch.detach().numpy(),
            data_orig,
            rtol=tolerance,
            atol=tolerance,
            err_msg="Receiver projections are not equal.",
        )
        print("Receiver Projections are equal.")

    def test_compare_DMfRhoI_matrices(self, tolerance=1e-8):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        resistivity_model = 1.0 / self.sigma

        # Calculate D @ MfRhoI for simpeg-torch
        D_torch = simulation_torch.Div
        MfRhoI_torch = simulation_torch.mesh.get_face_inner_product(
            resistivity_model, invert_matrix=True
        )
        DMfRhoI_torch = D_torch @ MfRhoI_torch
        DMfRhoI_torch_dense = DMfRhoI_torch.to_dense().numpy()

        # Calculate D @ MfRhoI for simpeg
        D_orig = simulation_orig.Div
        MfRhoI_orig = simulation_orig.mesh.get_face_inner_product(
            1.0 / self.sigma.numpy(), invert_matrix=True
        )
        DMfRhoI_orig = D_orig @ MfRhoI_orig
        DMfRhoI_orig_dense = DMfRhoI_orig.toarray()

        print("\nComparing D @ MfRhoI Matrices...")
        np.testing.assert_allclose(
            DMfRhoI_torch_dense,
            DMfRhoI_orig_dense,
            rtol=tolerance,
            atol=tolerance,
            err_msg="D @ MfRhoI matrices are not equal.",
        )
        print("D @ MfRhoI Matrices are equal.")

    def test_compare_MfRhoIG_matrices(self, tolerance=1e-8):
        self.setUp()
        simulation_torch = DCStaticSimulationCellCentered(
            self.mesh_torch,
            survey=self.survey_torch,
            sigma=self.sigma,
            bc_type="Dirichlet",
        )
        sigma_map = maps.IdentityMap(nP=self.mesh_orig.nC)
        simulation_orig = dc.Simulation3DCellCentered(
            self.mesh_orig,
            survey=self.survey_orig,
            sigmaMap=sigma_map,
            bc_type="Dirichlet",
        )

        resistivity_model = 1.0 / self.sigma

        # Calculate MfRhoI @ G for simpeg-torch
        MfRhoI_torch = simulation_torch.mesh.get_face_inner_product(
            resistivity_model, invert_matrix=True
        )
        G_torch = simulation_torch.Grad
        MfRhoIG_torch = MfRhoI_torch @ G_torch
        MfRhoIG_torch_dense = MfRhoIG_torch.to_dense().numpy()

        # Calculate MfRhoI @ G for simpeg
        MfRhoI_orig = simulation_orig.mesh.get_face_inner_product(
            1.0 / self.sigma.numpy(), invert_matrix=True
        )
        G_orig = simulation_orig.Grad
        MfRhoIG_orig = MfRhoI_orig @ G_orig
        MfRhoIG_orig_dense = MfRhoIG_orig.toarray()

        print("\nComparing MfRhoI @ G Matrices...")
        np.testing.assert_allclose(
            MfRhoIG_torch_dense,
            MfRhoIG_orig_dense,
            rtol=tolerance,
            atol=tolerance,
            err_msg="MfRhoI @ G matrices are not equal.",
        )
        print("MfRhoI @ G Matrices are equal.")
