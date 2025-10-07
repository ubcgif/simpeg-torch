"""
Integration tests for FDEM implementation.

Tests the complete FDEM workflow from source to data.
"""

import torch
import unittest
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM import (
    FDEM3DMagneticFluxDensity,
    MagneticDipole,
    RxPointMagneticFluxDensity,
    Survey,
)
from simpegtorch.simulation.base import DirectSolver


class TestFDEMIntegration(unittest.TestCase):
    """Integration tests for FDEM workflow."""

    def setUp(self):
        """Set up test mesh and conductivity model."""
        # Create mesh
        hx = torch.ones(8) * 25.0
        hy = torch.ones(8) * 25.0
        hz = torch.ones(8) * 25.0
        origin = torch.tensor([0.0, 0.0, 0.0])

        self.mesh = TensorMesh([hx, hy, hz], origin=origin)

        # Uniform conductivity
        self.sigma = torch.ones(self.mesh.n_cells, dtype=torch.float64) * 0.01

    def test_single_source_single_frequency(self):
        """Test basic FDEM forward simulation."""
        # Create receiver
        rx_locs = torch.tensor(
            [[50.0, 100.0, 50.0], [100.0, 100.0, 50.0], [150.0, 100.0, 50.0]]
        )
        rx = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )

        # Create source
        src = MagneticDipole(
            [rx],
            frequency=10.0,
            location=torch.tensor([100.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )

        survey = Survey([src])

        # Create PDE
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, self.sigma)

        # Create solver
        solver = DirectSolver(pde)

        # Run forward simulation
        data = solver.forward()

        # Check output
        self.assertEqual(data.shape[0], 3)  # 3 receivers
        self.assertTrue(torch.all(torch.isfinite(data)))
        self.assertTrue(data.dtype == torch.float64)  # Real component

    def test_multiple_sources_single_frequency(self):
        """Test FDEM with multiple sources at same frequency."""
        # Create receivers
        rx_locs = torch.tensor([[50.0, 100.0, 50.0], [150.0, 100.0, 50.0]])

        rx1 = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )
        rx2 = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )

        # Create sources at same frequency
        src1 = MagneticDipole(
            [rx1],
            frequency=10.0,
            location=torch.tensor([75.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )
        src2 = MagneticDipole(
            [rx2],
            frequency=10.0,
            location=torch.tensor([125.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )

        survey = Survey([src1, src2])

        # Create PDE and solver
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, self.sigma)
        solver = DirectSolver(pde)

        # Run forward simulation
        data = solver.forward()

        # Check output: 2 sources × 2 receivers = 4 data points
        self.assertEqual(data.shape[0], 4)
        self.assertTrue(torch.all(torch.isfinite(data)))

    def test_multiple_frequencies(self):
        """Test FDEM with multiple frequencies."""
        # Create receivers
        rx_locs = torch.tensor([[100.0, 100.0, 50.0]])

        rx1 = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )
        rx2 = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )
        rx3 = RxPointMagneticFluxDensity(
            locations=rx_locs, orientation="z", component="real"
        )

        # Create sources at different frequencies
        src1 = MagneticDipole(
            [rx1],
            frequency=1.0,
            location=torch.tensor([100.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )
        src2 = MagneticDipole(
            [rx2],
            frequency=10.0,
            location=torch.tensor([100.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )
        src3 = MagneticDipole(
            [rx3],
            frequency=100.0,
            location=torch.tensor([100.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )

        survey = Survey([src1, src2, src3])

        # Create PDE and solver
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, self.sigma)
        solver = DirectSolver(pde)

        # Run forward simulation
        data = solver.forward()

        # Check output: 3 sources × 1 receiver = 3 data points
        self.assertEqual(data.shape[0], 3)
        self.assertTrue(torch.all(torch.isfinite(data)))

    def test_gradient_computation(self):
        """Test that gradients can be computed through the simulation."""
        # Create receiver
        rx = RxPointMagneticFluxDensity(
            locations=torch.tensor([[100.0, 100.0, 50.0]]),
            orientation="z",
            component="real",
        )

        # Create source
        src = MagneticDipole(
            [rx],
            frequency=10.0,
            location=torch.tensor([100.0, 100.0, 100.0]),
            moment=torch.tensor([0.0, 0.0, 1.0]),
        )

        survey = Survey([src])

        # Create conductivity with gradients enabled
        sigma_with_grad = (
            torch.ones(self.mesh.n_cells, requires_grad=True, dtype=torch.float64)
            * 0.01
        )

        # Create PDE and solver
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, sigma_with_grad)
        solver = DirectSolver(pde)

        # Run forward simulation
        data = solver.forward()

        # Compute loss and backpropagate
        loss = torch.sum(data**2)
        loss = loss.real
        loss.backward()

        # Check gradients
        self.assertIsNotNone(sigma_with_grad.grad)
        self.assertTrue(torch.all(torch.isfinite(sigma_with_grad.grad)))
        self.assertTrue(torch.any(sigma_with_grad.grad != 0))

    def test_system_matrix_shape(self):
        """Test that system matrices have correct shape."""
        # Create minimal survey
        rx = RxPointMagneticFluxDensity(
            locations=torch.tensor([[100.0, 100.0, 50.0]]),
            orientation="z",
            component="real",
        )
        src1 = MagneticDipole(
            [rx], 10.0, torch.tensor([100.0, 100.0, 100.0]), moment=1.0
        )
        src2 = MagneticDipole(
            [rx], 100.0, torch.tensor([100.0, 100.0, 100.0]), moment=1.0
        )

        survey = Survey([src1, src2])
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, self.sigma)

        # Get system matrices
        A = pde.get_system_matrices()

        # Check shape: (n_frequencies, n_faces, n_faces)
        self.assertEqual(A.shape[0], 2)  # 2 frequencies
        self.assertEqual(A.shape[1], self.mesh.n_faces)
        self.assertEqual(A.shape[2], self.mesh.n_faces)
        self.assertTrue(A.is_complex())

    def test_rhs_tensor_shape(self):
        """Test that RHS tensors have correct shape."""
        # Create survey with 3 sources (2 at freq1, 1 at freq2)
        rx = RxPointMagneticFluxDensity(
            locations=torch.tensor([[100.0, 100.0, 50.0]]),
            orientation="z",
            component="real",
        )

        src1 = MagneticDipole(
            [rx], 10.0, torch.tensor([50.0, 100.0, 100.0]), moment=1.0
        )
        src2 = MagneticDipole(
            [rx], 10.0, torch.tensor([150.0, 100.0, 100.0]), moment=1.0
        )
        src3 = MagneticDipole(
            [rx], 100.0, torch.tensor([100.0, 100.0, 100.0]), moment=1.0
        )

        survey = Survey([src1, src2, src3])
        pde = FDEM3DMagneticFluxDensity(self.mesh, survey, self.sigma)

        # Get RHS tensors
        rhs = pde.get_rhs_tensors()

        # Check shape: (n_sources_total, n_grid_points)
        self.assertEqual(rhs.shape[0], 3)  # 3 sources total
        self.assertEqual(rhs.shape[1], self.mesh.n_faces)
        self.assertTrue(rhs.dtype == torch.complex128)


if __name__ == "__main__":
    unittest.main()
