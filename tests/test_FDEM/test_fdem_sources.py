"""
Unit tests for FDEM sources.

Tests individual source types to ensure they generate correct source terms.
"""

import torch
import unittest
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM.sources import (
    MagneticDipole,
    ElectricDipole,
    LoopSource,
)


class TestFDEMSources(unittest.TestCase):
    """Test FDEM source implementations."""

    def setUp(self):
        """Create a simple test mesh."""
        # Small mesh for testing
        hx = torch.ones(8) * 10.0
        hy = torch.ones(8) * 10.0
        hz = torch.ones(8) * 10.0
        origin = torch.tensor([0.0, 0.0, 0.0])

        self.mesh = TensorMesh([hx, hy, hz], origin=origin)
        self.frequency = 10.0

    def test_magnetic_dipole_eb_formulation(self):
        """Test MagneticDipole source with EB formulation."""
        # Create source at mesh center
        location = torch.tensor([40.0, 40.0, 40.0])
        moment = torch.tensor([0.0, 0.0, 1.0])  # Vertical dipole

        src = MagneticDipole(
            receiver_list=[],
            frequency=self.frequency,
            location=location,
            moment=moment,
        )

        # Evaluate source terms
        s_m, s_e = src.evaluate(self.mesh, formulation="EB")

        # Check shapes
        self.assertEqual(s_m.shape, (self.mesh.n_faces,))
        self.assertEqual(s_e.shape, (self.mesh.n_edges,))

        # Check data types
        self.assertTrue(s_m.dtype == torch.complex128)
        self.assertTrue(s_e.dtype == torch.complex128)

        # Check that magnetic source is non-zero
        self.assertTrue(torch.any(s_m != 0))

        # Check that electric source is zero (magnetic dipole)
        self.assertTrue(torch.all(s_e == 0))

    def test_magnetic_dipole_hj_formulation(self):
        """Test MagneticDipole source with HJ formulation."""
        location = torch.tensor([40.0, 40.0, 40.0])
        moment = torch.tensor([0.0, 0.0, 1.0])

        src = MagneticDipole(
            receiver_list=[],
            frequency=self.frequency,
            location=location,
            moment=moment,
        )

        # Evaluate source terms
        s_m, s_e = src.evaluate(self.mesh, formulation="HJ")

        # Check shapes (HJ formulation: H on edges, J on faces)
        self.assertEqual(s_m.shape, (self.mesh.n_edges,))
        self.assertEqual(s_e.shape, (self.mesh.n_faces,))

        # Check that magnetic source is non-zero
        self.assertTrue(torch.any(s_m != 0))

        # Check that electric source is zero
        self.assertTrue(torch.all(s_e == 0))

    def test_electric_dipole_eb_formulation(self):
        """Test ElectricDipole source with EB formulation."""
        location = torch.tensor([40.0, 40.0, 40.0])

        src = ElectricDipole(
            receiver_list=[],
            frequency=self.frequency,
            location=location,
            current=1.0,
            length=10.0,
            orientation="z",
        )

        # Evaluate source terms
        s_m, s_e = src.evaluate(self.mesh, formulation="EB")

        # Check shapes
        self.assertEqual(s_m.shape, (self.mesh.n_faces,))
        self.assertEqual(s_e.shape, (self.mesh.n_edges,))

        # Check that electric source is non-zero
        self.assertTrue(torch.any(s_e != 0))

        # Check that magnetic source is zero (electric dipole)
        self.assertTrue(torch.all(s_m == 0))

        # Check source magnitude (current * length)
        self.assertTrue(torch.max(torch.abs(s_e)) > 0)

    def test_loop_source_eb_formulation(self):
        """Test LoopSource with EB formulation."""
        location = torch.tensor([40.0, 40.0, 40.0])

        src = LoopSource(
            receiver_list=[],
            frequency=self.frequency,
            location=location,
            radius=5.0,
            current=1.0,
            orientation="z",
            n_segments=8,
        )

        # Evaluate source terms
        s_m, s_e = src.evaluate(self.mesh, formulation="EB")

        # Check shapes
        self.assertEqual(s_m.shape, (self.mesh.n_faces,))
        self.assertEqual(s_e.shape, (self.mesh.n_edges,))

        # Check that electric source is non-zero (loop is electric current)
        self.assertTrue(torch.any(s_e != 0))

        # Check that magnetic source is zero
        self.assertTrue(torch.all(s_m == 0))

    def test_source_frequency_property(self):
        """Test that source frequency is correctly stored."""
        location = torch.tensor([40.0, 40.0, 40.0])
        freq = 25.0

        src = MagneticDipole(
            receiver_list=[],
            frequency=freq,
            location=location,
            moment=1.0,
            orientation="z",
        )

        self.assertEqual(src.frequency, freq)

    def test_source_location_conversion(self):
        """Test that source location is correctly converted to tensor."""
        # Test with list input
        location_list = [10.0, 20.0, 30.0]
        src1 = MagneticDipole(
            receiver_list=[],
            frequency=self.frequency,
            location=location_list,
            moment=1.0,
        )
        self.assertTrue(torch.is_tensor(src1.location))
        self.assertTrue(
            torch.allclose(
                src1.location, torch.tensor(location_list, dtype=torch.float64)
            )
        )

        # Test with tensor input
        location_tensor = torch.tensor([15.0, 25.0, 35.0], dtype=torch.float32)
        src2 = MagneticDipole(
            receiver_list=[],
            frequency=self.frequency,
            location=location_tensor,
            moment=1.0,
        )
        self.assertTrue(torch.is_tensor(src2.location))
        self.assertEqual(src2.location.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
