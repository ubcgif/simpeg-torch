"""
Unit tests for FDEM survey.

Tests survey organization by frequency and source tensor generation.
"""

import torch
import unittest
from simpegtorch.discretize import TensorMesh
from simpegtorch.simulation.FDEM.sources import MagneticDipole
from simpegtorch.simulation.FDEM.survey import Survey
from simpegtorch.simulation.FDEM.receivers import (
    PointMagneticFluxDensity as RxPointMagneticFluxDensity,
)


class TestFDEMSurvey(unittest.TestCase):
    """Test FDEM survey functionality."""

    def setUp(self):
        """Create test mesh and sources."""
        # Create mesh
        hx = torch.ones(8) * 10.0
        hy = torch.ones(8) * 10.0
        hz = torch.ones(8) * 10.0
        origin = torch.tensor([0.0, 0.0, 0.0])
        self.mesh = TensorMesh([hx, hy, hz], origin=origin)

        # Create receiver locations
        self.rx_locs = torch.tensor(
            [[20.0, 40.0, 40.0], [40.0, 40.0, 40.0], [60.0, 40.0, 40.0]]
        )

    def test_single_frequency_survey(self):
        """Test survey with sources at a single frequency."""
        freq = 10.0

        # Create sources
        src1 = MagneticDipole(
            [], freq, torch.tensor([30.0, 30.0, 30.0]), moment=1.0, orientation="z"
        )
        src2 = MagneticDipole(
            [], freq, torch.tensor([50.0, 50.0, 50.0]), moment=1.0, orientation="z"
        )

        survey = Survey([src1, src2])

        # Check frequency organization
        self.assertEqual(len(survey.frequencies), 1)
        self.assertEqual(survey.frequencies[0], freq)

        # Check sources by frequency
        sources = survey.get_sources_by_frequency(freq)
        self.assertEqual(len(sources), 2)

    def test_multiple_frequency_survey(self):
        """Test survey with sources at multiple frequencies."""
        freq1 = 10.0
        freq2 = 100.0
        freq3 = 1000.0

        # Create sources at different frequencies
        src1 = MagneticDipole([], freq1, torch.tensor([30.0, 30.0, 30.0]), moment=1.0)
        src2 = MagneticDipole([], freq2, torch.tensor([40.0, 40.0, 40.0]), moment=1.0)
        src3 = MagneticDipole([], freq1, torch.tensor([50.0, 50.0, 50.0]), moment=1.0)
        src4 = MagneticDipole([], freq3, torch.tensor([60.0, 60.0, 60.0]), moment=1.0)

        survey = Survey([src1, src2, src3, src4])

        # Check frequencies are sorted
        self.assertEqual(len(survey.frequencies), 3)
        self.assertEqual(survey.frequencies, [freq1, freq2, freq3])

        # Check sources at each frequency
        sources_freq1 = survey.get_sources_by_frequency(freq1)
        sources_freq2 = survey.get_sources_by_frequency(freq2)
        sources_freq3 = survey.get_sources_by_frequency(freq3)

        self.assertEqual(len(sources_freq1), 2)  # src1 and src3
        self.assertEqual(len(sources_freq2), 1)  # src2
        self.assertEqual(len(sources_freq3), 1)  # src4

    def test_get_source_tensor(self):
        """Test source tensor generation."""
        freq1 = 10.0
        freq2 = 100.0

        # Create sources
        src1 = MagneticDipole([], freq1, torch.tensor([30.0, 30.0, 30.0]), moment=1.0)
        src2 = MagneticDipole([], freq1, torch.tensor([50.0, 50.0, 50.0]), moment=1.0)
        src3 = MagneticDipole([], freq2, torch.tensor([40.0, 40.0, 40.0]), moment=1.0)

        survey = Survey([src1, src2, src3])

        # Get source tensors
        source_dict = survey.get_source_tensor(self.mesh, formulation="EB")

        # Check that we have tensors for each frequency
        self.assertIn(freq1, source_dict)
        self.assertIn(freq2, source_dict)

        # Check shapes
        # freq1 has 2 sources
        self.assertEqual(source_dict[freq1].shape[0], 2)
        self.assertEqual(source_dict[freq1].shape[1], self.mesh.n_faces)

        # freq2 has 1 source
        self.assertEqual(source_dict[freq2].shape[0], 1)
        self.assertEqual(source_dict[freq2].shape[1], self.mesh.n_faces)

        # Check data type
        self.assertTrue(source_dict[freq1].dtype == torch.complex128)
        self.assertTrue(source_dict[freq2].dtype == torch.complex128)

    def test_survey_nD_property(self):
        """Test total number of data points calculation."""
        freq = 10.0

        # Create receivers
        rx1 = RxPointMagneticFluxDensity(
            locations=self.rx_locs[:2], orientation="z", component="real"
        )
        rx2 = RxPointMagneticFluxDensity(
            locations=self.rx_locs, orientation="z", component="real"
        )

        # Create sources with receivers
        src1 = MagneticDipole([rx1], freq, torch.tensor([30.0, 30.0, 30.0]), moment=1.0)
        src2 = MagneticDipole([rx2], freq, torch.tensor([50.0, 50.0, 50.0]), moment=1.0)

        survey = Survey([src1, src2])

        # Total data points: 2 (from rx1) + 3 (from rx2) = 5
        self.assertEqual(survey.nD, 5)

    def test_survey_properties(self):
        """Test various survey properties."""
        freq = 10.0

        rx = RxPointMagneticFluxDensity(
            locations=self.rx_locs, orientation="z", component="real"
        )

        src1 = MagneticDipole([rx], freq, torch.tensor([30.0, 30.0, 30.0]), moment=1.0)
        src2 = MagneticDipole([rx], freq, torch.tensor([50.0, 50.0, 50.0]), moment=1.0)

        survey = Survey([src1, src2])

        # Test n_sources
        self.assertEqual(survey.n_sources, 2)

        # Test n_receivers
        self.assertEqual(survey.n_receivers, 2)  # 2 sources × 1 receiver each

        # Test length
        self.assertEqual(len(survey), 2)

        # Test iteration
        sources = list(survey)
        self.assertEqual(len(sources), 2)

    def test_empty_survey(self):
        """Test empty survey initialization."""
        survey = Survey([])

        self.assertEqual(len(survey.frequencies), 0)
        self.assertEqual(survey.nD, 0)
        self.assertEqual(survey.n_sources, 0)


if __name__ == "__main__":
    unittest.main()
