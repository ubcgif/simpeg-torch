"""
Test suite for data_misfit module in SimPEG-Torch

Tests for various data misfit classes including L2, L1, Huber, and combination misfits.
"""

import pytest
import torch
import numpy as np

from simpegtorch.data_misfit import (
    BaseDataMisfit,
    L2DataMisfit,
    L1DataMisfit,
    HuberDataMisfit,
)


class MockSolver:
    """Mock solver for testing data misfit classes"""

    def __init__(self, predicted_data):
        self.predicted_data = predicted_data

    def forward(self):
        """Return predicted data"""
        return self.predicted_data


@pytest.fixture
def test_data():
    """Create test data and model"""
    data_obs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    weights = torch.ones_like(data_obs)
    return data_obs, weights


class TestBaseDataMisfit:
    """Test BaseDataMisfit class"""

    def test_initialization_with_torch_tensor(self):
        """Test initialization with torch tensor data"""
        data_obs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        weights = torch.ones_like(data_obs)
        solver = MockSolver(torch.zeros_like(data_obs))

        dmisfit = BaseDataMisfit(solver, data_obs, weights=weights)

        assert torch.equal(dmisfit.data_obs, data_obs)
        assert torch.equal(dmisfit.weights, weights)
        assert dmisfit.n_data == 3

    def test_initialization_with_numpy_array(self):
        """Test initialization with numpy array data"""
        data_obs = np.array([1.0, 2.0, 3.0])
        weights = np.ones_like(data_obs)
        solver = MockSolver(torch.zeros(3, dtype=torch.float64))

        dmisfit = BaseDataMisfit(solver, data_obs, weights=weights)

        assert dmisfit.data_obs.dtype == torch.float64
        assert dmisfit.weights.dtype == torch.float64
        assert dmisfit.n_data == 3

    def test_initialization_without_weights(self):
        """Test initialization without explicit weights"""
        data_obs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        solver = MockSolver(torch.zeros_like(data_obs))

        dmisfit = BaseDataMisfit(solver, data_obs)

        assert torch.equal(dmisfit.weights, torch.ones_like(data_obs))

    def test_forward_with_solver(self, test_data):
        """Test forward pass with solver"""
        data_obs, weights = test_data

        # Mock solver that returns predicted data
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = BaseDataMisfit(solver, data_obs, weights=weights)
        misfit = dmisfit()

        # Expected misfit: sum(weights * (pred_data - data_obs)^2)
        residual = pred_data - data_obs
        weighted_residual = weights * residual
        expected = torch.sum(weighted_residual**2)

        assert torch.allclose(misfit, expected)


class TestL2DataMisfit:
    """Test L2DataMisfit class"""

    def test_l2_misfit_computation(self, test_data):
        """Test L2 misfit computation"""
        data_obs, weights = test_data

        pred_data = torch.tensor([1.2, 2.3, 2.8, 4.1, 5.2], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = L2DataMisfit(solver, data_obs, weights=weights)
        misfit = dmisfit()

        # L2 misfit: sum(weights * (pred - obs)^2)
        residual = pred_data - data_obs
        weighted_residual = weights * residual
        expected = torch.sum(weighted_residual**2)

        assert torch.allclose(misfit, expected)

    def test_l2_with_custom_weights(self, test_data):
        """Test L2 misfit with custom weights"""
        data_obs, _ = test_data
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = L2DataMisfit(solver, data_obs, weights=weights)
        misfit = dmisfit()

        residual = pred_data - data_obs
        weighted_residual = weights * residual
        expected = torch.sum(weighted_residual**2)

        assert torch.allclose(misfit, expected)


class TestL1DataMisfit:
    """Test L1DataMisfit class"""

    def test_l1_misfit_computation(self, test_data):
        """Test L1 misfit computation"""
        data_obs, weights = test_data

        pred_data = torch.tensor([1.2, 2.3, 2.8, 4.1, 5.2], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = L1DataMisfit(solver, data_obs, weights=weights)
        misfit = dmisfit()

        # L1 misfit: sum(|weights * (pred - obs)|)
        residual = pred_data - data_obs
        weighted_residual = weights * residual
        expected = torch.sum(torch.abs(weighted_residual))

        assert torch.allclose(misfit, expected)


class TestHuberDataMisfit:
    """Test HuberDataMisfit class"""

    def test_huber_misfit_small_residuals(self, test_data):
        """Test Huber misfit with small residuals (L2 regime)"""
        data_obs, weights = test_data
        delta = 2.0

        # Small residuals (< delta) should use L2
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = HuberDataMisfit(
            delta=delta, solver=solver, data=data_obs, weights=weights
        )
        misfit = dmisfit()

        residual = pred_data - data_obs
        weighted_residual = weights * residual
        # All weighted residuals are 0.1, which is < delta=2.0, so use L2
        expected = torch.sum(0.5 * weighted_residual**2)

        assert torch.allclose(misfit, expected)

    def test_huber_misfit_large_residuals(self, test_data):
        """Test Huber misfit with large residuals (L1 regime)"""
        data_obs, weights = test_data
        delta = 0.05

        # Large residuals (> delta) should use L1
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        solver = MockSolver(predicted_data=pred_data)

        dmisfit = HuberDataMisfit(
            delta=delta, solver=solver, data=data_obs, weights=weights
        )
        misfit = dmisfit()

        residual = pred_data - data_obs
        weighted_residual = weights * residual
        abs_residual = torch.abs(weighted_residual)
        # All weighted residuals are 0.1, which is > delta=0.05, so use L1
        expected = torch.sum(delta * (abs_residual - 0.5 * delta))

        assert torch.allclose(misfit, expected)
