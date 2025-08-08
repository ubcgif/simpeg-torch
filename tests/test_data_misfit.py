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
    ComboDataMisfit,
    PseudoHuberDataMisfit,
    WeightedDataMisfit,
    LogDataMisfit,
)


class MockSimulation:
    """Mock simulation class for testing"""

    def __init__(self, predicted_data=None):
        self.predicted_data = predicted_data

    def dpred(self, model):
        """Return predicted data"""
        if self.predicted_data is not None:
            # Ensure predicted data inherits gradient requirements
            if model.requires_grad and not self.predicted_data.requires_grad:
                return self.predicted_data * torch.ones(1, requires_grad=True)
            return self.predicted_data
        # Simple function of model to enable gradients - return 5 data points
        return (model * 0.1)[:5]

    def __call__(self, model):
        return self.dpred(model)


@pytest.fixture
def mock_simulation():
    """Create mock simulation"""
    return MockSimulation()


@pytest.fixture
def test_data():
    """Create test data and model"""
    data_obs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    model = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=torch.float64
    )
    weights = torch.ones_like(data_obs)
    return data_obs, model, weights


class TestBaseDataMisfit:
    """Test BaseDataMisfit class"""

    def test_initialization_with_torch_tensor(self, mock_simulation):
        """Test initialization with torch tensor data"""
        data_obs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        weights = torch.ones_like(data_obs)

        dmisfit = BaseDataMisfit(mock_simulation, data_obs, weights=weights)

        assert torch.equal(dmisfit.data_obs, data_obs)
        assert torch.equal(dmisfit.weights, weights)
        assert dmisfit.n_data == 3

    def test_initialization_with_numpy_array(self, mock_simulation):
        """Test initialization with numpy array data"""
        data_obs = np.array([1.0, 2.0, 3.0])
        weights = np.ones_like(data_obs)

        dmisfit = BaseDataMisfit(mock_simulation, data_obs, weights=weights)

        assert dmisfit.data_obs.dtype == torch.float64
        assert dmisfit.weights.dtype == torch.float64
        assert dmisfit.n_data == 3

    def test_initialization_without_weights(self, mock_simulation):
        """Test initialization without explicit weights"""
        data_obs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        dmisfit = BaseDataMisfit(mock_simulation, data_obs)

        assert torch.equal(dmisfit.weights, torch.ones_like(data_obs))

    def test_forward_with_simulation_dpred(self, test_data):
        """Test forward pass with simulation.dpred method"""
        data_obs, model, weights = test_data

        # Mock simulation that returns predicted data
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = BaseDataMisfit(simulation, data_obs, weights=weights)
        misfit = dmisfit(model)

        # Expected misfit: sum((pred_data - data_obs)^2)
        residual = pred_data - data_obs
        expected = torch.sum(residual**2)

        assert torch.allclose(misfit, expected)

    def test_forward_with_mapping(self, test_data):
        """Test forward pass with model mapping"""
        data_obs, model, weights = test_data

        # Mock simulation and mapping
        pred_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        # Simple mapping that doubles the input
        def mapping_fn(m):
            return 2.0 * m

        dmisfit = BaseDataMisfit(
            simulation, data_obs, weights=weights, mapping=mapping_fn
        )
        misfit = dmisfit(model)

        # Expected misfit with zero residual
        expected = torch.tensor(0.0, dtype=torch.float64)
        assert torch.allclose(misfit, expected)


class TestL2DataMisfit:
    """Test L2DataMisfit class"""

    def test_l2_misfit_computation(self, test_data):
        """Test L2 misfit computation"""
        data_obs, model, weights = test_data

        pred_data = torch.tensor([1.2, 2.3, 2.8, 4.1, 5.2], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = L2DataMisfit(simulation, data_obs, weights=weights)
        misfit = dmisfit(model)

        # L2 misfit: sum((pred - obs)^2)
        residual = pred_data - data_obs
        expected = torch.sum(residual**2)

        assert torch.allclose(misfit, expected)

    def test_l2_with_custom_weights(self, test_data):
        """Test L2 misfit with custom weights"""
        data_obs, model, _ = test_data
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = L2DataMisfit(simulation, data_obs, weights=weights)
        misfit = dmisfit(model)

        residual = pred_data - data_obs
        weighted_residual = weights * residual
        expected = torch.sum(weighted_residual**2)

        assert torch.allclose(misfit, expected)


class TestL1DataMisfit:
    """Test L1DataMisfit class"""

    def test_l1_misfit_computation(self, test_data):
        """Test L1 misfit computation"""
        data_obs, model, weights = test_data

        pred_data = torch.tensor([1.2, 2.3, 2.8, 4.1, 5.2], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = L1DataMisfit(simulation, data_obs, weights=weights)
        misfit = dmisfit(model)

        # L1 misfit: sum(|pred - obs|)
        residual = pred_data - data_obs
        expected = torch.sum(torch.abs(residual))

        assert torch.allclose(misfit, expected)


class TestHuberDataMisfit:
    """Test HuberDataMisfit class"""

    def test_huber_misfit_small_residuals(self, test_data):
        """Test Huber misfit with small residuals (L2 regime)"""
        data_obs, model, weights = test_data
        delta = 2.0

        # Small residuals (< delta) should use L2
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = HuberDataMisfit(
            delta=delta, simulation=simulation, data=data_obs, weights=weights
        )
        misfit = dmisfit(model)

        residual = pred_data - data_obs
        # All residuals are 0.1, which is < delta=2.0, so use L2
        expected = torch.sum(0.5 * residual**2)

        assert torch.allclose(misfit, expected)

    def test_huber_misfit_large_residuals(self, test_data):
        """Test Huber misfit with large residuals (L1 regime)"""
        data_obs, model, weights = test_data
        delta = 0.05

        # Large residuals (> delta) should use L1
        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = HuberDataMisfit(
            delta=delta, simulation=simulation, data=data_obs, weights=weights
        )
        misfit = dmisfit(model)

        residual = pred_data - data_obs
        abs_residual = torch.abs(residual)
        # All residuals are 0.1, which is > delta=0.05, so use L1
        expected = torch.sum(delta * (abs_residual - 0.5 * delta))

        assert torch.allclose(misfit, expected)


class TestComboDataMisfit:
    """Test ComboDataMisfit class"""

    def test_combo_misfit_equal_weights(self, test_data):
        """Test combination misfit with equal weights"""
        data_obs, model, weights = test_data

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        # Create individual misfits
        l2_misfit = L2DataMisfit(simulation, data_obs, weights=weights)
        l1_misfit = L1DataMisfit(simulation, data_obs, weights=weights)

        # Create combo misfit
        combo_misfit = ComboDataMisfit([l2_misfit, l1_misfit])
        misfit = combo_misfit(model)

        # Expected: sum of individual misfits with equal weights
        expected = l2_misfit(model) + l1_misfit(model)
        assert torch.allclose(misfit, expected)

    def test_combo_misfit_custom_weights(self, test_data):
        """Test combination misfit with custom weights"""
        data_obs, model, weights = test_data

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        # Create individual misfits
        l2_misfit = L2DataMisfit(simulation, data_obs, weights=weights)
        l1_misfit = L1DataMisfit(simulation, data_obs, weights=weights)

        # Create combo misfit with custom weights
        combo_weights = [2.0, 0.5]
        combo_misfit = ComboDataMisfit([l2_misfit, l1_misfit], weights=combo_weights)
        misfit = combo_misfit(model)

        # Expected: weighted sum of individual misfits
        expected = 2.0 * l2_misfit(model) + 0.5 * l1_misfit(model)
        assert torch.allclose(misfit, expected)

    def test_combo_misfit_total_data(self, test_data):
        """Test that combo misfit correctly counts total data"""
        data_obs, _, weights = test_data

        simulation = MockSimulation()

        # Create individual misfits with different data sizes
        data1 = data_obs[:3]
        data2 = data_obs[:2]

        l2_misfit = L2DataMisfit(simulation, data1)
        l1_misfit = L1DataMisfit(simulation, data2)

        combo_misfit = ComboDataMisfit([l2_misfit, l1_misfit])

        assert combo_misfit.n_data == 3 + 2


class TestPseudoHuberDataMisfit:
    """Test PseudoHuberDataMisfit class"""

    def test_pseudo_huber_computation(self, test_data):
        """Test pseudo-Huber misfit computation"""
        data_obs, model, weights = test_data
        delta = 1.0

        pred_data = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = PseudoHuberDataMisfit(
            delta=delta, simulation=simulation, data=data_obs, weights=weights
        )
        misfit = dmisfit(model)

        # Pseudo-Huber: δ²(√(1 + (r/δ)²) - 1)
        residual = pred_data - data_obs
        pseudo_huber = delta**2 * (torch.sqrt(1 + (residual / delta) ** 2) - 1)
        expected = torch.sum(pseudo_huber)

        assert torch.allclose(misfit, expected)


class TestWeightedDataMisfit:
    """Test WeightedDataMisfit class"""

    def test_uncertainty_based_weighting(self, test_data):
        """Test uncertainty-based weighting"""
        data_obs, model, _ = test_data
        uncertainties = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = WeightedDataMisfit(simulation, data_obs, uncertainties)
        misfit = dmisfit(model)

        # Weights should be 1/uncertainties
        expected_weights = 1.0 / uncertainties
        residual = pred_data - data_obs
        weighted_residual = expected_weights * residual
        expected = torch.sum(weighted_residual**2)

        assert torch.allclose(misfit, expected)

    def test_uncertainty_clamping(self, test_data):
        """Test that very small uncertainties are clamped"""
        data_obs, model, _ = test_data
        uncertainties = torch.tensor(
            [1e-15, 0.1, 1e-20, 0.2, 1e-10], dtype=torch.float64
        )

        simulation = MockSimulation()

        dmisfit = WeightedDataMisfit(simulation, data_obs, uncertainties)

        # Check that weights are finite (no infinite values from division by tiny numbers)
        assert torch.all(torch.isfinite(dmisfit.weights))


class TestLogDataMisfit:
    """Test LogDataMisfit class"""

    def test_log_misfit_computation(self, test_data):
        """Test logarithmic misfit computation"""
        # Use positive data for log transform
        data_obs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        model = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        weights = torch.ones_like(data_obs)

        pred_data = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = LogDataMisfit(simulation=simulation, data=data_obs, weights=weights)
        misfit = dmisfit(model)

        # Log misfit: sum((log(pred) - log(obs))^2)
        log_pred = torch.log(pred_data)
        log_obs = torch.log(data_obs)
        residual = log_pred - log_obs
        expected = torch.sum(residual**2)

        assert torch.allclose(misfit, expected)

    def test_log_misfit_epsilon_protection(self, test_data):
        """Test that epsilon prevents log(0)"""
        # Include zero and negative values to test epsilon protection
        data_obs = torch.tensor([0.0, -1.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        model = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)

        pred_data = torch.tensor([0.0, -0.5, 1.1, 2.1, 3.1], dtype=torch.float64)
        simulation = MockSimulation(predicted_data=pred_data)

        epsilon = 1e-12
        dmisfit = LogDataMisfit(epsilon=epsilon, simulation=simulation, data=data_obs)
        misfit = dmisfit(model)

        # Should not raise any errors and should return finite value
        assert torch.isfinite(misfit)


class TestDataMisfitGradients:
    """Test that data misfits properly compute gradients"""

    def test_l2_gradient_computation(self, test_data):
        """Test that L2 misfit computes gradients correctly"""
        data_obs, model, weights = test_data

        # Use simulation that doesn't have fixed predicted data to enable gradients
        simulation = MockSimulation()  # Uses model-dependent prediction

        dmisfit = L2DataMisfit(simulation, data_obs, weights=weights)

        # Enable gradient computation
        model_with_grad = model.clone().requires_grad_(True)
        misfit = dmisfit(model_with_grad)

        # Compute gradients
        misfit.backward()

        # Check that gradients exist
        assert model_with_grad.grad is not None
        assert model_with_grad.grad.shape == model.shape

    def test_huber_gradient_computation(self, test_data):
        """Test that Huber misfit computes gradients correctly"""
        data_obs, model, weights = test_data

        # Use simulation that doesn't have fixed predicted data to enable gradients
        simulation = MockSimulation()  # Uses model-dependent prediction

        dmisfit = HuberDataMisfit(
            delta=1.0, simulation=simulation, data=data_obs, weights=weights
        )

        # Enable gradient computation
        model_with_grad = model.clone().requires_grad_(True)
        misfit = dmisfit(model_with_grad)

        # Compute gradients
        misfit.backward()

        # Check that gradients exist and are finite
        assert model_with_grad.grad is not None
        assert torch.all(torch.isfinite(model_with_grad.grad))


class TestDataMisfitDeviceAndDtype:
    """Test device and dtype handling in data misfits"""

    def test_device_consistency(self, test_data):
        """Test that data misfit handles device consistently"""
        data_obs, model, weights = test_data
        device = "cpu"  # Use CPU for CI compatibility

        simulation = MockSimulation()
        dmisfit = BaseDataMisfit(simulation, data_obs, weights=weights, device=device)

        assert dmisfit.data_obs.device.type == device
        assert dmisfit.weights.device.type == device

    def test_dtype_consistency(self, test_data):
        """Test that data misfit handles dtype consistently"""
        data_obs, model, weights = test_data
        dtype = torch.float32

        simulation = MockSimulation()
        dmisfit = BaseDataMisfit(simulation, data_obs, weights=weights, dtype=dtype)

        assert dmisfit.data_obs.dtype == dtype
        assert dmisfit.weights.dtype == dtype

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI
    def test_misfit_computation_on_device(self, test_data, device):
        """Test misfit computation on different devices"""
        data_obs, model, weights = test_data

        # Move everything to specified device
        data_obs = data_obs.to(device)
        model = model.to(device)
        weights = weights.to(device)

        pred_data = torch.tensor(
            [1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64, device=device
        )
        simulation = MockSimulation(predicted_data=pred_data)

        dmisfit = L2DataMisfit(simulation, data_obs, weights=weights, device=device)
        misfit = dmisfit(model)

        assert misfit.device.type == device
        assert torch.isfinite(misfit)


# Additional test for error handling
class TestDataMisfitErrorHandling:
    """Test error handling in data misfit classes"""

    def test_invalid_mapping_type(self, test_data):
        """Test error when invalid mapping is provided"""
        data_obs, model, weights = test_data
        simulation = MockSimulation()

        # Invalid mapping (not callable or module)
        invalid_mapping = "not_a_mapping"

        dmisfit = BaseDataMisfit(
            simulation, data_obs, weights=weights, mapping=invalid_mapping
        )

        with pytest.raises(TypeError, match="Mapping must be callable"):
            dmisfit(model)

    def test_simulation_without_dpred(self, test_data):
        """Test fallback when simulation doesn't have dpred method"""
        data_obs, model, weights = test_data

        # Mock simulation without dpred but callable
        class CallableSimulation:
            def __call__(self, m):
                return torch.ones(5, dtype=torch.float64)

        simulation = CallableSimulation()
        dmisfit = BaseDataMisfit(simulation, data_obs, weights=weights)

        # Should work by calling simulation directly
        misfit = dmisfit(model)
        assert torch.isfinite(misfit)


if __name__ == "__main__":
    pytest.main([__file__])
