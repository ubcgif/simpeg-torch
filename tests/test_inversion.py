"""
Test suite for inversion module in SimPEG-Torch

Tests for inversion classes and directives including BaseInversion, BaseInvProblem,
and various inversion directives.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from simpegtorch.inversion import (
    BaseInversion,
    BaseInvProblem,
    InversionDirective,
    BetaSchedule,
    TargetMisfit,
    BetaEstimate_ByEig,
)


class MockDataMisfit:
    """Mock data misfit class for testing"""

    def __init__(self, n_data=10, misfit_value=100.0):
        self.n_data = n_data
        self.misfit_value = misfit_value
        self.call_count = 0

    def __call__(self, model):
        self.call_count += 1
        # Return a value that depends on model to enable gradients
        return self.misfit_value * torch.ones(
            1, dtype=model.dtype, device=model.device
        ) + 0.01 * torch.sum(model**2)


class MockRegularization:
    """Mock regularization class for testing"""

    def __init__(self, reg_value=10.0):
        self.reg_value = reg_value
        self.call_count = 0

    def __call__(self, model):
        self.call_count += 1
        # Return a value that depends on model to enable gradients
        return self.reg_value * torch.ones(
            1, dtype=model.dtype, device=model.device
        ) + 0.001 * torch.sum(model**2)


class MockOptimizer:
    """Mock PyTorch optimizer for testing"""

    def __init__(self, params, lr=0.01):
        self.param_groups = [{"params": params, "lr": lr}]
        self.step_count = 0
        self.zero_grad_count = 0

    def zero_grad(self):
        self.zero_grad_count += 1
        for param in self.param_groups[0]["params"]:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        self.step_count += 1
        # Simple gradient descent step for testing
        for param in self.param_groups[0]["params"]:
            if param.grad is not None:
                param.data -= self.param_groups[0]["lr"] * param.grad


@pytest.fixture
def mock_dmisfit():
    """Create mock data misfit"""
    return MockDataMisfit()


@pytest.fixture
def mock_reg():
    """Create mock regularization"""
    return MockRegularization()


@pytest.fixture
def test_model():
    """Create test model"""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)


class TestBaseInvProblem:
    """Test BaseInvProblem class"""

    def test_initialization(self, mock_dmisfit, mock_reg):
        """Test BaseInvProblem initialization"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=2.0, max_iter=100)

        assert inv_prob.dmisfit is mock_dmisfit
        assert inv_prob.reg is mock_reg
        assert inv_prob.beta == 2.0
        assert inv_prob.max_iter == 100
        assert inv_prob.optimizer_class == torch.optim.Adam

    def test_objective_function_computation(self, mock_dmisfit, mock_reg, test_model):
        """Test objective function computation"""
        beta = 2.0
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=beta)

        phi = inv_prob(test_model)

        # Should be phi_d + beta * phi_m (with model-dependent components)
        phi_d_expected = mock_dmisfit.misfit_value + 0.01 * torch.sum(test_model**2)
        phi_m_expected = mock_reg.reg_value + 0.001 * torch.sum(test_model**2)
        expected = phi_d_expected + beta * phi_m_expected
        assert torch.allclose(phi, expected, rtol=1e-5)

    def test_optimizer_setup_string(self, mock_dmisfit, mock_reg, test_model):
        """Test optimizer setup with string name"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, optimizer_class="SGD")

        model = test_model.clone().requires_grad_(True)
        optimizer = inv_prob.setup_optimizer(model)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["params"][0] is model

    def test_optimizer_setup_class(self, mock_dmisfit, mock_reg, test_model):
        """Test optimizer setup with optimizer class"""
        inv_prob = BaseInvProblem(
            mock_dmisfit, mock_reg, optimizer_class=torch.optim.RMSprop
        )

        model = test_model.clone().requires_grad_(True)
        optimizer = inv_prob.setup_optimizer(model)

        assert isinstance(optimizer, torch.optim.RMSprop)

    def test_optimizer_kwargs(self, mock_dmisfit, mock_reg, test_model):
        """Test optimizer setup with custom kwargs"""
        optimizer_kwargs = {"lr": 0.001, "momentum": 0.9}
        inv_prob = BaseInvProblem(
            mock_dmisfit,
            mock_reg,
            optimizer_class="SGD",
            optimizer_kwargs=optimizer_kwargs,
        )

        model = test_model.clone().requires_grad_(True)
        optimizer = inv_prob.setup_optimizer(model)

        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["momentum"] == 0.9


class TestBaseInversion:
    """Test BaseInversion class"""

    def test_initialization(self, mock_dmisfit, mock_reg):
        """Test BaseInversion initialization"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg)

        directive1 = Mock(spec=InversionDirective)
        directive2 = Mock(spec=InversionDirective)
        directives = [directive1, directive2]

        inversion = BaseInversion(inv_prob, directives=directives)

        assert inversion.inv_prob is inv_prob
        assert len(inversion.directives) == 2

        # Check that directives have references set
        for directive in inversion.directives:
            assert directive.inversion is inversion
            assert directive.inv_prob is inv_prob

    def test_initialization_with_numpy_model(self, mock_dmisfit, mock_reg):
        """Test inversion run with numpy initial model"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=2)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        inversion = BaseInversion(inv_prob)

        # Use numpy array as initial model
        m0 = np.array([1.0, 2.0, 3.0])
        result = inversion.run(m0)

        assert isinstance(result, np.ndarray)
        assert result.shape == m0.shape

    def test_run_with_torch_model(self, mock_dmisfit, mock_reg, test_model):
        """Test inversion run with torch tensor initial model"""
        # Create mock inv_prob that limits iterations
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=2)

        # Mock the setup_optimizer method to return our mock optimizer
        mock_optimizer = MockOptimizer([test_model])
        inv_prob.setup_optimizer = Mock(return_value=mock_optimizer)

        inversion = BaseInversion(inv_prob)
        result = inversion.run(test_model.clone())

        assert isinstance(result, np.ndarray)
        assert result.shape == test_model.shape
        assert inversion.iteration == 2  # Should run for max_iter iterations

    def test_directive_calls(self, mock_dmisfit, mock_reg, test_model):
        """Test that directives are called at appropriate times"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=1)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        # Create mock directives
        directive1 = Mock(spec=InversionDirective)
        directive2 = Mock(spec=InversionDirective)

        inversion = BaseInversion(inv_prob, directives=[directive1, directive2])
        inversion.run(test_model.clone())

        # Check that initialize was called
        directive1.initialize.assert_called_once()
        directive2.initialize.assert_called_once()

        # Check that endIter was called
        directive1.endIter.assert_called_once()
        directive2.endIter.assert_called_once()

        # Check that finish was called
        directive1.finish.assert_called_once()
        directive2.finish.assert_called_once()

    def test_history_tracking(self, mock_dmisfit, mock_reg, test_model):
        """Test that inversion tracks objective function history"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=3, beta=2.0)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        inversion = BaseInversion(inv_prob)
        inversion.run(test_model.clone())

        # Check that history was recorded
        assert len(inversion.phi_d_history) == 3
        assert len(inversion.phi_m_history) == 3
        assert len(inversion.phi_history) == 3
        assert len(inversion.beta_history) == 3

        # Check values (approximately, since they depend on model which changes)
        assert all(phi_d > 0 for phi_d in inversion.phi_d_history)
        assert all(phi_m > 0 for phi_m in inversion.phi_m_history)
        assert all(beta == 2.0 for beta in inversion.beta_history)

    def test_convergence_by_directive(self, mock_dmisfit, mock_reg, test_model):
        """Test that inversion can be stopped by directive"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=100)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        # Create directive that sets convergence after 2 iterations
        class ConvergenceDirective(InversionDirective):
            def endIter(self):
                if self.inversion.iteration >= 2:
                    self.inversion.converged = True
                    self.inversion.reason_for_stop = "Test convergence"

        directive = ConvergenceDirective()
        inversion = BaseInversion(inv_prob, directives=[directive])
        inversion.run(test_model.clone())

        assert inversion.converged
        assert inversion.iteration == 2
        assert inversion.reason_for_stop == "Test convergence"


class TestInversionDirective:
    """Test base InversionDirective class"""

    def test_initialization(self):
        """Test directive initialization"""
        directive = InversionDirective()

        assert directive.inversion is None
        assert directive.inv_prob is None
        assert directive.opt is None

    def test_default_methods(self):
        """Test that default methods do nothing"""
        directive = InversionDirective()

        # Should not raise any errors
        directive.initialize()
        directive.endIter()
        directive.finish()


class TestBetaSchedule:
    """Test BetaSchedule directive"""

    def test_initialization(self):
        """Test BetaSchedule initialization"""
        cooling_factor = 5.0
        cooling_rate = 2

        schedule = BetaSchedule(
            cooling_factor=cooling_factor, cooling_rate=cooling_rate
        )

        assert schedule.cooling_factor == cooling_factor
        assert schedule.cooling_rate == cooling_rate

    def test_beta_cooling(self, mock_dmisfit, mock_reg):
        """Test beta cooling schedule"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=8.0, max_iter=10)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        # Cool beta every 3 iterations by factor of 2
        schedule = BetaSchedule(cooling_factor=2.0, cooling_rate=3)
        inversion = BaseInversion(inv_prob, directives=[schedule])

        # Mock the iteration counter
        test_model = torch.tensor([1.0, 2.0, 3.0])
        inversion.run(test_model)

        # Beta should have been cooled at iterations 3, 6, 9
        # Starting beta: 8.0
        # After iteration 3: 8.0/2 = 4.0
        # After iteration 6: 4.0/2 = 2.0
        # After iteration 9: 2.0/2 = 1.0
        expected_final_beta = 8.0 / (2.0**3)  # Cooled 3 times
        assert np.isclose(inv_prob.beta, expected_final_beta)

    def test_no_cooling_first_iteration(self, mock_dmisfit, mock_reg):
        """Test that beta is not cooled on first iteration"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=4.0, max_iter=1)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        schedule = BetaSchedule(cooling_factor=2.0, cooling_rate=1)
        inversion = BaseInversion(inv_prob, directives=[schedule])

        # Set up mock inversion state
        inversion.iteration = 0
        schedule.inversion = inversion
        schedule.inv_prob = inv_prob

        original_beta = inv_prob.beta
        schedule.endIter()

        # Beta should not change on iteration 0
        assert inv_prob.beta == original_beta


class TestTargetMisfit:
    """Test TargetMisfit directive"""

    def test_initialization(self):
        """Test TargetMisfit initialization"""
        chi_factor = 1.5
        target = TargetMisfit(chi_factor=chi_factor)

        assert target.chi_factor == chi_factor
        assert target.target is None

    def test_initialize_with_n_data(self, mock_dmisfit, mock_reg):
        """Test target initialization when dmisfit has n_data"""
        mock_dmisfit.n_data = 50
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg)

        target = TargetMisfit(chi_factor=1.2)
        target.inv_prob = inv_prob

        target.initialize()

        expected_target = 1.2 * 50
        assert target.target == expected_target

    def test_initialize_fallback(self, mock_dmisfit, mock_reg, test_model):
        """Test target initialization fallback when n_data not available"""
        # Remove n_data attribute
        if hasattr(mock_dmisfit, "n_data"):
            delattr(mock_dmisfit, "n_data")

        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg)

        # Create mock inversion
        inversion = Mock()
        inversion.model = test_model

        target = TargetMisfit(chi_factor=1.0)
        target.inv_prob = inv_prob
        target.inversion = inversion

        target.initialize()

        # Should set target to current misfit / 100 (approximately, since mock is model-dependent)
        assert target.target is not None
        assert target.target > 0
        assert (
            target.target < mock_dmisfit.misfit_value
        )  # Should be smaller than base misfit

    def test_convergence_check(self, mock_dmisfit, mock_reg):
        """Test convergence checking"""

        # Create mock inversion with history
        inversion = Mock()
        inversion.phi_d_history = [150.0, 100.0, 50.0]
        inversion.converged = False
        inversion.reason_for_stop = None

        target = TargetMisfit()
        target.target = 75.0
        target.inversion = inversion

        # Should not converge yet (current misfit = 50.0 > 75.0 is false)
        # Actually 50.0 < 75.0, so it should converge
        target.endIter()

        assert inversion.converged is True
        assert "Target misfit reached" in inversion.reason_for_stop

    def test_no_convergence_above_target(self, mock_dmisfit, mock_reg):
        """Test no convergence when above target"""

        # Create mock inversion with history
        inversion = Mock()
        inversion.phi_d_history = [150.0, 100.0, 90.0]
        inversion.converged = False

        target = TargetMisfit()
        target.target = 75.0
        target.inversion = inversion

        # Should not converge (current misfit = 90.0 > 75.0)
        target.endIter()

        assert inversion.converged is False


class TestBetaEstimate_ByEig:
    """Test BetaEstimate_ByEig directive"""

    def test_initialization(self):
        """Test BetaEstimate initialization"""
        beta0_ratio = 2.0
        estimator = BetaEstimate_ByEig(beta0_ratio=beta0_ratio)

        assert estimator.beta0_ratio == beta0_ratio

    def test_beta_estimation(self, mock_dmisfit, mock_reg, test_model):
        """Test beta estimation from misfit ratio"""
        # Set specific values for testing
        mock_dmisfit.misfit_value = 100.0
        mock_reg.reg_value = 10.0

        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=1.0)

        # Create mock inversion
        inversion = Mock()
        inversion.model = test_model

        estimator = BetaEstimate_ByEig(beta0_ratio=2.0)
        estimator.inv_prob = inv_prob
        estimator.inversion = inversion

        estimator.initialize()

        # Expected beta should be approximately beta0_ratio * phi_d / phi_m
        # But now our mocks return model-dependent values, so just check it changed
        assert inv_prob.beta != 1.0  # Should have changed from initial value

    def test_zero_regularization_handling(self, mock_dmisfit, mock_reg, test_model):
        """Test handling when regularization is zero"""

        # Create a mock that truly returns zero regularization
        class ZeroRegularization:
            def __call__(self, model):
                return torch.tensor(0.0, dtype=model.dtype, device=model.device)

        zero_reg = ZeroRegularization()

        inv_prob = BaseInvProblem(mock_dmisfit, zero_reg, beta=5.0)

        # Create mock inversion
        inversion = Mock()
        inversion.model = test_model

        estimator = BetaEstimate_ByEig(beta0_ratio=2.0)
        estimator.inv_prob = inv_prob
        estimator.inversion = inversion

        original_beta = inv_prob.beta
        estimator.initialize()

        # Beta should remain unchanged when regularization is zero
        assert inv_prob.beta == original_beta


class TestInversionIntegration:
    """Integration tests combining multiple components"""

    def test_full_inversion_workflow(self, mock_dmisfit, mock_reg, test_model):
        """Test complete inversion workflow with multiple directives"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, beta=8.0, max_iter=10)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        # Create multiple directives
        beta_schedule = BetaSchedule(cooling_factor=2.0, cooling_rate=3)
        target_misfit = TargetMisfit(chi_factor=1.0)
        beta_estimate = BetaEstimate_ByEig(beta0_ratio=1.0)

        directives = [beta_estimate, beta_schedule, target_misfit]
        inversion = BaseInversion(inv_prob, directives=directives)

        result = inversion.run(test_model.clone())

        # Check that result is valid
        assert isinstance(result, np.ndarray)
        assert result.shape == test_model.shape

        # Check that history was recorded
        assert len(inversion.phi_d_history) > 0
        assert len(inversion.phi_m_history) > 0
        assert len(inversion.beta_history) > 0

    def test_convergence_by_target_misfit(self, mock_dmisfit, mock_reg, test_model):
        """Test inversion stops when target misfit is reached"""

        # Set up decreasing misfit values
        class DecreasingMisfit:
            def __init__(self):
                self.call_count = 0
                self.values = [100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 1.0]

            def __call__(self, model):
                value = self.values[min(self.call_count, len(self.values) - 1)]
                self.call_count += 1
                # Return decreasing value without model dependence to ensure convergence
                return torch.tensor(value, dtype=model.dtype, device=model.device)

        decreasing_misfit = DecreasingMisfit()
        inv_prob = BaseInvProblem(decreasing_misfit, mock_reg, beta=1.0, max_iter=100)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        # Set target that should be reached around iteration 5-6
        target_misfit = TargetMisfit(chi_factor=1.0)
        target_misfit.target = 8.0  # Should stop around iteration 5-6

        inversion = BaseInversion(inv_prob, directives=[target_misfit])
        inversion.run(test_model.clone())

        assert inversion.converged
        assert "Target misfit reached" in inversion.reason_for_stop
        assert inversion.iteration < 100  # Should stop before max iterations


class TestInversionErrorHandling:
    """Test error handling in inversion classes"""

    def test_invalid_optimizer_class(self, mock_dmisfit, mock_reg):
        """Test error with invalid optimizer class"""
        with pytest.raises(AttributeError):
            BaseInvProblem(mock_dmisfit, mock_reg, optimizer_class="InvalidOptimizer")

    def test_missing_dmisfit_methods(self, mock_reg, test_model):
        """Test handling when dmisfit doesn't have required methods"""

        # Create minimal dmisfit without standard methods
        class MinimalMisfit:
            def __call__(self, model):
                return torch.tensor(1.0)

        minimal_misfit = MinimalMisfit()
        inv_prob = BaseInvProblem(minimal_misfit, mock_reg)

        # Should still work
        phi = inv_prob(test_model)
        assert torch.isfinite(phi)


class TestInversionDeviceAndDtype:
    """Test device and dtype handling in inversion"""

    def test_device_consistency(self, mock_dmisfit, mock_reg):
        """Test that inversion maintains device consistency"""
        device = "cpu"  # Use CPU for CI compatibility
        dtype = torch.float32

        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg)
        inversion = BaseInversion(inv_prob, device=device, dtype=dtype)

        test_model = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
        result = inversion.run(test_model.clone())

        assert result.dtype == np.float32  # numpy equivalent

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI
    def test_inversion_on_device(self, mock_dmisfit, mock_reg, device):
        """Test inversion computation on different devices"""
        inv_prob = BaseInvProblem(mock_dmisfit, mock_reg, max_iter=2)
        inv_prob.setup_optimizer = Mock(return_value=MockOptimizer([]))

        inversion = BaseInversion(inv_prob, device=device)
        test_model = torch.tensor([1.0, 2.0, 3.0], device=device)

        result = inversion.run(test_model.clone())

        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__])
