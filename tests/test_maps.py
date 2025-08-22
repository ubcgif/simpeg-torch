"""
Test suite for mapping classes in SimPEG-Torch

Tests for various mapping classes including LogMapping, LinearMapping, and SqrtMapping.
"""

import pytest
import torch

from simpegtorch.maps import (
    LogMapping,
)
from simpegtorch.discretize import TensorMesh

torch.set_default_dtype(torch.float64)


@pytest.fixture
def test_mesh():
    """Create a simple test mesh"""
    dx = dy = dz = 1.0
    nx, ny, nz = 4, 4, 4

    hx = torch.full((nx,), dx)
    hy = torch.full((ny,), dy)
    hz = torch.full((nz,), dz)

    mesh = TensorMesh([hx, hy, hz])
    return mesh


@pytest.fixture
def test_parameters():
    """Create test parameter sets"""
    n_params = 10

    # Various parameter ranges for testing
    log_params = torch.randn(n_params) * 2.0  # Random log values
    linear_params = torch.exp(log_params)  # Corresponding linear values
    squared_params = linear_params**2  # Corresponding squared values

    return {
        "log_params": log_params,
        "linear_params": linear_params,
        "squared_params": squared_params,
        "n_params": n_params,
    }


class TestLogMapping:
    """Test LogMapping class"""

    def test_log_mapping_basic(self, test_parameters):
        """Test basic log mapping functionality"""
        log_params = test_parameters["log_params"]
        expected_linear = test_parameters["linear_params"]

        mapper = LogMapping()
        result = mapper(log_params)

        # Should transform log to linear via exp
        assert torch.allclose(result, expected_linear, rtol=1e-12)

    def test_log_mapping_gradient(self, test_parameters):
        """Test gradient computation through log mapping"""
        log_params = test_parameters["log_params"].clone()
        log_params.requires_grad_(True)

        mapper = LogMapping()
        result = mapper(log_params)
        loss = torch.sum(result)
        loss.backward()

        # Gradient should be exp(log_params) = result
        expected_grad = result.detach()
        assert torch.allclose(log_params.grad, expected_grad, rtol=1e-12)

    def test_log_mapping_positive_output(self, test_parameters):
        """Test that log mapping always produces positive outputs"""
        log_params = test_parameters["log_params"]

        mapper = LogMapping()
        result = mapper(log_params)

        # All outputs should be positive
        assert torch.all(result > 0)

    def test_log_mapping_extreme_values(self):
        """Test log mapping with extreme input values"""
        # Very negative values (near zero output)
        log_params_small = torch.tensor([-10.0, -20.0, -30.0])
        mapper = LogMapping()
        result_small = mapper(log_params_small)

        assert torch.all(result_small > 0)
        assert torch.all(result_small < 1e-4)  # Very small but positive

        # Very positive values (large output)
        log_params_large = torch.tensor([10.0, 20.0, 30.0])
        result_large = mapper(log_params_large)

        assert torch.all(result_large > 1e4)  # Very large
        assert torch.all(torch.isfinite(result_large))  # But finite
