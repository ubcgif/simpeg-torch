"""
Test suite for mapping classes in SimPEG-Torch

Tests for various mapping classes including LogMapping, LinearMapping, and SqrtMapping.
"""

import pytest
import torch

from simpegtorch.maps import (
    LogMapping,
    LinearMapping,
)
from simpegtorch.discretize import TensorMesh
from simpegtorch.utils import InjectActiveCells, active_from_xyz, create_flat_topography

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
def active_mapping(test_mesh):
    """Create active cell mapping for test mesh"""
    mesh = test_mesh

    # Create simple topography
    topo_xyz = create_flat_topography(
        x_extent=(-2, 2),
        y_extent=(-2, 2),
        elevation=1.0,  # Surface at z=1
        n_points_x=5,
        n_points_y=5,
    )

    active_cells = active_from_xyz(mesh, topo_xyz)
    air_value = 1e8

    return InjectActiveCells(mesh, active_cells, valInactive=air_value)


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

    def test_log_mapping_with_active_cells(self, active_mapping, test_parameters):
        """Test log mapping with active cell injection"""
        # Get number of active cells
        n_active = torch.sum(active_mapping.active_cells).item()

        log_params = torch.randn(n_active)
        expected_linear = torch.exp(log_params)

        mapper = LogMapping(active_mapping)
        result = mapper(log_params)

        # Result should be on full mesh
        assert result.shape[0] == active_mapping.mesh.nC

        # Active cells should have expected values
        active_values = result[active_mapping.active_cells]
        assert torch.allclose(active_values, expected_linear, rtol=1e-12)

        # Inactive cells should have air value
        inactive_values = result[~active_mapping.active_cells]
        assert torch.allclose(inactive_values, torch.tensor(active_mapping.valInactive))

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


class TestLinearMapping:
    """Test LinearMapping class"""

    def test_linear_mapping_basic(self, test_parameters):
        """Test basic linear mapping functionality (identity)"""
        params = test_parameters["linear_params"]

        mapper = LinearMapping()
        result = mapper(params)

        # Should be identical (identity mapping)
        assert torch.allclose(result, params, rtol=1e-15)
        assert result is not params  # But different tensor

    def test_linear_mapping_with_active_cells(self, active_mapping, test_parameters):
        """Test linear mapping with active cell injection"""
        # Get number of active cells
        n_active = torch.sum(active_mapping.active_cells).item()

        params = torch.randn(n_active) * 100.0  # Random parameters

        mapper = LinearMapping(active_mapping)
        result = mapper(params)

        # Result should be on full mesh
        assert result.shape[0] == active_mapping.mesh.nC

        # Active cells should have input values
        active_values = result[active_mapping.active_cells]
        assert torch.allclose(active_values, params, rtol=1e-15)

        # Inactive cells should have air value
        inactive_values = result[~active_mapping.active_cells]
        assert torch.allclose(inactive_values, torch.tensor(active_mapping.valInactive))

    def test_linear_mapping_gradient(self, test_parameters):
        """Test gradient computation through linear mapping"""
        params = test_parameters["linear_params"].clone()
        params.requires_grad_(True)

        mapper = LinearMapping()
        result = mapper(params)
        loss = torch.sum(result)
        loss.backward()

        # Gradient should be all ones (identity mapping)
        expected_grad = torch.ones_like(params)
        assert torch.allclose(params.grad, expected_grad, rtol=1e-15)

    def test_linear_mapping_preserves_values(self, test_parameters):
        """Test that linear mapping preserves all input characteristics"""
        params = test_parameters["linear_params"]

        mapper = LinearMapping()
        result = mapper(params)

        # Should preserve min, max, mean, std
        assert torch.allclose(torch.min(result), torch.min(params))
        assert torch.allclose(torch.max(result), torch.max(params))
        assert torch.allclose(torch.mean(result), torch.mean(params))
        assert torch.allclose(torch.std(result), torch.std(params))
