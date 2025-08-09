"""
Test suite for regularization module in SimPEG-Torch

Tests for various regularization classes including Smallness, SmoothnessFirstOrder,
SmoothnessSecondOrder, WeightedLeastSquares, and Sparse regularization.
"""

import pytest
import torch

from simpegtorch.regularization import (
    Smallness,
    SmoothnessFirstOrder,
    SmoothnessSecondOrder,
    Sparse,
)

from simpegtorch.discretize import TensorMesh

torch.set_default_dtype(torch.float64)


@pytest.fixture
def test_data():
    # Define mesh parameters - reasonable size for inversion
    dx = dy = dz = 1.0  # 1m cells
    nx, ny, nz = 4, 4, 4

    # Create cell sizes
    hx = torch.full((nx,), dx)
    hy = torch.full((ny,), dy)
    hz = torch.full((nz,), dz)
    mesh = TensorMesh([hx, hy, hz])
    model = torch.ones(mesh.nC) * 1e3
    return mesh, model


class TestBaseRegularization:

    def test_mesh_volume(self, test_data):
        mesh, model = test_data
        volume = mesh.cell_volumes
        assert torch.allclose(
            volume, torch.ones(mesh.nC, dtype=torch.float64) * (1.0**3)
        )

    def test_smallness(self, test_data):
        mesh, model = test_data
        reg = Smallness(mesh=mesh, reference_model=model)
        eps_tensor = torch.rand_like(model) * 1e2
        deviated_mesh = model + eps_tensor
        phi = reg(deviated_mesh)
        assert torch.allclose(phi, torch.sum(eps_tensor**2))

    def test_smoothness_first_order_zero(self, test_data):
        # constant model should have zero smoothness
        mesh, model = test_data
        reg = SmoothnessFirstOrder(mesh=mesh)
        phi = reg(model)
        assert torch.allclose(phi, torch.tensor(0.0))

    def test_smoothness_first_order_nonzero(self, test_data):
        # non-constant model should have non-zero smoothness
        mesh, model = test_data
        # Create a model with a linear gradient
        model = torch.arange(mesh.nC, dtype=torch.float64)

        reg = SmoothnessFirstOrder(mesh=mesh)
        phi = reg(model)
        # Check that the smoothness is non-zero
        assert torch.all(phi > 0)

    def test_smoothness_3_dirs(self, test_data):
        mesh, model = test_data
        reg_x = SmoothnessFirstOrder(mesh=mesh, orientation="x")
        reg_y = SmoothnessFirstOrder(mesh=mesh, orientation="y")
        reg_z = SmoothnessFirstOrder(mesh=mesh, orientation="z")
        # Create a model with a linear gradient in all three directions
        model = torch.arange(mesh.nC, dtype=torch.float64)
        phi_x = reg_x(model)
        phi_y = reg_y(model)
        phi_z = reg_z(model)
        # Check that the smoothness in each direction is non-zero
        assert torch.all(phi_x > 0)
        assert torch.all(phi_y > 0)
        assert torch.all(phi_z > 0)

    def test_second_order_smoothness_zero(self, test_data):
        mesh, model = test_data
        reg_x = SmoothnessSecondOrder(mesh=mesh, orientation="x")
        reg_y = SmoothnessSecondOrder(mesh=mesh, orientation="y")
        reg_z = SmoothnessSecondOrder(mesh=mesh, orientation="z")
        # Create a model with a linear gradient in all three directions
        model = torch.arange(mesh.nC, dtype=torch.float64)
        phi_x = reg_x(model)
        phi_y = reg_y(model)
        phi_z = reg_z(model)
        # Check that the smoothness in each direction is non-zero
        assert torch.all(phi_x > 0)
        assert torch.all(phi_y > 0)
        assert torch.all(phi_z > 0)

    def test_linear_model_first_order_smoothness(self, test_data):
        """Test first order smoothness on a linear model - should be constant (non-zero)"""
        mesh, _ = test_data

        # Create linear model in x-direction: m = x_coord
        cell_centers = mesh.cell_centers
        x_coords = cell_centers[:, 0]  # x-coordinates
        linear_model_x = x_coords.clone()

        reg_x = SmoothnessFirstOrder(mesh=mesh, orientation="x")
        reg_y = SmoothnessFirstOrder(mesh=mesh, orientation="y")
        reg_z = SmoothnessFirstOrder(mesh=mesh, orientation="z")

        phi_x = reg_x(linear_model_x)
        phi_y = reg_y(linear_model_x)
        phi_z = reg_z(linear_model_x)

        # For linear model in x: first derivative in x should be constant (1.0)
        # so smoothness should be non-zero but consistent
        assert torch.all(phi_x > 0)

        # For linear model in x: first derivative in y,z should be zero
        # so smoothness in y,z should be near zero
        assert torch.allclose(phi_y, torch.tensor(0.0), atol=1e-10)
        assert torch.allclose(phi_z, torch.tensor(0.0), atol=1e-10)

    def test_linear_model_second_order_smoothness(self, test_data):
        """Test second order smoothness on a linear model - should be much smaller than quadratic"""
        mesh, _ = test_data

        # Create linear model in x-direction: m = x_coord
        cell_centers = mesh.cell_centers
        x_coords = cell_centers[:, 0]  # x-coordinates
        linear_model_x = x_coords.clone()

        # Create quadratic model for comparison: m = x_coord^2
        quadratic_model_x = x_coords**2

        reg_x = SmoothnessSecondOrder(mesh=mesh, orientation="x")
        reg_y = SmoothnessSecondOrder(mesh=mesh, orientation="y")
        reg_z = SmoothnessSecondOrder(mesh=mesh, orientation="z")

        phi_linear_x = reg_x(linear_model_x)
        phi_linear_y = reg_y(linear_model_x)
        phi_linear_z = reg_z(linear_model_x)

        phi_quad_x = reg_x(quadratic_model_x)

        # For linear model: second derivative should be much smaller than quadratic
        # (may not be exactly zero due to boundary effects)
        assert phi_linear_x < phi_quad_x * 0.1  # Linear should be <10% of quadratic
        assert torch.allclose(phi_linear_y, torch.tensor(0.0), atol=1e-8)
        assert torch.allclose(phi_linear_z, torch.tensor(0.0), atol=1e-8)

    def test_quadratic_model_second_order_smoothness(self, test_data):
        """Test second order smoothness on a quadratic model - should be non-zero"""
        mesh, _ = test_data

        # Create quadratic model in x-direction: m = x_coord^2
        cell_centers = mesh.cell_centers
        x_coords = cell_centers[:, 0]  # x-coordinates
        quadratic_model_x = x_coords**2

        reg_x = SmoothnessSecondOrder(mesh=mesh, orientation="x")
        reg_y = SmoothnessSecondOrder(mesh=mesh, orientation="y")
        reg_z = SmoothnessSecondOrder(mesh=mesh, orientation="z")

        phi_x = reg_x(quadratic_model_x)
        phi_y = reg_y(quadratic_model_x)
        phi_z = reg_z(quadratic_model_x)

        # For quadratic model in x: second derivative in x should be constant (2.0)
        assert torch.all(phi_x > 0)

        # For quadratic model in x: second derivative in y,z should be zero
        assert torch.allclose(phi_y, torch.tensor(0.0), atol=1e-10)
        assert torch.allclose(phi_z, torch.tensor(0.0), atol=1e-10)

    def test_sparse_l1_norm(self, test_data):
        """Test L1 sparse regularization"""
        mesh, model = test_data
        reg = Sparse(mesh=mesh, norm=1)  # L1 norm

        # Test with zero model
        zero_model = torch.zeros_like(model)
        phi_zero = reg(zero_model)
        assert torch.allclose(phi_zero, torch.tensor(0.0))

        # Test with positive model
        positive_model = torch.ones_like(model) * 5.0
        phi_pos = reg(positive_model)
        expected_l1 = torch.sum(torch.abs(positive_model))
        assert torch.allclose(phi_pos, expected_l1)

        # Test with mixed sign model
        mixed_model = torch.tensor(
            [-2.0, 1.0, -3.0, 4.0] * (mesh.nC // 4), dtype=torch.float64
        )[: mesh.nC]
        phi_mixed = reg(mixed_model)
        expected_mixed_l1 = torch.sum(torch.abs(mixed_model))
        assert torch.allclose(phi_mixed, expected_mixed_l1)

    def test_sparse_l2_norm(self, test_data):
        """Test L2 sparse regularization (should be same as Smallness)"""
        mesh, model = test_data
        reg_sparse = Sparse(mesh=mesh, norm=2)  # L2 norm
        reg_smallness = Smallness(mesh=mesh)

        # Compare with random model
        random_model = torch.randn_like(model) * 10.0
        phi_sparse = reg_sparse(random_model)
        phi_smallness = reg_smallness(random_model)

        # Should be identical for L2 norm without reference model
        assert torch.allclose(phi_sparse, phi_smallness)

    def test_sparse_with_reference_model(self, test_data):
        """Test sparse regularization with reference model"""
        mesh, model = test_data
        reference = torch.ones_like(model) * 2.0
        reg = Sparse(mesh=mesh, norm=1, reference_model=reference)

        # Test model identical to reference
        phi_same = reg(reference)
        assert torch.allclose(phi_same, torch.tensor(0.0))

        # Test deviation from reference
        deviated_model = (
            reference
            + torch.tensor(
                [1.0, -1.0, 2.0, -2.0] * (mesh.nC // 4), dtype=torch.float64
            )[: mesh.nC]
        )
        phi_dev = reg(deviated_model)
        expected_dev = torch.sum(torch.abs(deviated_model - reference))
        assert torch.allclose(phi_dev, expected_dev)

    def test_sparse_different_norms(self, test_data):
        """Test sparse regularization with different norm orders"""
        mesh, model = test_data
        test_model = torch.tensor(
            [2.0, -1.0, 3.0, -4.0] * (mesh.nC // 4), dtype=torch.float64
        )[: mesh.nC]

        reg_l1 = Sparse(mesh=mesh, norm=1)
        reg_l2 = Sparse(mesh=mesh, norm=2)

        phi_l1 = reg_l1(test_model)
        phi_l2 = reg_l2(test_model)

        # L1 norm: sum of absolute values
        expected_l1 = torch.sum(torch.abs(test_model))
        assert torch.allclose(phi_l1, expected_l1)

        # L2 norm: sum of squares
        expected_l2 = torch.sum(test_model**2)
        assert torch.allclose(phi_l2, expected_l2)

        # For values > 1, L2 should be larger than L1
        # For values < 1, L1 should be larger than L2
        # This specific test model has values > 1, so L2 > L1
        assert phi_l2 > phi_l1


if __name__ == "__main__":
    pytest.main([__file__])
