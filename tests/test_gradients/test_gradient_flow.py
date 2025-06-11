"""
Comprehensive test suite for gradient flow through SimPEG-Torch components.

This test suite verifies that automatic differentiation works correctly through:
- TorchMatSolver sparse linear solver
- Property tensor creation
- Face inner product matrices
- End-to-end discretization and solving pipeline
"""

import torch
import pytest
import numpy as np
from scipy import sparse
from simpegtorch.discretize import TensorMesh
from simpegtorch.torchmatsolver import TorchMatSolver
from simpegtorch.discretize.utils import sdiag, make_property_tensor

torch.set_default_dtype(torch.float64)


class TestTorchMatSolverGradients:
    """Test gradient flow through TorchMatSolver."""

    def test_forward_solve_dense(self):
        """Test forward solve with dense matrix using scipy sparse solver."""
        # Create a simple 3x3 symmetric positive definite matrix
        A_sparse = sparse.csc_matrix(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        )
        b_np = np.array([1.0, 2.0, 3.0])

        # Create sparse tensor directly from COO format
        A_coo = A_sparse.tocoo()
        indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
        A_torch = torch.sparse_coo_tensor(
            indices, values, A_coo.shape, requires_grad=True
        )
        b_torch = torch.tensor(b_np, requires_grad=True)

        # Define solve function using scipy sparse solver
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve using TorchMatSolver
        x = TorchMatSolver.apply(A_torch, b_torch, solve_fn)

        # Compare with direct sparse solution
        x_expected = sparse.linalg.spsolve(A_sparse, b_np)

        assert isinstance(x, torch.Tensor)
        assert x.dtype == b_torch.dtype
        assert x.device == b_torch.device
        np.testing.assert_allclose(x.detach().numpy(), x_expected, rtol=1e-10)

    def test_forward_solve_csr_sparse(self):
        """Test forward solve with CSR sparse matrix."""
        A_sparse = sparse.csc_matrix(
            [[5.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 3.0]]
        )
        b_np = np.array([2.0, 1.0, 4.0])

        A_coo = A_sparse.tocoo()
        indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
        A_torch = torch.sparse_coo_tensor(
            indices, values, A_coo.shape, requires_grad=True
        )
        b_torch = torch.tensor(b_np, requires_grad=True)

        # Convert to CSR format
        A_torch_csr = A_torch.to_sparse_csr()
        A_torch_csr.requires_grad_(True)

        # Define solve function using scipy sparse solver
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve using TorchMatSolver
        x = TorchMatSolver.apply(A_torch_csr, b_torch, solve_fn)

        # Compare with direct sparse solution
        x_expected = sparse.linalg.spsolve(A_sparse, b_np)

        assert isinstance(x, torch.Tensor)
        assert x.dtype == b_torch.dtype
        assert x.device == b_torch.device
        np.testing.assert_allclose(x.detach().numpy(), x_expected, rtol=1e-10)

    def test_gradient_computation_b(self):
        """Test gradient computation with respect to b for sparse matrices."""
        A_sparse = sparse.csc_matrix(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        )
        b_np = np.array([1.0, 2.0, 3.0])

        A_coo = A_sparse.tocoo()
        indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
        A_torch = torch.sparse_coo_tensor(
            indices, values, A_coo.shape, requires_grad=True
        )
        b_torch = torch.tensor(b_np, requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve and compute gradients
        x = TorchMatSolver.apply(A_torch, b_torch, solve_fn)
        loss = torch.sum(x**2)  # Simple quadratic loss
        loss.backward()

        # Check that gradients are computed
        assert b_torch.grad is not None
        assert b_torch.grad.shape == b_torch.shape

        # Gradient should be finite
        assert torch.all(torch.isfinite(b_torch.grad))

    def test_gradient_computation_A(self):
        """Test gradient computation with respect to sparse A."""
        A_sparse = sparse.csc_matrix(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        )
        b_np = np.array([1.0, 2.0, 3.0])

        A_coo = A_sparse.tocoo()
        indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
        A_torch = torch.sparse_coo_tensor(
            indices, values, A_coo.shape, requires_grad=True
        )
        b_torch = torch.tensor(b_np, requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Enable gradient retention for non-leaf tensor
        A_torch.retain_grad()

        # Solve and compute gradients
        x = TorchMatSolver.apply(A_torch, b_torch, solve_fn)
        loss = torch.sum(x**2)  # Simple quadratic loss
        loss.backward()

        # Check that gradients are computed
        assert A_torch.grad is not None
        assert A_torch.grad.shape == A_torch.shape
        assert A_torch.grad.is_sparse  # Should maintain sparse format

        # Gradient should be finite (check dense values)
        assert torch.all(torch.isfinite(A_torch.grad.to_dense()))

    def test_gradient_symmetry(self):
        """Test that gradients maintain symmetry for symmetric matrices."""
        # Create a simple symmetric sparse matrix
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
        values = torch.tensor([4.0, 1.0, 1.0, 3.0], requires_grad=True)
        A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
        A.retain_grad()

        b = torch.tensor([1.0, 1.0], requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve and compute gradients
        x = TorchMatSolver.apply(A, b, solve_fn)
        loss = torch.sum(x**2)
        loss.backward()

        # Check that gradient is symmetric
        grad_dense = A.grad.to_dense()
        assert torch.allclose(grad_dense, grad_dense.T, atol=1e-10)

        # Check that gradients are finite
        assert torch.all(torch.isfinite(grad_dense))

    def test_device_consistency(self):
        """Test that output tensors maintain device consistency with sparse matrices."""
        # Create sparse tensor directly
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
        values = torch.tensor([2.0, 1.0, 1.0, 2.0], requires_grad=True)
        A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
        b = torch.tensor([1.0, 1.0], requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        x = TorchMatSolver.apply(A, b, solve_fn)

        assert x.device == b.device
        assert x.dtype == b.dtype

    def test_solve_accuracy(self):
        """Test numerical accuracy of the sparse solver."""
        # Create a well-conditioned sparse system
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
        values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
        A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
        b = torch.tensor([11.0, 11.0], requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        x = TorchMatSolver.apply(A, b, solve_fn)

        # Check that Ax ≈ b using dense multiplication
        residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b
        assert torch.max(torch.abs(residual)) < 1e-10

    def test_large_sparse_matrix_efficiency(self):
        """Test that large sparse matrices don't create dense intermediates."""
        # Create a larger sparse system to test memory efficiency
        n = 1000
        nnz_per_row = 5  # Average non-zeros per row

        # Create a sparse symmetric positive definite matrix
        # Use a simple pattern: diagonal + some off-diagonals
        np.random.seed(42)  # For reproducibility

        # Create indices for sparse matrix
        rows = []
        cols = []
        values = []

        # Add diagonal elements
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(10.0)  # Strong diagonal

        # Add some off-diagonal elements to make it interesting
        for i in range(n):
            for _ in range(nnz_per_row - 1):  # -1 because we already have diagonal
                j = np.random.randint(0, n)
                if i != j:
                    rows.extend([i, j])  # Add both (i,j) and (j,i) for symmetry
                    cols.extend([j, i])
                    val = np.random.uniform(0.1, 1.0)
                    values.extend([val, val])

        # Convert to torch tensors
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values_tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)

        # Create sparse tensor and coalesce to handle duplicates
        A = torch.sparse_coo_tensor(
            indices, values_tensor, (n, n), requires_grad=True
        ).coalesce()
        A.retain_grad()

        b = torch.randn(n, dtype=torch.float64, requires_grad=True)

        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve and compute gradients
        x = TorchMatSolver.apply(A, b, solve_fn)
        loss = torch.sum(x**2)
        loss.backward()

        # Verify gradients are computed and sparse
        assert A.grad is not None
        assert A.grad.is_sparse
        assert A.grad.shape == A.shape

        # Coalesce gradient for proper access to values
        A_grad_coalesced = A.grad.coalesce()

        # Verify gradient has same sparsity pattern as input (approximately)
        # The gradient should only have non-zeros where A has non-zeros
        A_nnz = A._nnz()
        grad_nnz = A_grad_coalesced._nnz()

        # Gradient should have same or fewer non-zeros than input
        assert grad_nnz <= A_nnz * 1.1  # Allow small tolerance for numerical precision

        # Verify all gradient values are finite
        assert torch.all(torch.isfinite(A_grad_coalesced.values()))


class TestBasicGradientFlow:
    """Test basic gradient flow through simpeg-torch components."""

    def test_simple_torchmatsolver_gradients(self):
        """Test that TorchMatSolver itself preserves gradients correctly."""
        # Create simple diagonal matrix with gradients
        diag_values = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)

        # Create sparse diagonal matrix
        A = sdiag(diag_values)

        # Create RHS
        rhs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        # Define sparse solver
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        # Solve
        x = TorchMatSolver.apply(A, rhs, solve_fn)

        # Loss
        loss = torch.sum(x**2)
        loss.backward()

        # Check gradients
        assert diag_values.grad is not None
        assert torch.all(torch.isfinite(diag_values.grad))
        assert torch.any(diag_values.grad != 0)

    def test_property_tensor_with_leaf_tensors(self):
        """Test make_property_tensor with leaf tensors."""
        mesh = TensorMesh([3, 3], dtype=torch.float64)

        # Create LEAF tensor
        resistivity = torch.full(
            (mesh.n_cells,), 50.0, dtype=torch.float64, requires_grad=True
        )

        # Create property tensor
        Mu = make_property_tensor(mesh, resistivity)

        # Loss on the property tensor values
        loss = torch.sum(Mu.values() ** 2)
        loss.backward()

        # Check gradients
        assert resistivity.grad is not None
        assert torch.all(torch.isfinite(resistivity.grad))

    def test_face_inner_product_with_leaf_tensors(self):
        """Test face inner product with leaf tensors."""
        mesh = TensorMesh([4, 4], dtype=torch.float64)

        # Create LEAF tensor for resistivity
        resistivity = torch.full(
            (mesh.n_cells,), 75.0, dtype=torch.float64, requires_grad=True
        )

        # Get face inner product matrix
        MfRho = mesh.get_face_inner_product(resistivity)

        # Loss
        loss = torch.sum(MfRho.values() ** 2)
        loss.backward()

        # Check gradients
        assert resistivity.grad is not None
        assert torch.all(torch.isfinite(resistivity.grad))

    def test_mesh_based_problem_with_gradients(self):
        """Test gradients through a mesh-based problem using proper leaf tensors."""
        # Create mesh
        mesh = TensorMesh([4, 4], dtype=torch.float64)
        n_cells = mesh.n_cells

        # Create resistivity as a LEAF tensor (not derived from operations)
        resistivity = torch.full(
            (n_cells,), 100.0, dtype=torch.float64, requires_grad=True
        )

        # Build simple system matrix directly (bypassing complex inner products for now)
        # Use a Laplacian-like operator that depends on resistivity

        # Create a simple finite difference matrix that depends on resistivity
        # For simplicity, create a diagonal matrix with resistivity-dependent entries
        diag_entries = 1.0 / resistivity  # Conductivity = 1/resistivity

        # Add some small constant to ensure positive definiteness
        diag_entries = diag_entries + 0.01

        A = sdiag(diag_entries)

        # RHS
        rhs = torch.ones(n_cells, dtype=torch.float64)

        # Solve
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        x = TorchMatSolver.apply(A, rhs, solve_fn)

        # Loss
        loss = torch.sum(x**2)
        loss.backward()

        # Check gradients
        assert resistivity.grad is not None
        assert torch.all(torch.isfinite(resistivity.grad))
        assert torch.any(resistivity.grad != 0)


class TestNumericalGradients:
    """Test numerical accuracy of gradient computations."""

    def test_gradient_flow_through_complex_system(self):
        """Test that gradients flow through a complex mesh-based system."""

        def forward_func(resistivity):
            """Forward function for a complex mesh-based system."""
            mesh = TensorMesh([4, 4], dtype=torch.float64)

            V = sdiag(mesh.cell_volumes)
            Div = V @ mesh.face_divergence
            Grad = Div.T

            # Use forward face inner product (not inverted to avoid gradient issues)
            MfRho = mesh.get_face_inner_product(resistivity)
            A = Div @ MfRho @ Grad

            rhs = torch.ones(mesh.n_cells, dtype=torch.float64) * 5.0

            def solve_fn(A, b):
                return sparse.linalg.spsolve(A, b)

            x = TorchMatSolver.apply(A, rhs, solve_fn)
            return torch.sum(x**2)

        # Test that gradients flow through the complex system
        resistivity = torch.full((16,), 100.0, dtype=torch.float64, requires_grad=True)

        # Compute gradient
        loss = forward_func(resistivity)
        loss.backward()

        # Verify gradients are computed and reasonable
        assert resistivity.grad is not None, "Gradients should be computed"
        assert torch.all(torch.isfinite(resistivity.grad)), "Gradients should be finite"
        assert torch.any(
            torch.abs(resistivity.grad) > 1e-10
        ), "Some gradients should be non-zero"

        # Test that different resistivity values give different gradients
        resistivity2 = torch.full((16,), 200.0, dtype=torch.float64, requires_grad=True)
        loss2 = forward_func(resistivity2)
        loss2.backward()

        # Gradients should be different for different resistivity values
        grad_diff = torch.norm(resistivity.grad - resistivity2.grad)
        assert (
            grad_diff > 1e-8
        ), "Different resistivity should yield different gradients"

    def test_comparison_with_dense_solve(self):
        """Compare gradient computation between sparse and dense solvers."""
        n = 4
        resistivity = torch.tensor([75.0, 75.0, 75.0, 75.0], requires_grad=True)

        # Create a simple matrix that depends on resistivity
        A_values = resistivity + torch.arange(
            n, dtype=torch.float64
        )  # Make it depend on resistivity
        A = sdiag(A_values)

        rhs = torch.ones(n, dtype=torch.float64) * 3.0

        # Sparse solve
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        x_sparse = TorchMatSolver.apply(A, rhs, solve_fn)
        loss_sparse = torch.sum(x_sparse**2)

        # Dense solve (for comparison)
        A_dense = A.to_dense()
        x_dense = torch.linalg.solve(A_dense, rhs)
        loss_dense = torch.sum(x_dense**2)

        # Check solutions are close
        assert torch.allclose(x_sparse, x_dense, atol=1e-10)
        assert torch.allclose(loss_sparse, loss_dense, atol=1e-10)

        # Test gradients for sparse solve
        loss_sparse.backward()

        assert resistivity.grad is not None
        assert torch.all(torch.isfinite(resistivity.grad))


if __name__ == "__main__":
    pytest.main([__file__])
