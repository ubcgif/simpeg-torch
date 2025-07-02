import torch
import pytest
import numpy as np
from scipy import sparse
from simpegtorch.torchmatsolver import TorchMatSolver

torch.set_default_dtype(torch.float64)


@pytest.fixture
def simple_system():
    """Create a simple sparse test system Ax = b"""
    # Create a simple 3x3 symmetric positive definite matrix
    A_sparse = sparse.csc_matrix([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b_np = np.array([1.0, 2.0, 3.0])

    # Create sparse tensor directly from COO format
    A_coo = A_sparse.tocoo()
    indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
    values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
    A_torch = torch.sparse_coo_tensor(indices, values, A_coo.shape, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)

    return A_torch, b_torch, A_sparse, b_np


@pytest.fixture
def sparse_system():
    """Create a sparse test system"""
    # Create a sparse symmetric positive definite matrix
    A_sparse = sparse.csc_matrix([[5.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 3.0]])
    b_np = np.array([2.0, 1.0, 4.0])

    # Create sparse tensor directly from COO format
    A_coo = A_sparse.tocoo()
    indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long)
    values = torch.tensor(A_coo.data, dtype=torch.float64, requires_grad=True)
    A_torch = torch.sparse_coo_tensor(indices, values, A_coo.shape, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)

    return A_torch, b_torch, A_sparse, b_np


def test_forward_solve_sparse(simple_system):
    """Test forward solve with sparse matrix using scipy sparse solver"""
    A_torch, b_torch, A_sparse, b_np = simple_system

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


def test_forward_solve_csr_sparse(sparse_system):
    """Test forward solve with CSR sparse matrix"""
    A_torch, b_torch, A_sparse, b_np = sparse_system

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


def test_gradient_computation_b(simple_system):
    """Test gradient computation with respect to b for sparse matrices"""
    A_torch, b_torch, A_sparse, b_np = simple_system

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


def test_gradient_computation_A(simple_system):
    """Test gradient computation with respect to sparse A"""
    A_torch, b_torch, A_sparse, b_np = simple_system

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


def test_device_consistency():
    """Test that output tensors maintain device consistency with sparse matrices"""
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


def test_solve_accuracy():
    """Test numerical accuracy of the sparse solver"""
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


def test_vmap():
    """Tests batching with vmap functionality"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)
    A_b = A.unsqueeze(0)  # Add batch dimension
    A_batched = torch.cat([A_b, A_b, A_b], dim=0)  # Batch of 3 identical matrices

    b_batched = b.unsqueeze(0).repeat(3, 1)  # Batch of 3 identical vectors

    def solve_fn(A, b):
        return sparse.linalg.spsolve(A, b)

    # Use TorchMatSolver with vmap
    x_batched = torch.vmap(lambda A, b: TorchMatSolver.apply(A, b, solve_fn))(
        A_batched, b_batched
    )

    # test batching over only rhs
    x_batched_rhs = torch.vmap(
        lambda A, b: TorchMatSolver.apply(A, b, solve_fn), in_dims=(None, 0)
    )(A, b_batched)

    # test batching over only A
    x_batched_A = torch.vmap(
        lambda A, b: TorchMatSolver.apply(A, b, solve_fn), in_dims=(0, None)
    )(A_batched, b)

    assert x_batched.shape == (3, 2)  # Should return a batch of solutions
    assert x_batched_rhs.shape == (3, 2)  # Should return a batch of solutions
    assert x_batched_A.shape == (3, 2)  # Should return a batch of solutions


def test_large_sparse_matrix_efficiency():
    """Test that large sparse matrices don't create dense intermediates"""
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

    print(f"Large matrix test: {n}x{n} matrix with {A_nnz} non-zeros")
    print(f"Gradient has {grad_nnz} non-zeros")
    print("Memory-efficient gradient computation successful!")


def test_gradient_correctness_A(simple_system):
    """Test gradient correctness w.r.t A using finite differences"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    def solve_fn(A, b):
        return sparse.linalg.spsolve(A, b)

    # Compute loss and gradients
    A_torch.retain_grad()
    x = TorchMatSolver.apply(A_torch, b_torch, solve_fn)
    loss = torch.sum(x**2)
    loss.backward()

    # Finite difference check
    eps = 1e-7
    A_values = A_torch.coalesce().values()
    A_indices = A_torch.coalesce().indices()
    grad_values = A_torch.grad.coalesce().values()

    # Test a few values
    for i in range(min(3, len(A_values))):
        # Perturb value
        values_plus = A_values.clone().detach()
        values_plus[i] += eps
        A_plus = torch.sparse_coo_tensor(A_indices, values_plus, A_torch.shape)
        x_plus = TorchMatSolver.apply(A_plus, b_torch.detach(), solve_fn)
        loss_plus = torch.sum(x_plus**2)

        values_minus = A_values.clone().detach()
        values_minus[i] -= eps
        A_minus = torch.sparse_coo_tensor(A_indices, values_minus, A_torch.shape)
        x_minus = TorchMatSolver.apply(A_minus, b_torch.detach(), solve_fn)
        loss_minus = torch.sum(x_minus**2)

        grad_fd = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        rel_error = abs(grad_values[i] - grad_fd) / (abs(grad_values[i]) + 1e-10)
        assert rel_error < 1e-4, f"Gradient mismatch at index {i}: {rel_error:.2e}"
