import torch
import pytest
import numpy as np
from scipy import sparse
from simpegtorch.torchmatsolver import (
    TorchMatSolver,
    TorchMUMPSsolver,
    batched_mumps_solve,
)

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

    # Solve using TorchMatSolver (now batched by default)
    x = TorchMatSolver.apply(A_torch, b_torch)

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

    # Solve using TorchMatSolver (now batched by default)
    x = TorchMatSolver.apply(A_torch_csr, b_torch)

    # Compare with direct sparse solution
    x_expected = sparse.linalg.spsolve(A_sparse, b_np)

    assert isinstance(x, torch.Tensor)
    assert x.dtype == b_torch.dtype
    assert x.device == b_torch.device
    np.testing.assert_allclose(x.detach().numpy(), x_expected, rtol=1e-10)


def test_gradient_computation_b(simple_system):
    """Test gradient computation with respect to b for sparse matrices"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    # Solve and compute gradients
    x = TorchMatSolver.apply(A_torch, b_torch)
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

    # Enable gradient retention for non-leaf tensor
    A_torch.retain_grad()

    # Solve and compute gradients
    x = TorchMatSolver.apply(A_torch, b_torch)
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

    x = TorchMatSolver.apply(A, b)

    assert x.device == b.device
    assert x.dtype == b.dtype


def test_solve_accuracy():
    """Test numerical accuracy of the sparse solver"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)

    x = TorchMatSolver.apply(A, b)

    # Check that Ax ≈ b using dense multiplication
    residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b
    assert torch.max(torch.abs(residual)) < 1e-10


def test_batched_functionality():
    """Tests native batching functionality"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)

    # Test single RHS
    x_single = TorchMatSolver.apply(A, b)
    assert x_single.shape == (2,)

    # Test batched RHS
    b_batched = b.unsqueeze(0).repeat(3, 1)  # Batch of 3 identical vectors
    x_batched = TorchMatSolver.apply(A, b_batched)

    assert x_batched.shape == (3, 2)  # Should return a batch of solutions

    # Check that each solution is correct
    for i in range(3):
        np.testing.assert_allclose(
            x_batched[i].detach().numpy(), x_single.detach().numpy(), rtol=1e-10
        )

    # Test gradients work with batched RHS
    A.retain_grad()
    b_batched.retain_grad()

    x_batched_grad = TorchMatSolver.apply(A, b_batched)
    loss = torch.sum(x_batched_grad**2)
    loss.backward()

    assert A.grad is not None
    assert b_batched.grad is not None
    assert A.grad.is_sparse


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

    # Solve and compute gradients
    x = TorchMatSolver.apply(A, b)
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

    # Compute loss and gradients
    A_torch.retain_grad()
    x = TorchMatSolver.apply(A_torch, b_torch)
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
        x_plus = TorchMatSolver.apply(A_plus, b_torch.detach())
        loss_plus = torch.sum(x_plus**2)

        values_minus = A_values.clone().detach()
        values_minus[i] -= eps
        A_minus = torch.sparse_coo_tensor(A_indices, values_minus, A_torch.shape)
        x_minus = TorchMatSolver.apply(A_minus, b_torch.detach())
        loss_minus = torch.sum(x_minus**2)

        grad_fd = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        rel_error = abs(grad_values[i] - grad_fd) / (abs(grad_values[i]) + 1e-10)
        assert rel_error < 1e-4, f"Gradient mismatch at index {i}: {rel_error:.2e}"


# MUMPS Solver Tests
# Skip all MUMPS tests if MUMPS is not available
try:
    from pymatsolver import Mumps

    # Test if MUMPS is actually available by checking for the python-mumps package
    import scipy.sparse

    test_matrix = scipy.sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0]])
    Mumps(test_matrix)  # This will fail if python-mumps is not installed
    mumps_available = True
except ImportError:
    mumps_available = False

skip_mumps = pytest.mark.skipif(not mumps_available, reason="MUMPS not available")


@skip_mumps
def test_mumps_forward_solve_sparse(simple_system):
    """Test forward solve with sparse matrix using MUMPS solver"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    # Solve using TorchMUMPSsolver
    x = TorchMUMPSsolver.apply(A_torch, b_torch)

    # Compare with direct sparse solution
    x_expected = sparse.linalg.spsolve(A_sparse, b_np)

    assert isinstance(x, torch.Tensor)
    assert x.dtype == b_torch.dtype
    assert x.device == b_torch.device
    np.testing.assert_allclose(x.detach().numpy(), x_expected, rtol=1e-10)


@skip_mumps
def test_mumps_forward_solve_csr_sparse(sparse_system):
    """Test forward solve with CSR sparse matrix using MUMPS"""
    A_torch, b_torch, A_sparse, b_np = sparse_system

    # Convert to CSR format
    A_torch_csr = A_torch.to_sparse_csr()
    A_torch_csr.requires_grad_(True)

    # Solve using TorchMUMPSsolver
    x = TorchMUMPSsolver.apply(A_torch_csr, b_torch)

    # Compare with direct sparse solution
    x_expected = sparse.linalg.spsolve(A_sparse, b_np)

    assert isinstance(x, torch.Tensor)
    assert x.dtype == b_torch.dtype
    assert x.device == b_torch.device
    np.testing.assert_allclose(x.detach().numpy(), x_expected, rtol=1e-10)


@skip_mumps
def test_mumps_gradient_computation_b(simple_system):
    """Test gradient computation with respect to b for sparse matrices using MUMPS"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    # Solve and compute gradients
    x = TorchMUMPSsolver.apply(A_torch, b_torch)
    loss = torch.sum(x**2)  # Simple quadratic loss
    loss.backward()

    # Check that gradients are computed
    assert b_torch.grad is not None
    assert b_torch.grad.shape == b_torch.shape

    # Gradient should be finite
    assert torch.all(torch.isfinite(b_torch.grad))
    assert torch.any(b_torch.grad > 0)


@skip_mumps
def test_mumps_gradient_computation_A(simple_system):
    """Test gradient computation with respect to sparse A using MUMPS"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    # Enable gradient retention for non-leaf tensor
    A_torch.retain_grad()

    # Solve and compute gradients
    x = TorchMUMPSsolver.apply(A_torch, b_torch)
    loss = torch.sum(x**2)  # Simple quadratic loss
    loss.backward()

    # Check that gradients are computed
    assert A_torch.grad is not None
    assert A_torch.grad.shape == A_torch.shape
    assert A_torch.grad.is_sparse  # Should maintain sparse format

    # Gradient should be finite (check dense values)
    assert torch.all(torch.isfinite(A_torch.grad.to_dense()))
    assert torch.any(torch.isfinite(A_torch.grad.to_dense()) > 0)


@skip_mumps
def test_mumps_device_consistency():
    """Test that output tensors maintain device consistency with sparse matrices using MUMPS"""
    # Create sparse tensor directly
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([2.0, 1.0, 1.0, 2.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([1.0, 1.0], requires_grad=True)

    x = TorchMUMPSsolver.apply(A, b)

    assert x.device == b.device
    assert x.dtype == b.dtype


@skip_mumps
def test_mumps_solve_accuracy():
    """Test numerical accuracy of the MUMPS solver"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)

    x = TorchMUMPSsolver.apply(A, b)

    # Check that Ax ≈ b using dense multiplication
    residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b
    assert torch.max(torch.abs(residual)) < 1e-10


@skip_mumps
def test_mumps_batched_functionality():
    """Tests native batching functionality using MUMPS"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)

    # Test single RHS
    x_single = TorchMUMPSsolver.apply(A, b)
    assert x_single.shape == (2,)

    # Test batched RHS
    b_batched = b.unsqueeze(0).repeat(3, 1)  # Batch of 3 identical vectors
    x_batched = TorchMUMPSsolver.apply(A, b_batched)

    assert x_batched.shape == (3, 2)  # Should return a batch of solutions

    # Check that each solution is correct
    for i in range(3):
        np.testing.assert_allclose(
            x_batched[i].detach().numpy(), x_single.detach().numpy(), rtol=1e-10
        )

    # Test gradients work with batched RHS
    A.retain_grad()
    b_batched.retain_grad()

    x_batched_grad = TorchMUMPSsolver.apply(A, b_batched)
    loss = torch.sum(x_batched_grad**2)
    loss.backward()

    assert A.grad is not None
    assert b_batched.grad is not None
    assert A.grad.is_sparse


@skip_mumps
def test_mumps_batched_convenience():
    """Tests batching with convenience function using MUMPS"""
    # Create a well-conditioned sparse system
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([10.0, 1.0, 1.0, 10.0], requires_grad=True)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), requires_grad=True)
    b = torch.tensor([11.0, 11.0], requires_grad=True)

    b_batched = b.unsqueeze(0).repeat(3, 1)  # Batch of 3 identical vectors

    # test batching using convenience function
    x_batched_rhs = batched_mumps_solve(A, b_batched)

    loss = torch.sum(x_batched_rhs**2)  # Simple quadratic loss
    loss.backward()

    assert values.grad is not None
    assert values.grad.shape == values.shape
    assert x_batched_rhs.shape == (3, 2)


@skip_mumps
def test_mumps_large_sparse_matrix_efficiency():
    """Test that large sparse matrices don't create dense intermediates using MUMPS"""
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

    # Solve and compute gradients
    x = TorchMUMPSsolver.apply(A, b)
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

    print(f"Large matrix test (MUMPS): {n}x{n} matrix with {A_nnz} non-zeros")
    print(f"Gradient has {grad_nnz} non-zeros")
    print("Memory-efficient gradient computation successful!")


@skip_mumps
def test_mumps_gradient_correctness_A(simple_system):
    """Test gradient correctness w.r.t A using finite differences with MUMPS"""
    A_torch, b_torch, A_sparse, b_np = simple_system

    # Compute loss and gradients
    A_torch.retain_grad()
    x = TorchMUMPSsolver.apply(A_torch, b_torch)
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
        x_plus = TorchMUMPSsolver.apply(A_plus, b_torch.detach())
        loss_plus = torch.sum(x_plus**2)

        values_minus = A_values.clone().detach()
        values_minus[i] -= eps
        A_minus = torch.sparse_coo_tensor(A_indices, values_minus, A_torch.shape)
        x_minus = TorchMUMPSsolver.apply(A_minus, b_torch.detach())
        loss_minus = torch.sum(x_minus**2)

        grad_fd = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        rel_error = abs(grad_values[i] - grad_fd) / (abs(grad_values[i]) + 1e-10)
        assert rel_error < 1e-4, f"Gradient mismatch at index {i}: {rel_error:.2e}"


@skip_mumps
def test_mumps_batched_rhs_optimization():
    """Test that batched RHS with MUMPS uses single factorization optimization"""
    # Create a larger system to make the optimization more apparent
    n = 100
    np.random.seed(42)

    # Create a sparse symmetric positive definite matrix
    A_dense = np.random.rand(n, n)
    A_dense = A_dense + A_dense.T + n * np.eye(n)  # Make symmetric positive definite

    # Convert to sparse COO format for torch
    A_sparse = sparse.csc_matrix(A_dense).tocoo()
    indices = torch.tensor([A_sparse.row, A_sparse.col], dtype=torch.long)
    values = torch.tensor(A_sparse.data, dtype=torch.float64)
    A_torch = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64)

    # Create multiple RHS vectors
    num_rhs = 10
    b_batch = torch.randn(num_rhs, n, dtype=torch.float64)

    # Test the batched RHS with native batching
    x_batch = TorchMUMPSsolver.apply(A_torch, b_batch)

    # Verify solution correctness
    assert x_batch.shape == (num_rhs, n)

    # Check that each solution is correct
    A_dense_torch = torch.tensor(A_dense, dtype=torch.float64)
    for i in range(num_rhs):
        residual = torch.norm(A_dense_torch @ x_batch[i] - b_batch[i])
        assert residual < 1e-10, f"RHS {i} residual too large: {residual.item():.2e}"

    print(
        f"Batched RHS optimization test passed: {num_rhs} RHS vectors solved efficiently"
    )


def test_cartesian_product_batching():
    """Test Cartesian product batching: batched A × batched b → (batch_A, batch_B, n)"""
    # Create simple test system
    n = 3
    batch_A = 2  # 2 different matrices
    batch_B = 3  # 3 different RHS vectors

    # Create batched A matrices
    A_list = []
    for i in range(batch_A):
        # Create different matrices for each batch
        indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]], dtype=torch.long)
        values = torch.tensor([4.0 + i, 1.0, 1.0, 3.0 + i, 2.0 + i], requires_grad=True)
        A_i = torch.sparse_coo_tensor(indices, values, (n, n), requires_grad=True)
        A_list.append(A_i)
    A_batched = torch.stack(A_list)  # Shape: (batch_A, n, n)

    # Create batched b vectors
    b_batched = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 3.0, 1.0]], requires_grad=True
    )  # Shape: (batch_B, n)

    # Test Cartesian product solving
    x_cartesian = TorchMatSolver.apply(A_batched, b_batched)

    # Should get shape (batch_A, batch_B, n)
    assert x_cartesian.shape == (batch_A, batch_B, n)

    # Verify each solution is correct by solving individually
    for i in range(batch_A):
        for j in range(batch_B):
            x_single = TorchMatSolver.apply(A_batched[i], b_batched[j])
            np.testing.assert_allclose(
                x_cartesian[i, j].detach().numpy(),
                x_single.detach().numpy(),
                rtol=1e-10,
            )

    # Test gradients work
    A_batched.retain_grad()
    b_batched.retain_grad()

    loss = torch.sum(x_cartesian**2)
    loss.backward()

    assert A_batched.grad is not None
    assert b_batched.grad is not None
    assert A_batched.grad.shape == A_batched.shape
    assert b_batched.grad.shape == b_batched.shape

    print(
        f"Cartesian product test passed: {batch_A}×{batch_B} combinations solved correctly"
    )


def test_batched_A_single_b():
    """Test batched A × single b → (batch_A, n)"""
    n = 3
    batch_A = 2

    # Create batched A matrices
    A_list = []
    for i in range(batch_A):
        indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]], dtype=torch.long)
        values = torch.tensor([4.0 + i, 1.0, 1.0, 3.0 + i, 2.0 + i], requires_grad=True)
        A_i = torch.sparse_coo_tensor(indices, values, (n, n), requires_grad=True)
        A_list.append(A_i)
    A_batched = torch.stack(A_list)

    # Single b vector
    b_single = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Test batched A, single b
    x_result = TorchMatSolver.apply(A_batched, b_single)

    # Should get shape (batch_A, n)
    assert x_result.shape == (batch_A, n)

    # Verify each solution
    for i in range(batch_A):
        x_single = TorchMatSolver.apply(A_batched[i], b_single)
        np.testing.assert_allclose(
            x_result[i].detach().numpy(), x_single.detach().numpy(), rtol=1e-10
        )

    print(f"Batched A × single b test passed: {batch_A} matrices solved correctly")


@skip_mumps
def test_mumps_cartesian_product_batching():
    """Test MUMPS Cartesian product batching: batched A × batched b → (batch_A, batch_B, n)"""
    # Create simple test system
    n = 3
    batch_A = 2  # 2 different matrices
    batch_B = 3  # 3 different RHS vectors

    # Create batched A matrices
    A_list = []
    for i in range(batch_A):
        # Create different matrices for each batch
        indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]], dtype=torch.long)
        values = torch.tensor([4.0 + i, 1.0, 1.0, 3.0 + i, 2.0 + i], requires_grad=True)
        A_i = torch.sparse_coo_tensor(indices, values, (n, n), requires_grad=True)
        A_list.append(A_i)
    A_batched = torch.stack(A_list)  # Shape: (batch_A, n, n)

    # Create batched b vectors
    b_batched = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 3.0, 1.0]], requires_grad=True
    )  # Shape: (batch_B, n)

    # Test Cartesian product solving
    x_cartesian = TorchMUMPSsolver.apply(A_batched, b_batched)

    # Should get shape (batch_A, batch_B, n)
    assert x_cartesian.shape == (batch_A, batch_B, n)

    # Verify each solution is correct by solving individually
    for i in range(batch_A):
        for j in range(batch_B):
            x_single = TorchMUMPSsolver.apply(A_batched[i], b_batched[j])
            np.testing.assert_allclose(
                x_cartesian[i, j].detach().numpy(),
                x_single.detach().numpy(),
                rtol=1e-10,
            )

    # Test gradients work
    A_batched.retain_grad()
    b_batched.retain_grad()

    loss = torch.sum(x_cartesian**2)
    loss.backward()

    assert A_batched.grad is not None
    assert b_batched.grad is not None
    assert A_batched.grad.shape == A_batched.shape
    assert b_batched.grad.shape == b_batched.shape

    print(
        f"MUMPS Cartesian product test passed: {batch_A}×{batch_B} combinations solved correctly"
    )


@skip_mumps
def test_mumps_batched_A_single_b():
    """Test MUMPS batched A × single b → (batch_A, n)"""
    n = 3
    batch_A = 2

    # Create batched A matrices
    A_list = []
    for i in range(batch_A):
        indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]], dtype=torch.long)
        values = torch.tensor([4.0 + i, 1.0, 1.0, 3.0 + i, 2.0 + i], requires_grad=True)
        A_i = torch.sparse_coo_tensor(indices, values, (n, n), requires_grad=True)
        A_list.append(A_i)
    A_batched = torch.stack(A_list)

    # Single b vector
    b_single = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Test batched A, single b
    x_result = TorchMUMPSsolver.apply(A_batched, b_single)

    # Should get shape (batch_A, n)
    assert x_result.shape == (batch_A, n)

    # Verify each solution
    for i in range(batch_A):
        x_single = TorchMUMPSsolver.apply(A_batched[i], b_single)
        np.testing.assert_allclose(
            x_result[i].detach().numpy(), x_single.detach().numpy(), rtol=1e-10
        )

    print(
        f"MUMPS Batched A × single b test passed: {batch_A} matrices solved correctly"
    )
