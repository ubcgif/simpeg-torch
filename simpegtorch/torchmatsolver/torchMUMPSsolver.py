import torch
import numpy as np
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer

try:
    from pymatsolver import Mumps

    _mumps_available = True
except ImportError:
    _mumps_available = False

## This is a wrapper for pymatsolver MUMPS solver to use with torch tensors.
# It natively supports batching multiple RHS vectors with proper gradient computation


class TorchMUMPSsolver(Function):
    @staticmethod
    def forward(ctx, A, b):
        """
        Forward pass to solve Ax = b using pymatsolver MUMPS solver.
        Supports batching across both A matrices and B vectors with full Cartesian product.

        Parameters:
        -----------
        A : torch.Tensor
            Sparse matrix: (n, n) for single matrix or (batch_A, n, n) for batched matrices
        b : torch.Tensor
            RHS vector(s): (n,) for single vector or (batch_B, n) for batched vectors

        Batching behavior (Cartesian product):
        - Single A, Single b: A(n,n), b(n) → x(n)
        - Single A, Batched b: A(n,n), b(batch_B,n) → x(batch_B,n)
        - Batched A, Single b: A(batch_A,n,n), b(n) → x(batch_A,n)
        - Batched A, Batched b: A(batch_A,n,n), b(batch_B,n) → x(batch_A,batch_B,n)
          Solves every A[i] with every b[j] for all i,j combinations

        Returns:
        --------
        x : torch.Tensor
            Solution tensor with shape determined by Cartesian product of batch dimensions
        """
        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        # Determine batching dimensions
        A_is_batched = A.ndim == 3  # (batch_A, n, n)
        b_is_batched = b.ndim == 2  # (batch_B, n)

        if not A_is_batched and not b_is_batched:
            # Single A, single b: (n,n) × (n,) → (n,)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
            mumps_solver = Mumps(A_np_csc)
            b_np = b.cpu().numpy()
            x_np = mumps_solver.solve(b_np)
            x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)

        elif not A_is_batched and b_is_batched:
            # Single A, batched b: (n,n) × (batch_B,n) → (batch_B,n)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
            mumps_solver = Mumps(A_np_csc)
            b_np = b.cpu().numpy()

            # MUMPS can handle multiple RHS efficiently
            x_np = mumps_solver.solve(b_np.T)  # Transpose for column-major

            if x_np.ndim == 1:
                # Single RHS case (batch_size=1)
                x = (
                    torch.from_numpy(x_np)
                    .to(dtype=b.dtype, device=b.device)
                    .unsqueeze(0)
                )
            else:
                # Multiple RHS case - transpose back to (batch_size, n)
                x = torch.from_numpy(x_np.T).to(dtype=b.dtype, device=b.device)

        elif A_is_batched and not b_is_batched:
            # Batched A, single b: (batch_A,n,n) × (n,) → (batch_A,n)
            b_np = b.cpu().numpy()

            x_list = []
            mumps_solvers = []
            for i in range(A.shape[0]):
                A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                mumps_solver_i = Mumps(A_i_np_csc)
                mumps_solvers.append(mumps_solver_i)
                x_i_np = mumps_solver_i.solve(b_np)
                x_list.append(x_i_np)

            x_np = torch.stack([torch.from_numpy(x_i) for x_i in x_list])
            x = x_np.to(dtype=b.dtype, device=b.device)

        else:
            # Batched A, batched b: (batch_A,n,n) × (batch_B,n) → (batch_A,batch_B,n)
            # Solve Cartesian product: every A[i] with every b[j]
            batch_A, n, _ = A.shape
            batch_B, _ = b.shape

            b_np = b.cpu().numpy()

            x_list = []
            mumps_solvers = []

            # Create solvers for each A matrix
            for i in range(batch_A):
                A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                mumps_solver_i = Mumps(A_i_np_csc)
                mumps_solvers.append(mumps_solver_i)

            # Solve Cartesian product
            for i in range(batch_A):
                x_i_list = []
                for j in range(batch_B):
                    x_ij_np = mumps_solvers[i].solve(b_np[j])
                    x_i_list.append(x_ij_np)
                x_i = torch.stack([torch.from_numpy(x_ij) for x_ij in x_i_list])
                x_list.append(x_i)

            x_np = torch.stack(x_list)
            x = x_np.to(dtype=b.dtype, device=b.device)

        # Save tensors for backward pass
        ctx.save_for_backward(A, b, x)
        ctx.needs_grad_A = A.requires_grad
        ctx.needs_grad_b = b.requires_grad
        ctx.A_is_batched = A_is_batched
        ctx.b_is_batched = b_is_batched

        # Store solvers for reuse in backward pass
        if not A_is_batched:
            ctx.mumps_solver = mumps_solver
            ctx.mumps_solvers = None
        else:
            ctx.mumps_solver = None
            ctx.mumps_solvers = mumps_solvers

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradients for Cartesian product batched operations.

        For the equation Ax = b, we have:
        - grad_b = A^{-1} * grad_output  (since A is symmetric)
        - grad_A = -grad_b * x^T  (if A requires gradients)

        Handles all batching combinations with proper gradient aggregation.
        """
        A, b, x = ctx.saved_tensors
        A_is_batched = ctx.A_is_batched
        b_is_batched = ctx.b_is_batched
        mumps_solver = ctx.mumps_solver
        mumps_solvers = ctx.mumps_solvers
        grad_A = None
        grad_b = None

        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        grad_output_np = grad_output.cpu().numpy()

        if not A_is_batched and not b_is_batched:
            # Single A, single b: grad(n) → grad_b(n), grad_A(n,n)
            grad_b_np = mumps_solver.solve(grad_output_np)
            grad_b = torch.from_numpy(grad_b_np).to(
                dtype=grad_output.dtype, device=grad_output.device
            )

            if ctx.needs_grad_A:
                grad_A = -sparsified_outer(A, grad_b, x)

        elif not A_is_batched and b_is_batched:
            # Single A, batched b: grad(batch_B,n) → grad_b(batch_B,n), grad_A(n,n)
            # MUMPS can handle multiple RHS efficiently
            grad_b_np = mumps_solver.solve(grad_output_np.T)

            if grad_b_np.ndim == 1:
                # Single RHS case
                grad_b = (
                    torch.from_numpy(grad_b_np)
                    .to(dtype=grad_output.dtype, device=grad_output.device)
                    .unsqueeze(0)
                )
            else:
                # Multiple RHS case - transpose back
                grad_b = torch.from_numpy(grad_b_np.T).to(
                    dtype=grad_output.dtype, device=grad_output.device
                )

            if ctx.needs_grad_A:
                # Sum gradients across batch dimension
                grad_A = None
                for i in range(grad_b.shape[0]):
                    grad_A_i = -sparsified_outer(A, grad_b[i], x[i])
                    if grad_A is None:
                        grad_A = grad_A_i
                    else:
                        grad_A = grad_A + grad_A_i

        elif A_is_batched and not b_is_batched:
            # Batched A, single b: grad(batch_A,n) → grad_b(n), grad_A(batch_A,n,n)
            grad_b_list = []

            for i in range(grad_output_np.shape[0]):
                grad_b_i_np = mumps_solvers[i].solve(grad_output_np[i])
                grad_b_list.append(grad_b_i_np)

            # Sum across A batch dimension to get grad_b
            grad_b_np = np.sum(grad_b_list, axis=0)
            grad_b = torch.from_numpy(grad_b_np).to(
                dtype=grad_output.dtype, device=grad_output.device
            )

            if ctx.needs_grad_A:
                grad_A_list = []
                for i in range(A.shape[0]):
                    grad_A_i = -sparsified_outer(
                        A[i], torch.from_numpy(grad_b_list[i]).to(A.device), x[i]
                    )
                    grad_A_list.append(grad_A_i)
                grad_A = torch.stack(grad_A_list)

        else:
            # Batched A, batched b: grad(batch_A,batch_B,n) → grad_b(batch_B,n), grad_A(batch_A,n,n)
            batch_A, batch_B, n = grad_output.shape

            # Compute grad_b by summing over A dimension
            grad_b_list = []
            for j in range(batch_B):
                grad_b_j_list = []
                for i in range(batch_A):
                    grad_b_ij_np = mumps_solvers[i].solve(grad_output_np[i, j])
                    grad_b_j_list.append(grad_b_ij_np)
                # Sum over A dimension
                grad_b_j_np = np.sum(grad_b_j_list, axis=0)
                grad_b_list.append(grad_b_j_np)

            grad_b_np = torch.stack(
                [torch.from_numpy(grad_b_j) for grad_b_j in grad_b_list]
            )
            grad_b = grad_b_np.to(dtype=grad_output.dtype, device=grad_output.device)

            if ctx.needs_grad_A:
                # Compute grad_A by summing over B dimension
                grad_A_list = []
                for i in range(batch_A):
                    grad_A_i = None
                    for j in range(batch_B):
                        grad_b_ij_np = mumps_solvers[i].solve(grad_output_np[i, j])
                        grad_b_ij = torch.from_numpy(grad_b_ij_np).to(A.device)
                        grad_A_ij = -sparsified_outer(A[i], grad_b_ij, x[i, j])
                        if grad_A_i is None:
                            grad_A_i = grad_A_ij
                        else:
                            grad_A_i = grad_A_i + grad_A_ij
                    grad_A_list.append(grad_A_i)
                grad_A = torch.stack(grad_A_list)

        return grad_A, grad_b


def batched_mumps_solve(A, b):
    """
    Convenient wrapper function for Cartesian product MUMPS solving.

    Parameters:
    -----------
    A : torch.Tensor
        Sparse matrix: (n, n) for single matrix or (batch_A, n, n) for batched matrices
    b : torch.Tensor
        RHS vector(s): (n,) for single vector or (batch_B, n) for batched vectors

    Returns:
    --------
    x : torch.Tensor
        Solution tensor with Cartesian product batching:
        - A(n,n) × b(n) → x(n)
        - A(n,n) × b(batch_B,n) → x(batch_B,n)
        - A(batch_A,n,n) × b(n) → x(batch_A,n)
        - A(batch_A,n,n) × b(batch_B,n) → x(batch_A,batch_B,n)
    """
    return TorchMUMPSsolver.apply(A, b)
