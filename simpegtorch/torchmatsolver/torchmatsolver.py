import torch
import numpy as np
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer
from scipy import sparse

## This is a wrapper for scipy sparse solvers to use with torch tensors.
# It natively supports batching multiple RHS vectors with proper gradient computation


class TorchMatSolver(Function):
    @staticmethod
    def forward(ctx, A, b, solve_fn=None):
        """
        Forward pass to solve Ax = b using scipy sparse solver.
        Supports batching across both A matrices and B vectors with full Cartesian product.

        Parameters:
        -----------
        A : torch.Tensor
            Sparse matrix: (n, n) for single matrix or (batch_A, n, n) for batched matrices
        b : torch.Tensor
            RHS vector(s): (n,) for single vector or (batch_B, n) for batched vectors
        solve_fn : callable, optional
            Scipy sparse solver function. Defaults to sparse.linalg.spsolve

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
        if solve_fn is None:
            solve_fn = sparse.linalg.spsolve

        # Determine batching dimensions
        A_is_batched = A.ndim == 3  # (batch_A, n, n)
        b_is_batched = b.ndim == 2  # (batch_B, n)

        if not A_is_batched and not b_is_batched:
            # Single A, single b: (n,n) × (n,) → (n,)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
            b_np = b.cpu().numpy()
            x_np = solve_fn(A_np_csc, b_np)
            x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)

        elif not A_is_batched and b_is_batched:
            # Single A, batched b: (n,n) × (batch_B,n) → (batch_B,n)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
            b_np = b.cpu().numpy()

            x_list = []
            for i in range(b_np.shape[0]):
                x_i_np = solve_fn(A_np_csc, b_np[i])
                x_list.append(x_i_np)

            x_np = torch.stack([torch.from_numpy(x_i) for x_i in x_list])
            x = x_np.to(dtype=b.dtype, device=b.device)

        elif A_is_batched and not b_is_batched:
            # Batched A, single b: (batch_A,n,n) × (n,) → (batch_A,n)
            b_np = b.cpu().numpy()

            x_list = []
            for i in range(A.shape[0]):
                A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                x_i_np = solve_fn(A_i_np_csc, b_np)
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
            for i in range(batch_A):
                A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                x_i_list = []
                for j in range(batch_B):
                    x_ij_np = solve_fn(A_i_np_csc, b_np[j])
                    x_i_list.append(x_ij_np)
                x_i = torch.stack([torch.from_numpy(x_ij) for x_ij in x_i_list])
                x_list.append(x_i)

            x_np = torch.stack(x_list)
            x = x_np.to(dtype=b.dtype, device=b.device)

        # Save tensors for backward pass
        ctx.save_for_backward(A, b, x)
        ctx.solve_fn = solve_fn
        ctx.needs_grad_A = A.requires_grad
        ctx.needs_grad_b = b.requires_grad
        ctx.A_is_batched = A_is_batched
        ctx.b_is_batched = b_is_batched

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
        solve_fn = ctx.solve_fn
        A_is_batched = ctx.A_is_batched
        b_is_batched = ctx.b_is_batched
        grad_A = None
        grad_b = None

        grad_output_np = grad_output.cpu().numpy()

        if not A_is_batched and not b_is_batched:
            # Single A, single b: grad(n) → grad_b(n), grad_A(n,n)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
            grad_b_np = solve_fn(A_np_csc, grad_output_np)
            grad_b = torch.from_numpy(grad_b_np).to(
                dtype=grad_output.dtype, device=grad_output.device
            )

            if ctx.needs_grad_A:
                grad_A = -sparsified_outer(A, grad_b, x)

        elif not A_is_batched and b_is_batched:
            # Single A, batched b: grad(batch_B,n) → grad_b(batch_B,n), grad_A(n,n)
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

            grad_b_list = []
            for i in range(grad_output_np.shape[0]):
                grad_b_i_np = solve_fn(A_np_csc, grad_output_np[i])
                grad_b_list.append(grad_b_i_np)

            grad_b_np = torch.stack(
                [torch.from_numpy(grad_b_i) for grad_b_i in grad_b_list]
            )
            grad_b = grad_b_np.to(dtype=grad_output.dtype, device=grad_output.device)

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
                A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                grad_b_i_np = solve_fn(A_i_np_csc, grad_output_np[i])
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
                    A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                    grad_b_ij_np = solve_fn(A_i_np_csc, grad_output_np[i, j])
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
                        A_i_np_csc = torch_tensor_to_sp(A[i], sp_type="csc")
                        grad_b_ij_np = solve_fn(A_i_np_csc, grad_output_np[i, j])
                        grad_b_ij = torch.from_numpy(grad_b_ij_np).to(A.device)
                        grad_A_ij = -sparsified_outer(A[i], grad_b_ij, x[i, j])
                        if grad_A_i is None:
                            grad_A_i = grad_A_ij
                        else:
                            grad_A_i = grad_A_i + grad_A_ij
                    grad_A_list.append(grad_A_i)
                grad_A = torch.stack(grad_A_list)

        return grad_A, grad_b, None  # None for solve_fn as it's not a tensor input


def batched_sparse_solve(A, b, solve_fn=None):
    """
    Convenient wrapper function for Cartesian product sparse solving.

    Parameters:
    -----------
    A : torch.Tensor
        Sparse matrix: (n, n) for single matrix or (batch_A, n, n) for batched matrices
    b : torch.Tensor
        RHS vector(s): (n,) for single vector or (batch_B, n) for batched vectors
    solve_fn : callable, optional
        Scipy sparse solver function. Defaults to sparse.linalg.spsolve

    Returns:
    --------
    x : torch.Tensor
        Solution tensor with Cartesian product batching:
        - A(n,n) × b(n) → x(n)
        - A(n,n) × b(batch_B,n) → x(batch_B,n)
        - A(batch_A,n,n) × b(n) → x(batch_A,n)
        - A(batch_A,n,n) × b(batch_B,n) → x(batch_A,batch_B,n)
    """
    return TorchMatSolver.apply(A, b, solve_fn)
