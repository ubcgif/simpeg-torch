import torch
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer
from scipy import sparse

## This is a wrapper for scipy sparse solvers to use with torch tensors.
# It natively supports batching multiple RHS vectors with proper gradient computation


class TorchMatSolverBatched(Function):
    @staticmethod
    def forward(ctx, A, b, solve_fn):
        """
        Forward pass to solve Ax = b using scipy sparse solver.
        Natively supports batching multiple RHS vectors.

        Parameters:
        -----------
        A : torch.Tensor
            Sparse matrix (n x n)
        b : torch.Tensor
            RHS vector(s) - can be (n,) for single RHS or (batch_size, n) for multiple RHS
        solve_fn : callable
            Scipy sparse solver function (e.g., sparse.linalg.spsolve)

        Returns:
        --------
        x : torch.Tensor
            Solution vector(s) - same shape as b
        """
        # Convert A to sparse CSC format once
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

        # Handle both single and batched RHS
        if b.ndim == 1:
            # Single RHS vector
            b_np = b.cpu().numpy()
            x_np = solve_fn(A_np_csc, b_np)
            x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)
        else:
            # Multiple RHS vectors (batch_size, n)
            b_np = b.cpu().numpy()

            # Solve for each RHS vector
            x_list = []
            for i in range(b_np.shape[0]):
                x_i_np = solve_fn(A_np_csc, b_np[i])
                x_list.append(x_i_np)

            x_np = torch.stack([torch.from_numpy(x_i) for x_i in x_list])
            x = x_np.to(dtype=b.dtype, device=b.device)

        # Save tensors for backward pass
        ctx.save_for_backward(A, b, x)
        ctx.solve_fn = solve_fn
        ctx.needs_grad_A = A.requires_grad
        ctx.needs_grad_b = b.requires_grad

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradients for batched operations.

        For the equation Ax = b, we have:
        - grad_b = A^{-1} * grad_output  (since A is symmetric)
        - grad_A = -grad_b * x^T  (if A requires gradients)
        """
        A, b, x = ctx.saved_tensors
        solve_fn = ctx.solve_fn
        grad_A = None
        grad_b = None

        # Convert A to sparse CSC format
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

        # Compute grad_b = A^{-1} * grad_output
        if grad_output.ndim == 1:
            # Single RHS gradient
            grad_output_np = grad_output.cpu().numpy()
            grad_b_np = solve_fn(A_np_csc, grad_output_np)
            grad_b = torch.from_numpy(grad_b_np).to(
                dtype=grad_output.dtype, device=grad_output.device
            )
        else:
            # Multiple RHS gradients (batch_size, n)
            grad_output_np = grad_output.cpu().numpy()

            # Solve for each gradient
            grad_b_list = []
            for i in range(grad_output_np.shape[0]):
                grad_b_i_np = solve_fn(A_np_csc, grad_output_np[i])
                grad_b_list.append(grad_b_i_np)

            grad_b_np = torch.stack(
                [torch.from_numpy(grad_b_i) for grad_b_i in grad_b_list]
            )
            grad_b = grad_b_np.to(dtype=grad_output.dtype, device=grad_output.device)

        if ctx.needs_grad_A:
            # Compute grad_A = -grad_b * x^T
            # For batched operations, we need to sum over the batch dimension
            if grad_b.ndim == 1 and x.ndim == 1:
                # Single RHS case
                grad_A = -sparsified_outer(A, grad_b, x)
            else:
                # Batched case - sum gradients across batch dimension
                grad_A = None
                for i in range(grad_b.shape[0]):
                    grad_A_i = -sparsified_outer(A, grad_b[i], x[i])
                    if grad_A is None:
                        grad_A = grad_A_i
                    else:
                        grad_A = grad_A + grad_A_i

        return grad_A, grad_b, None  # None for solve_fn as it's not a tensor input


def batched_sparse_solve(A, b, solve_fn=None):
    """
    Convenient wrapper function for batched sparse solving.

    Parameters:
    -----------
    A : torch.Tensor
        Sparse matrix (n x n)
    b : torch.Tensor
        RHS vector(s) - can be (n,) for single RHS or (batch_size, n) for multiple RHS
    solve_fn : callable, optional
        Scipy sparse solver function. Defaults to sparse.linalg.spsolve

    Returns:
    --------
    x : torch.Tensor
        Solution vector(s) - same shape as b
    """
    if solve_fn is None:
        solve_fn = sparse.linalg.spsolve
    return TorchMatSolverBatched.apply(A, b, solve_fn)
