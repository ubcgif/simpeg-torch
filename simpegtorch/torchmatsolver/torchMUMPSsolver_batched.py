import torch
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer

try:
    from pymatsolver import Mumps

    _mumps_available = True
except ImportError:
    _mumps_available = False

## This is a wrapper for pymatsolver MUMPS solver to use with torch tensors.
# It natively supports batching multiple RHS vectors with proper gradient computation


class TorchMUMPSsolverBatched(Function):
    @staticmethod
    def forward(ctx, A, b):
        """
        Forward pass to solve Ax = b using pymatsolver MUMPS solver.
        Natively supports batching multiple RHS vectors.

        Parameters:
        -----------
        A : torch.Tensor
            Sparse matrix (n x n)
        b : torch.Tensor
            RHS vector(s) - can be (n,) for single RHS or (batch_size, n) for multiple RHS

        Returns:
        --------
        x : torch.Tensor
            Solution vector(s) - same shape as b
        """
        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        # Convert A to sparse CSC format once
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

        # Create MUMPS solver instance and factor once
        mumps_solver = Mumps(A_np_csc)

        # Handle both single and batched RHS
        if b.ndim == 1:
            # Single RHS vector
            b_np = b.cpu().numpy()
            x_np = mumps_solver.solve(b_np)
            x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)
        else:
            # Multiple RHS vectors (batch_size, n)
            b_np = b.cpu().numpy()

            # MUMPS expects column-major format for multiple RHS
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

        # Save tensors for backward pass
        ctx.save_for_backward(A, b, x)
        ctx.mumps_solver = mumps_solver  # Store solver in context for reuse
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
        mumps_solver = ctx.mumps_solver  # Retrieve saved solver from context
        grad_A = None
        grad_b = None

        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        # Compute grad_b = A^{-1} * grad_output
        if grad_output.ndim == 1:
            # Single RHS gradient
            grad_output_np = grad_output.cpu().numpy()
            grad_b_np = mumps_solver.solve(grad_output_np)
            grad_b = torch.from_numpy(grad_b_np).to(
                dtype=grad_output.dtype, device=grad_output.device
            )
        else:
            # Multiple RHS gradients (batch_size, n)
            grad_output_np = grad_output.cpu().numpy()

            # MUMPS expects column-major format
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

        return grad_A, grad_b


def batched_mumps_solve(A, b):
    """
    Convenient wrapper function for batched MUMPS solving.

    Parameters:
    -----------
    A : torch.Tensor
        Sparse matrix (n x n)
    b : torch.Tensor
        RHS vector(s) - can be (n,) for single RHS or (batch_size, n) for multiple RHS

    Returns:
    --------
    x : torch.Tensor
        Solution vector(s) - same shape as b
    """
    return TorchMUMPSsolverBatched.apply(A, b)
