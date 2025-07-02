import torch
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer

## This is a wrapper for pymatsolver to use with torch tensors.
# It will also compute the gradient of the solution for backpropoagation


class TorchMatSolver(Function):
    @staticmethod
    def forward(A, b, solve_fn):
        """
        Forward pass to solve Ax = b using pymatsolver.
        """
        ## Check if A is a sparse matrix
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
        x = solve_fn(A_np_csc, b.cpu().numpy())

        return torch.tensor(x, dtype=b.dtype, device=b.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for the backward pass.
        """
        A, b, solve_fn = inputs
        x = output
        ctx.save_for_backward(A, b, x)
        ctx.solve_fn = solve_fn

        # Ensure A and b are tensors
        if not isinstance(A, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Inputs A and b must be torch tensors.")

        # Check if A is a sparse tensor
        if not (A.is_sparse or A.is_sparse_csr):
            raise TypeError("Input A must be a torch sparse COO or CSR tensor.")

    @staticmethod
    def vmap(info, in_dims, A, b, solve_fn):
        """
        Vectorized map function to handle multiple inputs.
        Simple batching by applying the function multiple times.
        """
        A_bdim, b_bdim, _ = in_dims

        # Determine batch size from the batched dimension
        if A_bdim is not None:
            batch_size = A.shape[A_bdim]
        elif b_bdim is not None:
            batch_size = b.shape[b_bdim]
        else:
            # No batching needed
            return TorchMatSolver.apply(A, b, solve_fn), 0

        # Handle different batching scenarios
        if A_bdim is not None and b_bdim is not None:
            # Both A and b are batched
            if A_bdim != 0 or b_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            results = []
            for i in range(batch_size):
                A_i = A[i]  # Extract i-th matrix
                b_i = b[i]  # Extract i-th vector
                x_i = TorchMatSolver.apply(A_i, b_i, solve_fn)
                results.append(x_i)

            return torch.stack(results, dim=0), 0

        elif A_bdim is not None:
            # Only A is batched, b is broadcast
            if A_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            results = []
            for i in range(batch_size):
                A_i = A[i]  # Extract i-th matrix
                x_i = TorchMatSolver.apply(A_i, b, solve_fn)
                results.append(x_i)

            return torch.stack(results, dim=0), 0

        elif b_bdim is not None:
            # Only b is batched, A is broadcast
            if b_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            results = []
            for i in range(batch_size):
                b_i = b[i]  # Extract i-th vector
                x_i = TorchMatSolver.apply(A, b_i, solve_fn)
                results.append(x_i)

            return torch.stack(results, dim=0), 0

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute the gradient of the solution.
        For symmetric sparse matrices A, we have A^T = A.
        """
        A, b, x = ctx.saved_tensors
        grad_A = None
        grad_b = None
        solve_fn = ctx.solve_fn

        needs_grad_A, _, _ = ctx.needs_input_grad

        # Compute the gradient of the solution with respect to b
        # grad_b = A^{-1} * grad_output (since A is symmetric)
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
        grad_b_np = solve_fn(A_np_csc, grad_output.cpu().numpy())

        grad_b = torch.tensor(
            grad_b_np, dtype=grad_output.dtype, device=grad_output.device
        )

        if needs_grad_A:
            # grad_A =  -(A_inv^T * grad_x) * x^T
            grad_A = -sparsified_outer(A, grad_b, x)

        return grad_A, grad_b, None  # None for solve_fn as it is not a tensor input
