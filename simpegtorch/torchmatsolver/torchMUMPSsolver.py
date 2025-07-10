import torch
from torch.autograd import Function
from .utils import torch_tensor_to_sp, sparsified_outer

try:
    from pymatsolver import Mumps

    _mumps_available = True
except ImportError:
    _mumps_available = False

## This is a wrapper for pymatsolver MUMPS solver to use with torch tensors.
# It will also compute the gradient of the solution for backpropoagation


class TorchMUMPSsolver(Function):
    @staticmethod
    def forward(A, b):
        """
        Forward pass to solve Ax = b using pymatsolver MUMPS solver.
        """
        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

        # Create MUMPS solver instance
        mumps_solver = Mumps(A_np_csc)

        # Solve the system
        x = mumps_solver.solve(b.cpu().numpy())

        return torch.tensor(x, dtype=b.dtype, device=b.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for the backward pass.
        """
        A, b = inputs
        x = output
        ctx.save_for_backward(A, b, x)

        # Ensure A and b are tensors
        if not isinstance(A, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Inputs A and b must be torch tensors.")

        # Check if A is a sparse tensor
        if not (A.is_sparse or A.is_sparse_csr):
            raise TypeError("Input A must be a torch sparse COO or CSR tensor.")

    @staticmethod
    def vmap(_info, in_dims, A, b):
        """
        Vectorized map function to handle multiple inputs.
        Optimized batching using single factorization for multiple RHS vectors.
        """
        A_bdim, b_bdim = in_dims

        # Determine batch size from the batched dimension
        if A_bdim is not None:
            batch_size = A.shape[A_bdim]
        elif b_bdim is not None:
            batch_size = b.shape[b_bdim]
        else:
            # No batching needed
            return TorchMUMPSsolver.apply(A, b), 0

        # Handle different batching scenarios
        if A_bdim is not None and b_bdim is not None:
            # Both A and b are batched
            if A_bdim != 0 or b_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            results = []
            for i in range(batch_size):
                A_i = A[i]  # Extract i-th matrix
                b_i = b[i]  # Extract i-th vector
                x_i = TorchMUMPSsolver.apply(A_i, b_i)
                results.append(x_i)

            return torch.stack(results, dim=0), 0

        elif A_bdim is not None:
            # Only A is batched, b is broadcast
            if A_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            results = []
            for i in range(batch_size):
                A_i = A[i]  # Extract i-th matrix
                x_i = TorchMUMPSsolver.apply(A_i, b)
                results.append(x_i)

            return torch.stack(results, dim=0), 0

        elif b_bdim is not None:
            # Only b is batched, A is broadcast - OPTIMIZED FOR MULTIPLE RHS
            if b_bdim != 0:
                raise NotImplementedError("Only batching on dimension 0 is supported")

            if not _mumps_available:
                raise ImportError(
                    "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
                )

            # Convert A to sparse CSC format once
            A_np_csc = torch_tensor_to_sp(A, sp_type="csc")

            # Create MUMPS solver instance and factor once
            mumps_solver = Mumps(A_np_csc)

            # Convert all RHS vectors to numpy and solve in batch
            b_np = b.detach().cpu().numpy()  # Shape: (batch_size, n)

            # Solve for all RHS vectors at once using MUMPS multiple RHS capability
            x_np = mumps_solver.solve(b_np.T)  # Transpose for column-major multiple RHS

            # Convert back to torch tensor
            if x_np.ndim == 1:
                # Single RHS case
                x = torch.tensor(x_np, dtype=b.dtype, device=b.device)
                return x.unsqueeze(0), 0
            else:
                # Multiple RHS case
                x = torch.tensor(
                    x_np.T, dtype=b.dtype, device=b.device, requires_grad=True
                )  # Transpose back
                return x, 0

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute the gradient of the solution.
        For symmetric sparse matrices A, we have A^T = A.
        """
        A, _b, x = ctx.saved_tensors
        grad_A = None
        grad_b = None

        needs_grad_A, _ = ctx.needs_input_grad

        if not _mumps_available:
            raise ImportError(
                "pymatsolver with MUMPS support is required. Install with: conda install -c conda-forge python-mumps"
            )

        # Compute the gradient of the solution with respect to b
        # grad_b = A^{-1} * grad_output (since A is symmetric)
        A_np_csc = torch_tensor_to_sp(A, sp_type="csc")
        mumps_solver = Mumps(A_np_csc)
        grad_b_np = mumps_solver.solve(grad_output.cpu().numpy())

        grad_b = torch.tensor(
            grad_b_np, dtype=grad_output.dtype, device=grad_output.device
        )

        if needs_grad_A:
            # grad_A =  -(A_inv^T * grad_x) * x^T
            grad_A = -sparsified_outer(A, grad_b, x)

        return grad_A, grad_b
