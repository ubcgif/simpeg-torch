import torch
from torch.autograd import Function
import scipy.sparse as sp

## This is a wrapper for pymatsolver to use with torch tensors.
# It will also compute the gradient of the solution for backpropoagation


class TorchMatSolver(Function):
    @staticmethod
    def forward(A, b, solve_fn):
        """
        Forward pass to solve Ax = b using pymatsolver.
        """
        ## Check if A is a sparse matrix
        if isinstance(A, torch.Tensor) and A.is_sparse:
            A_coalesced = A.coalesce()
            A_np_coo = sp.coo_matrix(
                (
                    A_coalesced.values().cpu().numpy(),
                    (
                        A_coalesced.indices()[0].cpu().numpy(),
                        A_coalesced.indices()[1].cpu().numpy(),
                    ),
                ),
                shape=A.shape,
            )
            A_np = sp.csc_matrix(A_np_coo)
            x = solve_fn(A_np, b.cpu().numpy())

        elif isinstance(A, torch.Tensor) and A.is_sparse_csr:
            A_np = sp.csr_matrix(
                (
                    A.values().cpu().numpy(),
                    A.col_indices().cpu().numpy(),
                    A.crow_indices().cpu().numpy(),
                ),
                shape=A.shape,
            )
            A_np_csc = sp.csc_matrix(A_np)
            x = solve_fn(A_np_csc, b.cpu().numpy())

        else:
            raise TypeError("Input A must be a torch sparse COO or CSR tensor.")

        return torch.tensor(x, dtype=b.dtype, device=b.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for the backward pass.
        """
        A, b, solve_fn = inputs
        ctx.save_for_backward(A, b)
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
        A, b = ctx.saved_tensors
        grad_A = None
        grad_b = None
        solve_fn = ctx.solve_fn

        if ctx.needs_input_grad[1]:
            # Compute the gradient of the solution with respect to b
            # grad_b = A^{-1} * grad_output (since A is symmetric)
            if A.is_sparse:
                A_coalesced = A.coalesce()
                A_np_coo = sp.coo_matrix(
                    (
                        A_coalesced.values().cpu().numpy(),
                        (
                            A_coalesced.indices()[0].cpu().numpy(),
                            A_coalesced.indices()[1].cpu().numpy(),
                        ),
                    ),
                    shape=A.shape,
                )
                A_np = sp.csc_matrix(A_np_coo)
                grad_b_np = solve_fn(A_np, grad_output.cpu().numpy())
            elif A.is_sparse_csr:
                A_np = sp.csr_matrix(
                    (
                        A.values().cpu().numpy(),
                        A.col_indices().cpu().numpy(),
                        A.crow_indices().cpu().numpy(),
                    ),
                    shape=A.shape,
                )
                A_np_csc = sp.csc_matrix(A_np)
                grad_b_np = solve_fn(A_np_csc, grad_output.cpu().numpy())
            else:
                raise TypeError("Input A must be a torch sparse tensor.")

            grad_b = torch.tensor(
                grad_b_np, dtype=grad_output.dtype, device=grad_output.device
            )

        if ctx.needs_input_grad[0]:
            # Compute the gradient of the solution with respect to A
            # For symmetric A: dL/dA = -x * (dL/dx)^T where x = A^{-1}b
            if grad_b is None:
                # Need to solve for x if we didn't compute grad_b
                if A.is_sparse:
                    A_coalesced = A.coalesce()
                    A_np_coo = sp.coo_matrix(
                        (
                            A_coalesced.values().cpu().numpy(),
                            (
                                A_coalesced.indices()[0].cpu().numpy(),
                                A_coalesced.indices()[1].cpu().numpy(),
                            ),
                        ),
                        shape=A.shape,
                    )
                    A_np = sp.csc_matrix(A_np_coo)
                    x_np = solve_fn(A_np, b.cpu().numpy())
                elif A.is_sparse_csr:
                    A_np = sp.csr_matrix(
                        (
                            A.values().cpu().numpy(),
                            A.col_indices().cpu().numpy(),
                            A.crow_indices().cpu().numpy(),
                        ),
                        shape=A.shape,
                    )
                    A_np_csc = sp.csc_matrix(A_np)
                    x_np = solve_fn(A_np_csc, b.cpu().numpy())
                x = torch.tensor(x_np, device=A.device)
            else:
                x = grad_b

            # Compute sparse gradient directly without dense outer product
            # For symmetric A, grad_A[i,j] = -(x[i] * grad_output[j] + x[j] * grad_output[i]) / 2
            # We only need to compute gradients at the same sparse locations as A

            if A.is_sparse:
                A_coalesced = A.coalesce()
                indices = A_coalesced.indices()
                i_idx = indices[0]
                j_idx = indices[1]

                # Compute gradient values only at sparse locations
                grad_values = (
                    -(x[i_idx] * grad_output[j_idx] + x[j_idx] * grad_output[i_idx]) / 2
                )

                # Create sparse gradient tensor
                grad_A = torch.sparse_coo_tensor(
                    indices, grad_values, A.shape, device=A.device, dtype=A.dtype
                )

            elif A.is_sparse_csr:
                # For CSR format, we need to extract indices differently
                crow_indices = A.crow_indices()
                col_indices = A.col_indices()

                # Convert CSR to COO indices for easier computation
                row_indices = []
                for i in range(len(crow_indices) - 1):
                    row_indices.extend([i] * (crow_indices[i + 1] - crow_indices[i]))
                row_indices = torch.tensor(row_indices, device=A.device)

                # Compute gradient values only at sparse locations
                grad_values = (
                    -(
                        x[row_indices] * grad_output[col_indices]
                        + x[col_indices] * grad_output[row_indices]
                    )
                    / 2
                )

                # Create sparse CSR gradient tensor
                grad_A = torch.sparse_csr_tensor(
                    crow_indices,
                    col_indices,
                    grad_values,
                    A.shape,
                    device=A.device,
                    dtype=A.dtype,
                )
            else:
                raise TypeError("Input A must be a torch sparse tensor.")

        return grad_A, grad_b, None  # None for solve_fn as it is not a tensor input
