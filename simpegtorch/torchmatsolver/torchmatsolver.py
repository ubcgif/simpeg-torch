import torch
from torch.autograd import Function
import scipy.sparse as sp

## This is a wrapper for pymatsolver to use with torch tensors.
# It will also compute the gradient of the solution for backpropoagation


class TorchMatSolver(Function):
    @staticmethod
    def forward(ctx, A, b, solve_fn):
        """
        Forward pass to solve Ax = b using pymatsolver.
        """
        ctx.save_for_backward(A, b)
        ctx.solve_fn = solve_fn

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
