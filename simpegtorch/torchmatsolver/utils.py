import torch
import scipy.sparse as sp


def torch_tensor_to_sp(tensor: torch.tensor, sp_type="csr"):
    ## SP type must be one of csr csc or coo
    if sp_type not in {"csr", "coo", "csc"}:
        raise NotImplementedError

    ## Check if A is a sparse matrix
    if tensor.is_sparse:
        tensor_coalesced = tensor.coalesce()
        tensor_np = sp.coo_matrix(
            (
                tensor_coalesced.values().cpu().numpy(),
                (
                    tensor_coalesced.indices()[0].cpu().numpy(),
                    tensor_coalesced.indices()[1].cpu().numpy(),
                ),
            ),
            shape=tensor_coalesced.shape,
        )

    elif tensor.is_sparse_csr:
        tensor_np = sp.csr_matrix(
            (
                tensor.values().cpu().numpy(),
                tensor.col_indices().cpu().numpy(),
                tensor.crow_indices().cpu().numpy(),
            ),
            shape=tensor.shape,
        )

    if sp_type == "csc":
        return sp.csc_matrix(tensor_np)
    elif sp_type == "coo":
        return sp.coo_matrix(tensor_np)

    return sp.csr_matrix(tensor_np)


def sparsified_outer(template_A, x, y):
    """
    calculates the outerproduct of vectors x and y at only non-zero indices of A, returns a sparse matrix of the same shape as A,
    the outer product of x and y must be the same shape as A
    """

    if template_A.is_sparse_csr:
        raise NotImplementedError(
            "sparsified outer product not implemented for CSR matrices"
        )

    # Validate input shapes
    if x.shape[0] != template_A.shape[0]:
        raise ValueError(
            f"x length {x.shape[0]} doesn't match A rows {template_A.shape[0]}"
        )
    if y.shape[0] != template_A.shape[1]:
        raise ValueError(
            f"y length {y.shape[0]} doesn't match A cols {template_A.shape[1]}"
        )

    # Ensure x and y are 1D
    x = x.flatten()
    y = y.flatten()

    if template_A.is_sparse:  # COO format
        # Get the indices where A is non-zero
        A_coalesced = template_A.coalesce()
        indices = A_coalesced.indices()  # Shape: (2, nnz)

        # Extract row and column indices
        row_indices = indices[0]  # i indices
        col_indices = indices[1]  # j indices

        # Compute outer product values only at sparse positions
        # outer[i,j] = x[i] * y[j]
        values = x[row_indices] * y[col_indices]

        # Create sparse tensor with same structure
        result = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=template_A.shape,
            dtype=torch.result_type(x, y),  # Use appropriate dtype
            device=template_A.device,
        )

    return result
