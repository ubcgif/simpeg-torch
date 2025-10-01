import torch
from simpegtorch.discretize.utils.code_utils import is_scalar


def mkvc(x, n_dims=1, dtype=torch.float64, device=None):
    """
    Coerce a tensor to the specified dimensionality with column-major flattening.

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input to be flattened in column-major order.
    n_dims : int
        Desired number of dimensions in the output tensor.
    dtype : torch.dtype, optional
        Desired data type.
    device : torch.device, optional
        Desired device.

    Returns
    -------
    torch.Tensor
        A flattened tensor with at least ``n_dims`` dimensions.
    """
    if isinstance(x, Zero) or isinstance(x, Identity):
        return x

    x = torch.as_tensor(x, dtype=dtype, device=device)

    if x.ndim > 1:
        # Simulate Fortran-order flattening by transposing first
        dims = list(range(x.ndim))[::-1]
        x = x.permute(*dims).contiguous().view(-1)
    else:
        x = x.view(-1)

    while x.ndim < n_dims:
        x = x.unsqueeze(-1)

    return x


def sdiag(v, dtype=None, device=None, sparse_type="coo"):
    """
    Generate a sparse diagonal matrix from a vector using PyTorch.

    Parameters
    ----------
    v : (n,) array-like or Zero
        The vector defining the diagonal elements of the sparse matrix.
    dtype : torch.dtype, optional
        Data type of the tensor. If None, preserves input dtype.
    device : torch.device, optional
        Device to store the tensor on. If None, preserves input device.

    sparse_type : {'csr', 'coo'}, optional
        The format of the sparse tensor. Default is 'coo'.

    Returns
    -------
    torch.sparse.Tensor
        A (n, n) sparse diagonal tensor with preserved dtype.
    """
    if isinstance(v, Zero):
        return Zero()

    # Determine dtype and device
    if torch.is_tensor(v):
        dtype = dtype if dtype is not None else v.dtype
        device = device if device is not None else v.device
    else:
        dtype = dtype if dtype is not None else torch.float64
        device = device if device is not None else torch.device('cpu')

    # Preserve gradient flow by checking if tensor already has the right properties
    if (
        torch.is_tensor(v)
        and v.device == device
        and v.dtype == dtype
    ):
        # Don't convert if already correct to preserve gradients
        pass
    else:
        v = torch.as_tensor(v, dtype=dtype, device=device)
    v = v.view(-1)
    n = v.numel()

    indices = torch.arange(n, device=v.device).repeat(2, 1)
    values = v

    tensor = torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=v.dtype, device=v.device
    ).coalesce()

    if sparse_type == "csr":
        return tensor.to_sparse_csr()

    else:
        return tensor


def speye(n, dtype=torch.float64, device=None, sparse_type="coo"):
    """
    Generate a sparse identity matrix using PyTorch.

    Parameters
    ----------
    n : int
        Size of the identity matrix.
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.

    Returns
    -------
    torch.sparse.Tensor
        A (n, n) sparse identity tensor.
    """
    return sdiag(
        torch.ones(n, dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        sparse_type=sparse_type,
    )


def spzeros(n1, n2, dtype=torch.float64, device=None, sparse_type="coo"):
    """Generate sparse matrix of zeros of shape=(n1, n2).

    Parameters
    ----------
    n1 : int
        Number of rows.
    n2 : int
        Number of columns.
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.
    sparse_type : {'coo', 'csr'}, optional
        Output sparse format.

    Returns
    -------
    torch.sparse.Tensor
        A sparse matrix of zeros.
    """
    if device is None:
        device = "cpu"

    # Create empty indices and values for a sparse zero matrix
    indices = torch.empty((2, 0), dtype=torch.long, device=device)
    values = torch.empty(0, dtype=dtype, device=device)

    if sparse_type == "coo":
        return torch.sparse_coo_tensor(
            indices, values, (n1, n2), dtype=dtype, device=device
        )
    elif sparse_type == "csr":
        coo = torch.sparse_coo_tensor(
            indices, values, (n1, n2), dtype=dtype, device=device
        )
        return coo.to_sparse_csr()
    else:
        raise ValueError(f"sparse_type must be 'coo' or 'csr', got {sparse_type}")


def kron(A, B, sparse_type="coo"):
    """

    Method to compute the Kronecker product of torch sparse matrices
    Based on scipy's sparse Kronecker product

    Parameters
    ----------
    A : sparse matrix of the product
    B : sparse matrix of the product
    sparse_type : {'coo', 'csr'}, optional

    Returns
    -------
    kronecker product in a tensor sparse coo matrix format

    """

    # Ensure both matrices are on the same device
    if A.device != B.device:
        # Move B to A's device to maintain device consistency
        B = B.to(device=A.device)

    # # check if A and B are coalesced
    if A.is_sparse & (not A.is_coalesced()):
        A = A.coalesce()

    if B.is_sparse & (not B.is_coalesced()):
        B = B.coalesce()

    # convert to COO format if CSR
    # TODO: natively implement Kron in CSR format
    if A.is_sparse_csr:
        A = A.to_sparse_coo()
    if B.is_sparse_csr:
        B = B.to_sparse_coo()

    # calculate the output dimensions
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    # determine row and columns and extract the data from sparse matrix A
    row = A.indices()[0, :].repeat_interleave(B.values().shape[0])
    col = A.indices()[1, :].repeat_interleave(B.values().shape[0])
    data = A.values().repeat_interleave(B.values().shape[0])

    # take into account sparse matrix B
    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row, col = row.reshape(-1, B.values().shape[0]), col.reshape(
        -1, B.values().shape[0]
    )
    row += B.indices()[0, :]
    col += B.indices()[1, :]
    row, col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, B.values().shape[0]) * B.values()
    data = data.reshape(-1)

    # return output_shape
    output = torch.sparse_coo_tensor(
        torch.vstack([row, col]), data, output_shape
    ).coalesce()

    if sparse_type == "csr":
        return output.to_sparse_csr()
    else:
        return output


def kron3(A, B, C, sparse_type="coo"):
    """
    Compute the Kronecker product of three sparse matrices.

    Parameters
    ----------
    A : torch.sparse.Tensor
        First sparse matrix.
    B : torch.sparse.Tensor
        Second sparse matrix.
    C : torch.sparse.Tensor
        Third sparse matrix.

    Returns
    -------
    torch.sparse.Tensor
        The Kronecker product of A, B, and C.
    """
    return kron(kron(A, B), C, sparse_type=sparse_type)


def ddx(n, dtype=torch.float64, device=None, sparse_type="coo"):
    r"""Create 1D difference (derivative) operator from nodes to centers.

    For n cells, the 1D difference (derivative) operator from nodes to
    centers is sparse, has shape (n, n+1) and takes the form:

    .. math::
        \begin{bmatrix}
        -1 & 1 & & & \\
        & -1 & 1 & & \\
        & & \ddots & \ddots & \\
        & & & -1 & 1
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells (output rows).
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.
    sparse_type : {'coo', 'csr'}, optional
        Output sparse format.
    Returns
    -------
    torch.sparse.Tensor
        1D sparse difference operator of shape (n, n+1).
    """
    # Create row indices: each row i has two entries
    row_indices = torch.arange(n, dtype=torch.long, device=device).repeat(2)

    # Create column indices: for row i, columns are i and i+1
    col_indices = torch.cat(
        [
            torch.arange(
                n, dtype=torch.long, device=device
            ),  # columns 0, 1, 2, ..., n-1
            torch.arange(
                1, n + 1, dtype=torch.long, device=device
            ),  # columns 1, 2, 3, ..., n
        ]
    )

    # Create values: -1 for first column, +1 for second column
    values = torch.cat(
        [
            -torch.ones(n, dtype=dtype, device=device),  # -1 values
            torch.ones(n, dtype=dtype, device=device),  # +1 values
        ]
    )

    indices = torch.vstack([row_indices, col_indices])

    output = torch.sparse_coo_tensor(
        indices, values, (n, n + 1), dtype=dtype, device=device
    ).coalesce()

    if sparse_type == "csr":
        return output.to_sparse_csr()
    else:
        return output


def av(n, dtype=torch.float64, device=None, sparse_type="coo"):
    r"""Create 1D averaging operator from nodes to cell-centers.
    For n cells, the 1D averaging operator from nodes to cell-centers is sparse,
    has shape (n, n+1) and takes the form:
    .. math::
        \begin{bmatrix}
        0.5 & 0.5 & & & \\
        & 0.5 & 0.5 & & \\
        & & \ddots & \ddots & \\
        & & & 0.5 & 0.5
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells (output rows).
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.
    sparse_type : {'coo', 'csr'}, optional
        Output sparse format.
    Returns
    -------
    torch.sparse.Tensor
        1D sparse averaging operator of shape (n, n+1).
    """
    # Create row indices: each row i has two entries
    row_indices = torch.arange(n, dtype=torch.long, device=device).repeat(2)

    # Create column indices: for row i, columns are i and i+1
    col_indices = torch.cat(
        [
            torch.arange(
                n, dtype=torch.long, device=device
            ),  # columns 0, 1, 2, ..., n-1
            torch.arange(
                1, n + 1, dtype=torch.long, device=device
            ),  # columns 1, 2, 3, ..., n
        ]
    )

    # Create values: 0.5 for both columns
    values = 0.5 * torch.ones(2 * n, dtype=dtype, device=device)

    indices = torch.vstack([row_indices, col_indices])

    output = torch.sparse_coo_tensor(
        indices, values, (n, n + 1), dtype=dtype, device=device
    ).coalesce()

    if sparse_type == "csr":
        return output.to_sparse_csr()
    else:
        return output


def av_extrap(n, dtype=torch.float64, device=None, sparse_type="coo"):
    r"""Create 1D averaging operator from cell-centers to nodes.

    For n cells, the 1D averaging operator from cell centers to nodes
    is sparse and has shape (n+1, n). Values at the outmost nodes are
    extrapolated from the nearest cell center value. Thus the operator
    takes the form:

    .. math::
        \begin{bmatrix}
        1 & & & & \\
        1/2 & 1/2 & & & \\
        & 1/2 & 1/2 & & & \\
        & & \ddots & \ddots & \\
        & & & 1/2 & 1/2 \\
        & & & & 1
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.
    sparse_type : {'coo', 'csr'}, optional
        Output sparse format.

    Returns
    -------
    torch.sparse.Tensor
        The 1D averaging operator from cell-centers to nodes with shape (n+1, n).
    """
    if n < 1:
        raise ValueError("Number of cells must be at least 1")

    # Handle special case of n=1
    if n == 1:
        row_indices = torch.tensor([0, 1], dtype=torch.long, device=device)
        col_indices = torch.tensor([0, 0], dtype=torch.long, device=device)
        values = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
    else:
        # Row indices for different parts of the matrix
        # First row (extrapolation): row 0, column 0, value 1
        first_row_idx = torch.tensor([0], dtype=torch.long, device=device)
        first_col_idx = torch.tensor([0], dtype=torch.long, device=device)
        first_values = torch.tensor([1.0], dtype=dtype, device=device)

        # Middle rows (averaging): rows 1 to n-1
        # Each row i has entries at columns i-1 and i with value 0.5
        middle_rows = torch.arange(1, n, dtype=torch.long, device=device)
        middle_row_indices = middle_rows.repeat(2)  # Each row appears twice
        middle_col_indices = torch.cat(
            [
                torch.arange(0, n - 1, dtype=torch.long, device=device),  # columns i-1
                torch.arange(1, n, dtype=torch.long, device=device),  # columns i
            ]
        )
        middle_values = 0.5 * torch.ones(2 * (n - 1), dtype=dtype, device=device)

        # Last row (extrapolation): row n, column n-1, value 1
        last_row_idx = torch.tensor([n], dtype=torch.long, device=device)
        last_col_idx = torch.tensor([n - 1], dtype=torch.long, device=device)
        last_values = torch.tensor([1.0], dtype=dtype, device=device)

        # Combine all parts
        row_indices = torch.cat([first_row_idx, middle_row_indices, last_row_idx])
        col_indices = torch.cat([first_col_idx, middle_col_indices, last_col_idx])
        values = torch.cat([first_values, middle_values, last_values])

    indices = torch.vstack([row_indices, col_indices])

    output = torch.sparse_coo_tensor(
        indices, values, (n + 1, n), dtype=dtype, device=device
    ).coalesce()

    if sparse_type == "csr":
        return output.to_sparse_csr()
    else:
        return output


def get_diag(A):
    """
    Extract the diagonal of a PyTorch sparse matrix.

    Parameters
    ----------
    A : torch.sparse.Tensor
        A 2D sparse tensor, can be either CSR or COO format.

    Returns
    -------
    torch.Tensor
        A 1D dense tensor containing the diagonal elements. Zero if no value on the diagonal.
    """
    if A.is_sparse_csr:
        A = A.to_sparse_coo()

    if not A.is_sparse:
        raise TypeError("Input must be a sparse tensor.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square 2D sparse matrix.")

    n = A.shape[0]
    diag = torch.zeros(n, dtype=A.dtype, device=A.device)

    indices = A._indices()
    values = A._values()

    mask = indices[0] == indices[1]  # positions where row == col
    diag_indices = indices[0, mask]
    diag_values = values[mask]

    diag[diag_indices] = diag_values
    return diag


class SparseDiagInverse(torch.autograd.Function):
    """Custom autograd function for sparse diagonal matrix inversion that preserves gradients."""

    @staticmethod
    def forward(ctx, M):
        """Forward pass: invert diagonal sparse matrix."""
        sparse_type = "coo"
        if M.is_sparse_csr:
            M = M.to_sparse_coo()
            sparse_type = "csr"

        if not M.is_sparse:
            raise TypeError("Input must be a torch sparse tensor.")

        # CRITICAL: Coalesce the matrix to handle duplicate entries correctly
        # This ensures proper behavior on CUDA where matrices may not be auto-coalesced
        if not M.is_coalesced():
            M = M.coalesce()

        indices = M._indices()
        values = M._values()

        # Check for non-diagonal entries
        is_diag = indices[0] == indices[1]
        if not torch.all(is_diag):
            raise ValueError("Input sparse matrix has non-zero off-diagonal entries.")

        diag_values = values[is_diag]
        if (diag_values == 0).any():
            raise ZeroDivisionError(
                "Cannot invert a diagonal matrix with zero entries."
            )

        # Compute inverse values
        inv_values = 1.0 / diag_values

        # Create the inverted sparse matrix
        inv_tensor = torch.sparse_coo_tensor(
            indices[:, is_diag], inv_values, M.shape, dtype=M.dtype, device=M.device
        ).coalesce()

        # Save for backward
        ctx.save_for_backward(inv_values)
        ctx.indices = indices[:, is_diag]
        ctx.shape = M.shape
        ctx.sparse_type = sparse_type

        if sparse_type == "csr":
            return inv_tensor.to_sparse_csr()
        else:
            return inv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient of diagonal matrix inversion."""
        (inv_values,) = ctx.saved_tensors
        indices = ctx.indices
        shape = ctx.shape

        # Convert grad_output to COO if needed
        if grad_output.is_sparse_csr:
            grad_output = grad_output.to_sparse_coo()

        if grad_output.is_sparse:
            grad_output = grad_output.coalesce()
        else:
            # If dense, extract diagonal elements
            grad_diag_values = grad_output.diag()[indices[0]]
            # For diagonal matrix inversion, d/dx(1/x) = -1/x^2
            grad_input_values = -grad_diag_values * (inv_values**2)
            grad_input = torch.sparse_coo_tensor(
                indices,
                grad_input_values,
                shape,
                dtype=grad_input_values.dtype,
                device=grad_input_values.device,
            ).coalesce()
            return grad_input

        # Extract only the diagonal values from grad_output
        grad_indices = grad_output._indices()
        grad_values = grad_output._values()

        # Find diagonal entries in grad_output
        is_grad_diag = grad_indices[0] == grad_indices[1]
        grad_diag_values = grad_values[is_grad_diag]

        # Ensure we have the same number of diagonal elements
        if grad_diag_values.numel() != inv_values.numel():
            # If grad_output has different sparsity pattern, create properly sized gradient
            grad_diag_values = torch.zeros_like(inv_values)
            grad_diag_indices = grad_indices[:, is_grad_diag]
            if grad_diag_indices.numel() > 0:
                # Map the grad diagonal values to the correct positions
                diag_positions = grad_diag_indices[
                    0
                ]  # Row indices of diagonal elements
                input_positions = indices[0]  # Row indices of input diagonal elements
                for i, pos in enumerate(input_positions):
                    mask = diag_positions == pos
                    if mask.any():
                        grad_diag_values[i] = grad_values[is_grad_diag][mask][0]

        # For diagonal matrix inversion, d/dx(1/x) = -1/x^2
        # The gradient flows back as: grad_input = -grad_output * (inv_values)^2
        grad_input_values = -grad_diag_values * (inv_values**2)

        # Create gradient tensor with same sparsity pattern as input
        grad_input = torch.sparse_coo_tensor(
            indices,
            grad_input_values,
            shape,
            dtype=grad_input_values.dtype,
            device=grad_input_values.device,
        ).coalesce()

        return grad_input


def sdinv(M):
    """
    Return the inverse of a sparse diagonal matrix in PyTorch.

    Parameters
    ----------
    M : torch.sparse.Tensor
        A sparse diagonal matrix of shape (n, n)
        Can be either COO or CSR format.

    Returns
    -------
    torch.sparse.Tensor
        Inverse of the input sparse diagonal matrix

    Raises
    ------
    TypeError
        If the input is not a sparse tensor.
    ValueError
        If the matrix contains non-diagonal entries.
    ZeroDivisionError
        If the matrix contains zeros on the diagonal.
    """
    return SparseDiagInverse.apply(M)


def make_boundary_bool(shape, bdir="xyz"):
    r"""Return boundary indices of a tensor grid using PyTorch.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the tensor grid.
    bdir : str
        Which boundaries to include: any combination of 'x', 'y', and 'z'.

    Returns
    -------
    torch.BoolTensor
        A flattened 1D boolean tensor where True indicates a boundary location.
    """

    is_b = torch.zeros(shape, dtype=torch.bool)

    if "x" in bdir:
        is_b[[0, -1]] = True

    if len(shape) > 1 and "y" in bdir:
        is_b[:, [0, -1]] = True

    if len(shape) > 2 and "z" in bdir:
        is_b[:, :, [0, -1]] = True

    # Flatten in Fortran order (column-major)
    return is_b.permute(*reversed(range(is_b.ndim))).reshape(-1)


def ndgrid(*args, vector=True, order="F", dtype=torch.float64, device=None):
    """
    PyTorch equivalent of numpy-based ndgrid for 1D/2D/3D Cartesian products.

    Parameters
    ----------
    *args : list of 1D tensors or individual 1D tensors
        Coordinate vectors (e.g., x, y, z) as separate inputs or a list.
    vector : bool, optional
        If True, returns an (n, ndim) stacked tensor. If False, returns a list of coordinate grids.
    order : {'F', 'C'}
        'F' flattens in column-major (Fortran) order, 'C' in row-major order. Default is 'F'.
    dtype : torch.dtype
        Data type for returned tensors.
    device : torch.device or str
        Device for returned tensors.

    Returns
    -------
    torch.Tensor or list[torch.Tensor]
        A (n, ndim) tensor or list of ndgrid tensors.
    """
    if not isinstance(vector, bool):
        raise TypeError("'vector' must be a boolean")

    if isinstance(args[0], list):
        xin = args[0]
    else:
        xin = args

    xin = [torch.as_tensor(x, dtype=dtype, device=device) for x in xin]

    if len(xin) == 1:
        return xin[0].clone()

    grids = torch.meshgrid(*xin, indexing="ij")  # Fortran-style

    if vector:
        # Flatten each in desired order
        if order == "F":
            # Fortran order: reverse permute before reshape
            flattened = [g.permute(*reversed(range(g.ndim))).reshape(-1) for g in grids]
        elif order == "C":
            # C order: just flatten normally
            flattened = [g.reshape(-1) for g in grids]
        else:
            raise ValueError("order must be 'F' or 'C'")

        return torch.stack(flattened, dim=1)  # shape: (n, ndim)

    return grids


def sub2ind(shape, subs):
    """Torch version of sub2ind using Fortran order."""
    subs = torch.as_tensor(subs, dtype=torch.long)
    shape = torch.as_tensor(shape, dtype=torch.long, device=subs.device)
    if subs.ndim == 1:
        subs = subs.unsqueeze(0)
    if subs.shape[1] != len(shape):
        raise ValueError("subs must have shape (N, ndim)")

    # Compute Fortran-order strides
    strides = torch.ones_like(shape)
    for i in range(1, len(shape)):
        strides[i] = strides[i - 1] * shape[i - 1]

    return (subs * strides).sum(dim=1)


def reshape_fortran(x, shape):
    """Reshape a tensor to a specified shape using Fortran order. Needed as PyTorch does not support this natively."""
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def ind2sub(shape, inds, dtype=torch.int64, device=None):
    r"""Return subscripts of tensor grid elements from indices using Fortran order.

    Parameters
    ----------
    shape : tuple of int
        Shape of the tensor (Fortran order assumed).
    inds : array-like of int
        Flattened indices into the tensor.

    Returns
    -------
    tuple of torch.Tensor
        Subscript tensors for each dimension, matching Fortran-order unraveling.
    """
    shape = torch.tensor(shape, dtype=torch.long)
    inds = torch.as_tensor(inds, dtype=torch.long).flatten()

    strides = [1]
    for s in shape[:-1]:
        strides.append(strides[-1] * s)
    strides = torch.tensor(strides, dtype=torch.long)

    subs = []
    for stride, dim_size in zip(strides, shape):
        subs.append((inds // stride) % dim_size)

    return tuple(subs)


class Zero(object):
    """Carries out arithmetic operations between 0 and arbitrary quantities.

    This class was designed to manage basic arithmetic operations between
    0 and :class:`numpy.ndarray` of any shape. It is a short circuiting evaluation that
    will return the expected values.

    Examples
    --------
    >>> import torch
    >>> from discretize.utils import Zero
    >>> Z = Zero()
    >>> Z
    Zero
    >>> x = torch.arange(5)
    >>> x + Z
    ([0, 1, 2, 3, 4])
    >>> Z - x
    ([ 0, -1, -2, -3, -4])
    >>> Z * x
    Zero
    >>> Z @ x
    Zero
    >>> Z[0]
    Zero
    """

    def __repr__(self):
        """Represent zeros a string."""
        return "Zero"

    def __add__(self, v):
        """Add a value to zero."""
        return v

    def __radd__(self, v):
        """Add zero to a value."""
        return v

    def __iadd__(self, v):
        """Add zero to a value inplace."""
        return v

    def __sub__(self, v):
        """Subtract a value from zero."""
        return -v

    def __rsub__(self, v):
        """Subtract zero from a value."""
        return v

    def __isub__(self, v):
        """Subtract zero from a value inplace."""
        return v

    def __mul__(self, v):
        """Multiply zero by a value."""
        return self

    def __rmul__(self, v):
        """Multiply a value by zero."""
        return self

    def __matmul__(self, v):
        """Multiply zero by a matrix."""
        return self

    def __rmatmul__(self, v):
        """Multiply a matrix by zero."""
        return self

    def __div__(self, v):
        """Divide zero by a value."""
        return self

    def __truediv__(self, v):
        """Divide zero by a value."""
        return self

    def __rdiv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rtruediv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rfloordiv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __pos__(self):
        """Return zero."""
        return self

    def __neg__(self):
        """Negate zero."""
        return self

    def __lt__(self, v):
        """Compare less than zero."""
        return 0 < v

    def __le__(self, v):
        """Compare less than or equal to zero."""
        return 0 <= v

    def __eq__(self, v):
        """Compare equal to zero."""
        return isinstance(v, Zero) or (v == 0)

    def __req__(self, v):
        return self.__eq__(v)

    def __ne__(self, v):
        """Compare not equal to zero."""
        return not (0 == v)

    def __ge__(self, v):
        """Compare greater than or equal to zero."""
        return 0 >= v

    def __gt__(self, v):
        """Compare greater than zero."""
        return 0 > v

    def transpose(self):
        """Return the transpose of the *Zero* class, i.e. itself."""
        return self

    def __getitem__(self, key):
        """Get an element of the *Zero* class, i.e. itself."""
        return self

    @property
    def ndim(self):
        """Return the dimension of *Zero* class, i.e. *None*."""
        return None

    @property
    def shape(self):
        """Return the shape *Zero* class, i.e. *None*."""
        return _inftup(None)

    @property
    def T(self):
        """Return the *Zero* class as an operator."""
        return self


class Identity:
    """Emulates arithmetic behavior of the identity matrix."""

    def __init__(self, positive=True):
        self._positive = positive

    def __repr__(self):
        return "I" if self._positive else "-I"

    def __pos__(self):
        return self

    def __neg__(self):
        return Identity(not self._positive)

    def __add__(self, v):
        return self._apply_to_tensor(v, op="add")

    def __radd__(self, v):
        return self.__add__(v)
        # return v + 1 if self._positive else v - 1

    def __sub__(self, v):
        return self + -v

    def __rsub__(self, v):
        return -self + v

    def __mul__(self, v):
        return v if self._positive else -v

    def __rmul__(self, v):
        return v if self._positive else -v

    def __matmul__(self, v):
        return v if self._positive else -v

    def __rmatmul__(self, v):
        return v if self._positive else -v

    def __truediv__(self, v):
        if torch.is_tensor(v):
            if v.ndim == 1:
                return (1.0 / v) if self._positive else (-1.0 / v)
            elif v.is_sparse:
                return (
                    torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
                    / v.to_dense()
                ).to_sparse()
            else:
                return (
                    torch.ones_like(v) / v
                    if self._positive
                    else -torch.ones_like(v) / v
                )
        return 1.0 / v if self._positive else -1.0 / v

    def __rtruediv__(self, v):
        if torch.is_tensor(v):
            return v if self._positive else -v
        return v if self._positive else -v

    def __floordiv__(self, v):
        return 1 // v if self._positive else -1 // v

    def __rfloordiv__(self, v):
        return v // 1 if self._positive else v // -1

    def __lt__(self, v):
        return 1 < v if self._positive else -1 < v

    def __le__(self, v):
        return 1 <= v if self._positive else -1 <= v

    def __eq__(self, v):
        val = 1 if self._positive else -1
        return (
            torch.equal(v, torch.tensor(val, dtype=v.dtype, device=v.device))
            if torch.is_tensor(v)
            else v == val
        )

    def __req__(self, v):
        return self.__eq__(v)

    def __ne__(self, v):
        return not self.__eq__(v)

    def __ge__(self, v):
        return 1 >= v if self._positive else -1 >= v

    def __gt__(self, v):
        return 1 > v if self._positive else -1 > v

    def to(self, *args, **kwargs):
        return self

    @property
    def ndim(self):
        return None

    @property
    def shape(self):
        return _inftup(None)

    @property
    def T(self):
        return self

    def transpose(self):
        return self

    def _apply_to_tensor(self, tensor, op="add"):
        if not torch.is_tensor(tensor):
            val = 1 if self._positive else -1
            return tensor + val if op == "add" else tensor - val

        sign = 1 if self._positive else -1

        if tensor.is_sparse_csr:
            ## workaround as sparse CSR tensors do not have addition implemented on CPU
            return tensor.to_sparse() + sign * sdiag(
                torch.ones(tensor.shape[0]), sparse_type="coo"
            )

        elif tensor.is_sparse:
            return tensor + sign * sdiag(torch.ones(tensor.shape[0]), sparse_type="coo")

        else:  # dense tensor
            return tensor + sign if op == "add" else tensor - sign


class _inftup(tuple):
    """An infinitely long tuple of a value repeated infinitely."""

    def __init__(self, val=None):
        self._val = val

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _inftup(self._val)
        return self._val

    def __len__(self):
        return 0

    def __repr__(self):
        return f"({self._val}, {self._val}, ...)"


class TensorType:
    r"""Class for determining property tensor type."""

    def __init__(self, mesh, tensor):
        if tensor is None:  # default is ones
            self._tt = -1
            self._tts = "none"

        elif is_scalar(tensor):
            self._tt = 0
            self._tts = "scalar"

        elif isinstance(tensor, torch.Tensor) and tensor.numel() == mesh.n_cells:
            self._tt = 1
            self._tts = "isotropic"

        elif isinstance(tensor, torch.Tensor) and (
            (mesh.dim == 2 and tensor.numel() == mesh.n_cells * 2)
            or (mesh.dim == 3 and tensor.numel() == mesh.n_cells * 3)
        ):
            self._tt = 2
            self._tts = "anisotropic"

        elif isinstance(tensor, torch.Tensor) and (
            (mesh.dim == 2 and tensor.numel() == mesh.n_cells * 3)
            or (mesh.dim == 3 and tensor.numel() == mesh.n_cells * 6)
        ):
            self._tt = 3
            self._tts = "tensor"

        else:
            raise Exception(f"Unexpected shape of tensor: {tensor.shape}")

    def __str__(self):
        return f"TensorType[{self._tt:d}]: {self._tts!s}"

    def __eq__(self, v):
        return self._tt == v

    def __le__(self, v):
        return self._tt <= v

    def __ge__(self, v):
        return self._tt >= v

    def __lt__(self, v):
        return self._tt < v

    def __gt__(self, v):
        return self._tt > v


def inverse_2x2_block_diagonal(a11, a12, a21, a22, return_matrix=True):
    r"""
    Compute the inverse of a 2x2 block diagonal matrix.
    The input matrix is assumed to be of the form:
    .. math::
        \begin{bmatrix}
        a_{11} & a_{12} \\
        a_{21} & a_{22}
        \end{bmatrix}

    Parameters
    ----------
    a11, a12, a21, a22 : torch.Tensor
        The components of the 2x2 block diagonal matrix.
    return_matrix : bool, optional
        If True, return the full matrix. If False, return the components of the inverse matrix.
        Default is True.
    Returns
    -------
    torch.Tensor
        If return_matrix is True, returns the full inverse matrix.
        If False, returns the components of the inverse matrix.

    """
    # Vectorized determinant and inverse formula for 2x2 matrices
    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a21 = mkvc(a21)
    a22 = mkvc(a22)

    det = a11 * a22 - a12 * a21
    inv_det = 1.0 / det

    b11 = a22 * inv_det
    b22 = a11 * inv_det
    b12 = -a12 * inv_det
    b21 = -a21 * inv_det

    if return_matrix:
        return torch.vstack(
            (
                torch.hstack((sdiag(b11), sdiag(b12))),
                torch.hstack((sdiag(b21), sdiag(b22))),
            )
        )

    return b11, b12, b21, b22


def inverse_3x3_block_diagonal(
    a11, a12, a13, a21, a22, a23, a31, a32, a33, return_matrix=True
):
    r"""Compute the inverse of a 3x3 block diagonal matrix.
    The input matrix is assumed to be of the form:
    .. math::
        \begin{bmatrix}
        a_{11} & a_{12} & a_{13} \\
        a_{21} & a_{22} & a_{23} \\
        a_{31} & a_{32} & a_{33}
        \end{bmatrix}

    Parameters
    ----------
    a11, a12, a13, a21, a22, a23, a31, a32, a33 : torch.Tensor
        The components of the 3x3 block diagonal matrix.
    return_matrix : bool, optional
        If True, return the full matrix. If False, return the components of the inverse matrix.
        Default is True.
    Returns
    -------
    torch.Tensor
        If return_matrix is True, returns the full inverse matrix.
        If False, returns the components of the inverse matrix.
    """
    # Vectorized determinant and inverse formula for 3x3 matrices

    # Construct batched 3x3 matrices: shape (n, 3, 3)
    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a13 = mkvc(a13)
    a21 = mkvc(a21)
    a22 = mkvc(a22)
    a23 = mkvc(a23)
    a31 = mkvc(a31)
    a32 = mkvc(a32)
    a33 = mkvc(a33)

    A = torch.stack(
        [
            torch.stack([a11, a12, a13], dim=-1),
            torch.stack([a21, a22, a23], dim=-1),
            torch.stack([a31, a32, a33], dim=-1),
        ],
        dim=-2,
    )  # shape: (n, 3, 3)

    A_inv = torch.linalg.inv(A)

    b11 = A_inv[:, 0, 0]
    b12 = A_inv[:, 0, 1]
    b13 = A_inv[:, 0, 2]
    b21 = A_inv[:, 1, 0]
    b22 = A_inv[:, 1, 1]
    b23 = A_inv[:, 1, 2]
    b31 = A_inv[:, 2, 0]
    b32 = A_inv[:, 2, 1]
    b33 = A_inv[:, 2, 2]

    if return_matrix:
        return torch.vstack(
            (
                torch.hstack((sdiag(b11), sdiag(b12), sdiag(b13))),
                torch.hstack((sdiag(b21), sdiag(b22), sdiag(b23))),
                torch.hstack((sdiag(b31), sdiag(b32), sdiag(b33))),
            )
        )

    # Return components in the same flattened structure
    return b11, b12, b13, b21, b22, b23, b31, b32, b33


def torch_blockdiag(matrices, sparse_type="coo"):
    """
    Create a block diagonal sparse matrix from a sequence of sparse matrices using PyTorch.

    This function is equivalent to scipy.sparse.block_diag but uses PyTorch sparse tensors.

    Parameters
    ----------
    matrices : sequence of torch.sparse.Tensor
        A sequence of sparse tensors to be arranged in block diagonal form.
    sparse_type : {'coo', 'csr'}, optional
        The format of the output sparse tensor. Default is 'coo'.

    Returns
    -------
    torch.sparse.Tensor
        A block diagonal sparse tensor where the input matrices are placed along the diagonal.

    Examples
    --------
    >>> import torch
    >>> from simpegtorch.discretize.utils.matrix_utils import torch_blockdiag, sdiag
    >>>
    >>> # Create some sample sparse matrices
    >>> A = sdiag(torch.tensor([1., 2., 3.]))
    >>> B = sdiag(torch.tensor([4., 5.]))
    >>> C = sdiag(torch.tensor([6.]))
    >>>
    >>> # Create block diagonal matrix
    >>> result = torch_blockdiag([A, B, C])
    >>> print(result.shape)
    torch.Size([6, 6])
    """
    if not matrices:
        raise ValueError("Input sequence cannot be empty")

    # Convert to list if it's a tuple
    if isinstance(matrices, tuple):
        matrices = list(matrices)

    # Ensure all matrices are sparse COO tensors and get their properties
    matrix_info = []
    total_rows = 0
    total_cols = 0

    for mat in matrices:
        if not (mat.is_sparse or mat.is_sparse_csr):
            raise TypeError("All input matrices must be sparse tensors")

        # Convert to COO if needed
        if mat.is_sparse_csr:
            mat = mat.to_sparse_coo()
        elif not mat.layout == torch.sparse_coo:
            raise TypeError("Unsupported sparse format")

        mat = mat.coalesce()
        matrix_info.append(mat)
        total_rows += mat.shape[0]
        total_cols += mat.shape[1]

    # Build the block diagonal matrix
    if len(matrix_info) == 1:
        # Special case: single matrix
        result = matrix_info[0]
    else:
        # Collect all indices and values
        all_indices = []
        all_values = []

        row_offset = 0
        col_offset = 0

        for mat in matrix_info:
            if mat._nnz() > 0:  # Only process if matrix has non-zero elements
                indices = mat._indices()
                values = mat._values()

                # Offset the indices
                offset_indices = indices.clone()
                offset_indices[0] += row_offset  # Row indices
                offset_indices[1] += col_offset  # Column indices

                all_indices.append(offset_indices)
                all_values.append(values)

            row_offset += mat.shape[0]
            col_offset += mat.shape[1]

        if all_indices:
            # Concatenate all indices and values
            combined_indices = torch.cat(all_indices, dim=1)
            combined_values = torch.cat(all_values, dim=0)

            # Create the block diagonal matrix
            result = torch.sparse_coo_tensor(
                combined_indices,
                combined_values,
                (total_rows, total_cols),
                dtype=combined_values.dtype,
                device=combined_values.device,
            ).coalesce()
        else:
            # All matrices are empty
            device = matrix_info[0].device
            dtype = matrix_info[0].dtype
            result = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=dtype, device=device),
                (total_rows, total_cols),
                dtype=dtype,
                device=device,
            )

    if sparse_type == "csr":
        return result.to_sparse_csr()
    else:
        return result


def inverse_property_tensor(
    mesh, tensor, return_matrix=False, dtype=torch.float64, device=None
):
    """Construct the inverse of the physical property tensor."""
    propType = TensorType(mesh, tensor)

    if is_scalar(tensor):
        T = torch.tensor(1.0 / tensor)
    elif propType < 3:
        T = 1.0 / mkvc(tensor)
    elif mesh.dim == 2 and tensor.numel() == mesh.n_cells * 3:
        tensor = reshape_fortran(tensor, (mesh.n_cells, 3))
        B = inverse_2x2_block_diagonal(
            tensor[:, 0], tensor[:, 2], tensor[:, 2], tensor[:, 1], return_matrix=False
        )
        b11, b12, b21, b22 = B
        T = torch.cat([b11, b22, b12], dim=0)
    elif mesh.dim == 3 and tensor.numel() == mesh.n_cells * 6:
        tensor = reshape_fortran(tensor, (mesh.n_cells, 6))
        B = inverse_3x3_block_diagonal(
            tensor[:, 0],
            tensor[:, 3],
            tensor[:, 4],
            tensor[:, 3],
            tensor[:, 1],
            tensor[:, 5],
            tensor[:, 4],
            tensor[:, 5],
            tensor[:, 2],
            return_matrix=False,
        )
        b11, b12, b13, b21, b22, b23, b31, b32, b33 = B
        T = torch.cat([b11, b22, b33, b12, b13, b23], dim=0)
    else:
        raise Exception("Unexpected shape of tensor")

    if return_matrix:
        return make_property_tensor(mesh, T, dtype, device)

    return T


def get_subarray(A, ind):
    """Extract a subarray from a PyTorch tensor.

    For a torch.Tensor, the function get_subarray extracts a subset of
    the tensor. The portion of the original tensor being extracted is defined
    by providing the indices along each axis.

    Parameters
    ----------
    A : torch.Tensor
        The original tensor. Must be 1, 2 or 3 dimensions.
    ind : list of array-like
        A list of arrays containing the indices being extracted along each
        dimension. The length of the list must equal the dimensions of the input tensor.

    Returns
    -------
    torch.Tensor
        The subtensor extracted from the original tensor
    """
    if not isinstance(ind, list):
        raise TypeError("ind must be a list of vectors")
    if len(A.shape) != len(ind):
        raise ValueError("ind must have the same length as the dimension of A")

    # Convert indices to tensors
    ind = [torch.as_tensor(i, dtype=torch.long) for i in ind]

    if len(A.shape) == 2:
        return A[ind[0], :][:, ind[1]]
    elif len(A.shape) == 3:
        return A[ind[0], :, :][:, ind[1], :][:, :, ind[2]]
    else:
        raise Exception("get_subarray does not support dimension asked.")


def make_property_tensor(
    mesh, tensor, dtype=None, device=None, sparse_type="coo"
):
    """Construct the physical property tensor.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object
    tensor : torch.Tensor, float, or None
        The property tensor values
    dtype : torch.dtype, optional
        Target dtype. If None, preserves input dtype or uses mesh.dtype
    device : torch.device, optional
        Target device. If None, uses mesh.device
    sparse_type : str
        Type of sparse tensor to return ('coo' or 'csr')

    Returns
    -------
    torch.Tensor
        Sparse property tensor with preserved dtype
    """
    n_cells = mesh.n_cells
    dim = mesh.dim
    device = mesh.device if device is None else device

    # Determine dtype: preserve input dtype if not specified
    if tensor is None:
        dtype = mesh.dtype if dtype is None else dtype
        tensor = torch.ones(n_cells, device=device, dtype=dtype)
    elif is_scalar(tensor):
        dtype = mesh.dtype if dtype is None else dtype
        tensor = tensor * torch.ones(n_cells, device=device, dtype=dtype)
    else:
        # If dtype not specified, preserve the input tensor's dtype
        if dtype is None:
            dtype = tensor.dtype if torch.is_tensor(tensor) else mesh.dtype

        # Preserve gradient flow by checking if tensor already has the right properties
        if (
            torch.is_tensor(tensor)
            and tensor.device == device
            and tensor.dtype == dtype
        ):
            # Don't convert if already correct to preserve gradients
            pass
        else:
            tensor = torch.as_tensor(tensor, device=device, dtype=dtype)

    propType = TensorType(mesh, tensor)

    if propType == 1:  # Isotropic
        Sigma = kron(speye(dim, device=device, dtype=dtype), sdiag(mkvc(tensor)))

    elif propType == 2:  # Anisotropic
        Sigma = sdiag(mkvc(tensor))

    elif dim == 2 and tensor.numel() == n_cells * 3:  # Fully anisotropic, 2D
        tensor = reshape_fortran(tensor, (n_cells, 3))
        row1 = torch.hstack((sdiag(tensor[:, 0]), sdiag(tensor[:, 2])))
        row2 = torch.hstack((sdiag(tensor[:, 2]), sdiag(tensor[:, 1])))
        Sigma = torch.vstack((row1, row2))

    elif dim == 3 and tensor.numel() == n_cells * 6:  # Fully anisotropic, 3D
        tensor = reshape_fortran(tensor, (n_cells, 6))
        row1 = torch.hstack(
            (sdiag(tensor[:, 0]), sdiag(tensor[:, 3]), sdiag(tensor[:, 4]))
        )
        row2 = torch.hstack(
            (sdiag(tensor[:, 3]), sdiag(tensor[:, 1]), sdiag(tensor[:, 5]))
        )
        row3 = torch.hstack(
            (sdiag(tensor[:, 4]), sdiag(tensor[:, 5]), sdiag(tensor[:, 2]))
        )
        Sigma = torch.vstack((row1, row2, row3))

    else:
        raise Exception("Unexpected shape of tensor")

    if sparse_type == "csr":
        return Sigma.to_sparse_csr()
    else:
        return Sigma


def cross2d(x, y):
    """Compute the cross product of two vectors.

    This function will calculate the cross product as if
    the third component of each of these vectors was zero.

    The returned direction is perpendicular to both inputs,
    making it be solely in the third dimension.

    Parameters
    ----------
    x, y : array_like
        The vectors for the cross product.

    Returns
    -------
    x_cross_y : numpy.ndarray
        The cross product of x and y.
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
