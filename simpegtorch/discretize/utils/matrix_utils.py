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


def sdiag(v, dtype=torch.float64, device=None):
    """
    Generate a sparse diagonal matrix from a vector using PyTorch.

    Parameters
    ----------
    v : (n,) array-like or Zero
        The vector defining the diagonal elements of the sparse matrix.
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to store the tensor on.

    Returns
    -------
    torch.sparse.Tensor
        A (n, n) sparse diagonal tensor.
    """
    if isinstance(v, Zero):
        return Zero()

    v = torch.as_tensor(v, dtype=dtype, device=device)
    v = v.view(-1)
    n = v.numel()

    indices = torch.arange(n, device=v.device).repeat(2, 1)
    values = v

    return torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=v.dtype, device=v.device
    )


def speye(n, dtype=torch.float64, device=None):
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
    return sdiag(torch.ones(n, dtype=dtype, device=device), dtype=dtype, device=device)


def get_diag(A):
    """
    Extract the diagonal of a PyTorch sparse matrix.

    Parameters
    ----------
    A : torch.sparse.Tensor
        A 2D sparse tensor in COO format.

    Returns
    -------
    torch.Tensor
        A 1D dense tensor containing the diagonal elements. Zero if no value on the diagonal.
    """
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


def sdinv(M):
    """
    Return the inverse of a sparse diagonal matrix in PyTorch.

    Parameters
    ----------
    M : torch.sparse.Tensor
        A sparse diagonal matrix of shape (n, n)

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
    if not M.is_sparse:
        raise TypeError("Input must be a torch sparse tensor.")

    indices = M._indices()
    values = M._values()

    # Check for non-diagonal entries
    is_diag = indices[0] == indices[1]
    if not torch.all(is_diag):
        raise ValueError("Input sparse matrix has non-zero off-diagonal entries.")

    diag = values[is_diag]
    if (diag == 0).any():
        raise ZeroDivisionError("Cannot invert a diagonal matrix with zero entries.")

    return sdiag(1.0 / diag)


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
    shape = torch.tensor(shape, dtype=torch.long)
    if subs.ndim == 1:
        subs = subs.unsqueeze(0)
    if subs.shape[1] != len(shape):
        raise ValueError("subs must have shape (N, ndim)")

    # Compute Fortran-order strides
    strides = torch.ones_like(shape)
    for i in range(1, len(shape)):
        strides[i] = strides[i - 1] * shape[i - 1]

    return (subs * strides).sum(dim=1)


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

        if tensor.is_sparse:
            return tensor + sign * sdiag(torch.ones(tensor.shape[0]))

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
    return tuple(A_inv[:, i, j] for i in range(3) for j in range(3))


def inverse_property_tensor(mesh, tensor, return_matrix=False, dtype=None, device=None):
    """Construct the inverse of the physical property tensor."""
    propType = TensorType(mesh, tensor)
    if is_scalar(tensor):
        T = torch.tensor(1.0 / tensor)
    elif propType < 3:
        T = 1.0 / mkvc(tensor)
    elif mesh.dim == 2 and tensor.numel() == mesh.n_cells * 3:
        tensor = tensor.view(mesh.n_cells, 3)
        B = inverse_2x2_block_diagonal(
            tensor[:, 0], tensor[:, 2], tensor[:, 2], tensor[:, 1], return_matrix=False
        )
        b11, b12, b21, b22 = B
        T = torch.cat([b11, b22, b12])
    elif mesh.dim == 3 and tensor.numel() == mesh.n_cells * 6:
        tensor = tensor.view(mesh.n_cells, 6)
        B = inverse_3x3_block_diagonal(
            tensor[:, 0],
            tensor[:, 3],
            tensor[:, 5],
            tensor[:, 3],
            tensor[:, 1],
            tensor[:, 4],
            tensor[:, 3],
            tensor[:, 4],
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


def make_property_tensor(mesh, tensor, dtype=None, device=None):
    """Construct the physical property tensor."""
    n_cells = mesh.n_cells
    dim = mesh.dim
    device = mesh.device if device is None else device
    dtype = mesh.dtype if dtype is None else dtype

    if tensor is None:
        tensor = torch.ones(n_cells, device=device, dtype=dtype)
    elif is_scalar(tensor):
        tensor = tensor * torch.ones(n_cells, device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(tensor, device=device, dtype=dtype)

    row_idx = []
    col_idx = []
    values = []

    if tensor.ndim == 1:
        # Isotropic
        for i in range(dim):
            base = i * n_cells
            row_idx.append(torch.arange(n_cells, device=device) + base)
            col_idx.append(torch.arange(n_cells, device=device) + base)
            values.append(tensor)
    elif tensor.ndim == 2 and tensor.shape[1] == dim:
        # Diagonal anisotropic
        for i in range(dim):
            base = i * n_cells
            row_idx.append(torch.arange(n_cells, device=device) + base)
            col_idx.append(torch.arange(n_cells, device=device) + base)
            values.append(tensor[:, i])
    elif dim == 2 and tensor.numel() == n_cells * 3:
        # Full anisotropic 2D: [xx, yy, xy]
        tensor = tensor.reshape(n_cells, 3)
        idx = torch.arange(n_cells, device=device)

        # Sigma_xx
        row_idx.append(idx)
        col_idx.append(idx)
        values.append(tensor[:, 0])

        # Sigma_yy
        row_idx.append(idx + n_cells)
        col_idx.append(idx + n_cells)
        values.append(tensor[:, 1])

        # Sigma_xy (off-diagonal, symmetric)
        row_idx.append(idx)
        col_idx.append(idx + n_cells)
        values.append(tensor[:, 2])

        row_idx.append(idx + n_cells)
        col_idx.append(idx)
        values.append(tensor[:, 2])
    elif dim == 3 and tensor.numel() == n_cells * 6:
        # Full anisotropic 3D: [xx, yy, zz, xy, xz, yz]
        tensor = tensor.reshape(n_cells, 6)
        idx = torch.arange(n_cells, device=device)

        offset = [0, n_cells, 2 * n_cells]

        # Diagonal entries
        row_idx += [idx + o for o in offset]
        col_idx += [idx + o for o in offset]
        values += [tensor[:, i] for i in range(3)]

        # Off-diagonals
        irow, icol, v = zip(
            (idx, idx + n_cells, tensor[:, 3]),  # xy
            (idx, idx + 2 * n_cells, tensor[:, 4]),  # xz
            (idx + n_cells, idx + 2 * n_cells, tensor[:, 5]),  # yz
        )
        for r, c, val in zip(irow, icol, v):
            row_idx += [r, c]
            col_idx += [c, r]
            values += [val, val]
    else:
        raise ValueError(f"Unexpected tensor shape for dim={dim}: {tensor.shape}")

    # Stack and build sparse tensor
    row_idx = torch.cat(row_idx)
    col_idx = torch.cat(col_idx)
    values = torch.cat(values)

    indices = torch.stack([row_idx, col_idx], dim=0)
    size = (dim * n_cells, dim * n_cells)

    Sigma = torch.sparse_coo_tensor(indices, values, size, dtype=dtype, device=device)
    return Sigma.coalesce()
