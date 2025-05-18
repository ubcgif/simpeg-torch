import torch


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
    Return the inverse of a sparse diagonal matrix.

    Parameters
    ----------
    M : torch.sparse.Tensor
        A (n, n) sparse diagonal matrix created by `sdiag`.

    Returns
    -------
    torch.sparse.Tensor
        A (n, n) sparse diagonal matrix whose diagonal elements are the reciprocal of M's.
    """
    if not M.is_sparse:
        raise ValueError("Input matrix M must be a sparse tensor.")

    indices = M._indices()
    values = M._values()

    row, col = indices
    if any(row != col):
        raise ValueError("Cannot invert a sparse matrix with off diagonal entries.")

    # Ensure all diagonal values are non-zero
    if torch.any(values == 0):
        raise ZeroDivisionError("Cannot invert a diagonal matrix with zero entries.")

    inv_values = 1.0 / values
    return torch.sparse_coo_tensor(
        indices, inv_values, M.shape, dtype=M.dtype, device=M.device
    )


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
