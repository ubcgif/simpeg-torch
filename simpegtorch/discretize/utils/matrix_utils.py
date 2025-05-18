import torch

def mkvc(x, n_dims=1, dtype=None, device=None):
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


def sdiag(v, dtype=None, device=None):
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

    return torch.sparse_coo_tensor(indices, values, (n, n), dtype=v.dtype, device=v.device)

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

    # Ensure all diagonal values are non-zero
    if torch.any(values == 0):
        raise ZeroDivisionError("Cannot invert a diagonal matrix with zero entries.")

    inv_values = 1.0 / values
    return torch.sparse_coo_tensor(indices, inv_values, M.shape, dtype=M.dtype, device=M.device)

class Zero(object):
    """Carries out arithmetic operations between 0 and arbitrary quantities.

    This class was designed to manage basic arithmetic operations between
    0 and :class:`numpy.ndarray` of any shape. It is a short circuiting evaluation that
    will return the expected values.

    Examples
    --------
    >>> import numpy as np
    >>> from discretize.utils import Zero
    >>> Z = Zero()
    >>> Z
    Zero
    >>> x = np.arange(5)
    >>> x + Z
    array([0, 1, 2, 3, 4])
    >>> Z - x
    array([ 0, -1, -2, -3, -4])
    >>> Z * x
    Zero
    >>> Z @ x
    Zero
    >>> Z[0]
    Zero
    """

    __numpy_ufunc__ = True
    __array_ufunc__ = None

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
        return v == 0

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

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func.__name__ in {"add", "radd"}:
            return args[1] if args[0] is self else args[0]
        if func.__name__ in {"sub"}:
            return -args[1]
        if func.__name__ in {"rsub"}:
            return args[0]
        if func.__name__ in {"mul", "mm", "matmul"}:
            return self
        return NotImplemented

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
        return v + 1 if self._positive else v - 1

    def __radd__(self, v):
        return self + v

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
        return 1.0 / v if self._positive else -1.0 / v

    def __rtruediv__(self, v):
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
        return torch.equal(v, torch.tensor(val, dtype=v.dtype, device=v.device)) if torch.is_tensor(v) else v == val

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

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in {"add", "radd"}:
            return args[1] + 1 if args[0] is self else args[0] + 1
        if func.__name__ in {"sub"}:
            return -args[1] + 1
        if func.__name__ == "rsub":
            return args[0] + (-1)
        if func.__name__ in {"mul", "mm", "matmul"}:
            return args[1] if self._positive else -args[1]
        if func.__name__ in {"div", "truediv", "floordiv"}:
            return (1.0 / args[1]) if self._positive else (-1.0 / args[1])
        if func.__name__ in {"rtruediv", "rfloordiv"}:
            return args[0] if self._positive else -args[0]
        return NotImplemented

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
