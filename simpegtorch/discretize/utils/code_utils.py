import numpy as np
import torch

SCALARTYPES = (complex, float, int, np.number)


def is_scalar(f):
    """Determine if the input argument is a scalar.

    Returns True if the input is an integer, float, complex number,
    a NumPy scalar/0-dim array, or a 1-element PyTorch tensor.

    Parameters
    ----------
    f : object
        Any input quantity

    Returns
    -------
    bool
        - True if the input is a scalar or a scalar-like object
        - False otherwise
    """
    if isinstance(f, SCALARTYPES):
        return True
    elif (
        isinstance(f, np.ndarray) and f.size == 1 and isinstance(f.item(), SCALARTYPES)
    ):
        return True
    elif torch.is_tensor(f):
        if f.numel() == 1:
            return True
    return False


def atleast_1d(value, dtype=None, device=None):
    """
    Torch version of numpy.atleast_1d

    Parameters
    ----------
    value : array-like
        Input points (list, array, or tensor).
    dtype : torch.dtype, optional
        Desired dtype for the output tensor.
    device : torch.device, optional
        Desired device for the output tensor.

    Returns
    -------
    tensor: torch.Tensor of shape (len(value), )
    """
    tensor = torch.as_tensor(value, dtype=None, device=None)
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    return tensor


def as_array_n_by_dim(pts, dim, dtype=None, device=None):
    """
    Coerce the input to a 2D PyTorch tensor with shape (n_pts, dim).

    Parameters
    ----------
    pts : array-like
        Input points (list, array, or tensor).
    dim : int
        Expected number of columns.
    dtype : torch.dtype, optional
        Desired dtype for the output tensor.
    device : torch.device, optional
        Desired device for the output tensor.

    Returns
    -------
    pts : torch.Tensor of shape (n_pts, dim)
    """
    pts = torch.as_tensor(pts, dtype=dtype, device=device)

    if dim > 1:
        if pts.ndim == 1:
            # Add a row if needed
            pts = pts.unsqueeze(0)
    elif pts.ndim == 1:
        # Convert to (n, 1)
        pts = pts.unsqueeze(1)

    if pts.ndim != 2 or pts.shape[1] != dim:
        raise ValueError(
            f"pts must be a column vector of shape (n_pts, {dim}) not {tuple(pts.shape)}"
        )

    return pts


def fortran_ravel_index(index, shape):
    stride = 1
    flat_index = 0
    for i, s in zip(index, shape):
        flat_index += i * stride
        stride *= s
    return flat_index
