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
    elif isinstance(f, np.ndarray) and f.size == 1 and isinstance(f.item(), SCALARTYPES):
        return True
    elif torch.is_tensor(f):
        if f.numel() == 1:
            return True
    return False

def atleast_1d(value):
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    return tensor
