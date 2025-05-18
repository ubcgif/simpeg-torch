import torch

from .code_utils import is_scalar


def unpack_widths(value, dtype=torch.float64, device=None):
    """Unpack a condensed representation of cell widths or time steps.

    For a list of numbers, if the same value is repeat or expanded by a constant
    factor, it may be represented in a condensed form using list of floats
    and/or tuples. **unpack_widths** takes a list of floats and/or tuples in
    condensed form, e.g.:

        [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

    and expands the representation to a list containing all widths in order. That is:

        [ w1, w2, w3, ..., wn ]

    Parameters
    ----------
    value : list of float and/or tuple
        The list of floats and/or tuples that are to be unpacked

    Returns
    -------
    numpy.ndarray
        The unpacked list with all widths in order

    Examples
    --------
    Time stepping for time-domain codes can be represented in condensed form, e.g.:

    >>> from discretize.utils import unpack_widths
    >>> dt = [ (1e-5, 10), (1e-4, 4), 1e-3 ]

    The above means to take 10 steps at a step width of 1e-5 s and then
    4 more at 1e-4 s, and then one step of 1e-3 s. When unpacked, the output is
    of length 15 and is given by:

    >>> unpack_widths(dt)
    array([1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05,
           1.e-05, 1.e-05, 1.e-04, 1.e-04, 1.e-04, 1.e-04, 1.e-03])

    Each axis of a tensor mesh can also be defined as a condensed list of floats
    and/or tuples. When a third number is defined in any tuple, the width value
    is successively expanded by that factor, e.g.:

    >>> dt = [ 6., 8., (10.0, 3), (8.0, 4, 2.) ]
    >>> unpack_widths(dt)
    array([  6.,   8.,  10.,  10.,  10.,  16.,  32.,  64., 128.])
    """
    if type(value) is not list:
        raise Exception("unpack_widths must be a list of scalars and tuples.")

    proposed = []
    for v in value:
        if is_scalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])] * int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = (
                (torch.ones(num, dtype=dtype, device=device) * torch.abs(factor))
                ** (torch.arange(num, dtype=dtype, device=device) + 1)
            ) * start
            if factor < 0:
                pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception(
                "unpack_widths must contain only scalars and len(2) or len(3) tuples."
            )

    return torch.tensor(proposed, dtype=dtype, device=device)
