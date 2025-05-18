import torch
from .utils import is_scalar
from .base import BaseRectangularMesh

class TensorMesh(BaseRectangularMesh):
    def __init__(self, h, origin=None):
        try:
            h = list(h)  # ensure value is a list (and make a copy)
        except TypeError:
            raise TypeError("h must be an iterable object, not {}".format(type(h)))
        if len(h) == 0 or len(h) > 3:
            raise ValueError("h must be of dimension 1, 2, or 3 not {}".format(len(h)))
        # expand value
        for i, h_i in enumerate(h):
            if is_scalar(h_i) and not isinstance(h_i, torch.tensor):
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * torch.ones(int(h_i)) / int(h_i)
            elif isinstance(h_i, (list, tuple)):
                h_i = unpack_widths(h_i)
            if not isinstance(h_i, torch.Tensor):
                try:
                    h_i = torch.tensor(h_i)
                except:
                    raise TypeError("h[{0:d}] cannot be cast to a torch tensor.".format(i))
            if len(h_i.shape) != 1:
                raise ValueError("h[{0:d}] must be a 1D array.".format(i))
            h[i] = h_i[:]  # make a copy.
        self._h = tuple(h)


        shape_cells = tuple([len(h_i) for h_i in h])
        super().__init__(shape_cells=shape_cells)  # do not pass origin here
        if origin is not None:
            self.origin = origin
