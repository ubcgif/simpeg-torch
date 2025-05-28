import itertools
import torch

from .utils import fortran_ravel_index


class TensorCell:
    """
    Representation of a cell in a TensorMesh.

    Parameters
    ----------
    h : (dim) torch.tensor
        Array with the cell widths along each direction. For a 2D mesh, it
        must have two elements (``hx``, ``hy``). For a 3D mesh it must have
        three elements (``hx``, ``hy``, ``hz``).
    origin : (dim) torch.tensor
        Array with the coordinates of the origin of the cell, i.e. the
        bottom-left-frontmost corner.
    index_unraveled : (dim) tuple
        Array with the unraveled indices of the cell in its parent mesh.
    mesh_shape : (dim) tuple
        Shape of the parent mesh.

    Examples
    --------
    Define a simple :class:`discretize.TensorMesh`.

    >>> from discretize import TensorMesh
    >>> mesh = TensorMesh([5, 8, 10])

    We can obtain a particular cell in the mesh by its index:

    >>> cell = mesh[3]
    >>> cell
    TensorCell(h=[0.2   0.125 0.1  ], origin=[0.6 0.  0. ], index=3, mesh_shape=(5, 8, 10))

    And then obtain information about it, like its
    :attr:`discretize.tensor_cell.TensorCell.origin`:

    >>> cell.origin
    array([0.6, 0. , 0. ])

    Or its
    :attr:`discretize.tensor_cell.TensorCell.bounds`:

    >>> cell.bounds
    array([0.6  , 0.8  , 0.   , 0.125, 0.   , 0.1  ])

    We can also get its neighboring cells:

    >>> neighbours = cell.get_neighbors(mesh)
    >>> for neighbor in neighbours:
    ...     print(neighbor.center)
    [0.5    0.0625 0.05  ]
    [0.9    0.0625 0.05  ]
    [0.7    0.1875 0.05  ]
    [0.7    0.0625 0.15  ]


    Alternatively, we can iterate over all cells in the mesh with a simple
    *for loop* or list comprehension:

    >>> cells = [cell for cell in mesh]
    >>> len(cells)
    400

    """

    def __init__(self, h, origin, index_unraveled, mesh_shape, dtype=None, device=None):
        self._h = h
        self._origin = origin
        self._index_unraveled = index_unraveled
        self._mesh_shape = mesh_shape

        if dtype is None:
            dtype = torch.float64
        self._dtype = dtype
        self._device = device

    def __repr__(self):
        """Represent a TensorCell."""
        attributes = ", ".join(
            [
                f"{attr}={getattr(self, attr)}"
                for attr in ("h", "origin", "index", "mesh_shape")
            ]
        )
        return f"TensorCell({attributes})"

    def __eq__(self, other):
        """Check if this cell is the same as another one."""
        if not isinstance(other, TensorCell):
            raise TypeError(
                f"Cannot compare an object of type '{other.__class__.__name__}' "
                "with a TensorCell"
            )

        # Use torch.allclose or (self.h == other.h).all() depending on dtype
        h_equal = (
            torch.equal(self.h, other.h)
            if torch.is_tensor(self.h)
            else self.h == other.h
        )
        origin_equal = (
            torch.equal(self.origin, other.origin)
            if torch.is_tensor(self.origin)
            else self.origin == other.origin
        )

        return (
            h_equal
            and origin_equal
            and self.index == other.index
            and self.mesh_shape == other.mesh_shape
        )

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        try:
            device = torch.device(value)  # will raise if invalid
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid device specification: {value!r}") from e

        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA device specified, but no CUDA-capable device is available."
            )

        self._device = device

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        try:
            dtype = torch.dtype(value) if not isinstance(value, torch.dtype) else value
        except Exception as e:
            raise ValueError(f"Invalid dtype provided: {value!r}") from e

        if dtype not in {torch.float16, torch.float32, torch.float64}:
            raise ValueError(
                f"dtype must be a floating-point type (float16, float32, float64), got {dtype}"
            )

        self._dtype = dtype

    @property
    def h(self):
        """Cell widths."""
        return self._h

    @property
    def origin(self):
        """Coordinates of the origin of the cell."""
        return self._origin

    @property
    def index(self):
        """Index of the cell in a TensorMesh (Fortran order)."""
        unraveled = torch.as_tensor(self.index_unraveled, dtype=torch.long)
        shape = torch.as_tensor(self.mesh_shape, dtype=torch.long)

        # Compute Fortran-style strides
        strides = torch.cumprod(
            torch.cat([torch.tensor([1], dtype=torch.long), shape[:-1]]), dim=0
        )

        # Dot product with strides to get Fortran-order flat index
        return int(torch.dot(unraveled, strides))

    @property
    def index_unraveled(self):
        """Unraveled index of the cell in a TensorMesh."""
        return self._index_unraveled

    @property
    def mesh_shape(self):
        """Shape of the parent mesh."""
        return self._mesh_shape

    @property
    def dim(self):
        """Dimensions of the cell (1, 2 or 3)."""
        return len(self.h)

    @property
    def center(self):
        """
        Coordinates of the cell center.

        Returns
        -------
        center : (dim) torch.Tensor
            Tensor with the coordinates of the cell center.
        """
        origin = torch.as_tensor(self.origin, dtype=self.dtype)
        h = torch.as_tensor(self.h, dtype=self.dtype)
        return origin + h / 2

    @property
    def bounds(self):
        """
        Bounds of the cell.

        Coordinates that define the bounds of the cell. Bounds are returned in
        the following order: ``x1``, ``x2``, ``y1``, ``y2``, ``z1``, ``z2``.

        Returns
        -------
        bounds : (2 * dim) array
            Array with the cell bounds.
        """
        bounds = torch.tensor(
            [
                origin_i + factor * h_i
                for origin_i, h_i in zip(self.origin, self.h)
                for factor in (0, 1)
            ],
            dtype=self.dtype,
            device=self.device,
        )
        return bounds

    @property
    def neighbors(self):
        """
        Indices for this cell's neighbors within its parent mesh.

        Returns
        -------
        list of int
        """
        neighbor_indices = []
        for dim in range(self.dim):
            for delta in (-1, 1):
                index = list(self.index_unraveled)
                index[dim] += delta
                if 0 <= index[dim] < self.mesh_shape[dim]:
                    neighbor_indices.append(fortran_ravel_index(index, self.mesh_shape))
        return neighbor_indices

    @property
    def nodes(self):
        """
        Indices for this cell's nodes within its parent mesh.

        Returns
        -------
        list of int
        """

        # Define shape of nodes in parent mesh
        nodes_shape = [s + 1 for s in self.mesh_shape]
        # Get indices of nodes per dimension
        nodes_index_per_dim = [[index, index + 1] for index in self.index_unraveled]
        # Combine the nodes_index_per_dim using itertools.product.
        # Because we want to follow a FORTRAN order, we need to reverse the
        # order of the nodes_index_per_dim and the indices.
        nodes_indices = [i[::-1] for i in itertools.product(*nodes_index_per_dim[::-1])]
        # Ravel indices
        nodes_indices = [
            fortran_ravel_index(index, nodes_shape) for index in nodes_indices
        ]
        return nodes_indices

    @property
    def edges(self):
        """
        Indices for this cell's edges within its parent mesh.

        Returns
        -------
        list of int
        """

        if self.dim == 1:
            return [self.index]

        if self.dim == 2:
            # x-edges (horizontal): (nx, ny+1)
            edges_x_shape = [self.mesh_shape[0], self.mesh_shape[1] + 1]
            # y-edges (vertical): (nx+1, ny)
            edges_y_shape = [self.mesh_shape[0] + 1, self.mesh_shape[1]]
            n_edges_x = edges_x_shape[0] * edges_x_shape[1]

            # x-edges: two horizontal edges for this cell
            edges_x = [
                fortran_ravel_index(
                    [self.index_unraveled[0], self.index_unraveled[1] + delta],
                    edges_x_shape,
                )
                for delta in (0, 1)
            ]

            # y-edges: two vertical edges for this cell
            edges_y = [
                n_edges_x
                + fortran_ravel_index(
                    [self.index_unraveled[0] + delta, self.index_unraveled[1]],
                    edges_y_shape,
                )
                for delta in (0, 1)
            ]

            return edges_x + edges_y

        if self.dim == 3:
            edges_x_shape = [
                self.mesh_shape[0],
                self.mesh_shape[1] + 1,
                self.mesh_shape[2] + 1,
            ]
            edges_y_shape = [
                self.mesh_shape[0] + 1,
                self.mesh_shape[1],
                self.mesh_shape[2] + 1,
            ]
            edges_z_shape = [
                self.mesh_shape[0] + 1,
                self.mesh_shape[1] + 1,
                self.mesh_shape[2],
            ]

            n_edges_x = torch.tensor(edges_x_shape).prod().item()
            n_edges_y = torch.tensor(edges_y_shape).prod().item()

            # x-edges
            edges_x = [
                fortran_ravel_index(
                    [
                        self.index_unraveled[0],
                        self.index_unraveled[1] + dy,
                        self.index_unraveled[2] + dz,
                    ],
                    edges_x_shape,
                )
                for dz in (0, 1)
                for dy in (0, 1)
            ]

            # y-edges
            edges_y = [
                n_edges_x
                + fortran_ravel_index(
                    [
                        self.index_unraveled[0] + dx,
                        self.index_unraveled[1],
                        self.index_unraveled[2] + dz,
                    ],
                    edges_y_shape,
                )
                for dz in (0, 1)
                for dx in (0, 1)
            ]

            # z-edges
            edges_z = [
                n_edges_x
                + n_edges_y
                + fortran_ravel_index(
                    [
                        self.index_unraveled[0] + dx,
                        self.index_unraveled[1] + dy,
                        self.index_unraveled[2],
                    ],
                    edges_z_shape,
                )
                for dy in (0, 1)
                for dx in (0, 1)
            ]

            return edges_x + edges_y + edges_z

    @property
    def faces(self):
        """
        Indices for cell's faces within its parent mesh.

        Returns
        -------
        list of int
        """

        if self.dim == 1:
            faces_indices = [self.index, self.index + 1]

        elif self.dim == 2:
            faces_x_shape = [self.mesh_shape[0] + 1, self.mesh_shape[1]]
            faces_y_shape = [self.mesh_shape[0], self.mesh_shape[1] + 1]
            n_faces_x = faces_x_shape[0] * faces_x_shape[1]

            faces_x_indices = [
                fortran_ravel_index(
                    [self.index_unraveled[0] + delta, self.index_unraveled[1]],
                    faces_x_shape,
                )
                for delta in (0, 1)
            ]

            faces_y_indices = [
                n_faces_x
                + fortran_ravel_index(
                    [self.index_unraveled[0], self.index_unraveled[1] + delta],
                    faces_y_shape,
                )
                for delta in (0, 1)
            ]

            faces_indices = faces_x_indices + faces_y_indices

        elif self.dim == 3:
            faces_x_shape = [
                n + 1 if i == 0 else n for i, n in enumerate(self.mesh_shape)
            ]
            faces_y_shape = [
                n + 1 if i == 1 else n for i, n in enumerate(self.mesh_shape)
            ]
            faces_z_shape = [
                n + 1 if i == 2 else n for i, n in enumerate(self.mesh_shape)
            ]

            n_faces_x = faces_x_shape[0] * faces_x_shape[1] * faces_x_shape[2]
            n_faces_y = faces_y_shape[0] * faces_y_shape[1] * faces_y_shape[2]

            faces_x_indices = [
                fortran_ravel_index(
                    [
                        self.index_unraveled[0] + delta,
                        self.index_unraveled[1],
                        self.index_unraveled[2],
                    ],
                    faces_x_shape,
                )
                for delta in (0, 1)
            ]

            faces_y_indices = [
                n_faces_x
                + fortran_ravel_index(
                    [
                        self.index_unraveled[0],
                        self.index_unraveled[1] + delta,
                        self.index_unraveled[2],
                    ],
                    faces_y_shape,
                )
                for delta in (0, 1)
            ]

            faces_z_indices = [
                n_faces_x
                + n_faces_y
                + fortran_ravel_index(
                    [
                        self.index_unraveled[0],
                        self.index_unraveled[1],
                        self.index_unraveled[2] + delta,
                    ],
                    faces_z_shape,
                )
                for delta in (0, 1)
            ]

            faces_indices = faces_x_indices + faces_y_indices + faces_z_indices

        return faces_indices

    def get_neighbors(self, mesh):
        """
        Return the neighboring cells in the mesh.

        Parameters
        ----------
        mesh : TensorMesh
            TensorMesh where the current cell lives.

        Returns
        -------
        list of TensorCell
        """
        return [mesh[index] for index in self.neighbors]
