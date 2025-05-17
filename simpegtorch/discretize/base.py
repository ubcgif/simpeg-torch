import torch
from .utils import is_scalar, atleast_1d


class BaseRegularMesh:
    def __init__(
        self,
        shape_cells,
        origin=None,
        dtype=None,
        device=None,
    ):
        shape_cells = tuple((int(val) for val in shape_cells))
        self._shape_cells = shape_cells

        if dtype is None:
            dtype = torch.float64
        self.dtype = dtype

        if device is None:
            device = "cpu"
        self.device = device

        if origin is None:
            origin = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.origin = origin


    @property
    def origin(self):
        """Origin or 'anchor point' of the mesh.

        For a mesh defined in Cartesian coordinates (e.g.
        :class:`~discretize.TensorMesh`, :class:`~discretize.CylindricalMesh`,
        :class:`~discretize.TreeMesh`), *origin* is the
        bottom southwest corner. For a :class:`~discretize.CylindricalMesh`,
        *origin* is the bottom of the axis of rotational symmetry
        for the mesh (i.e. bottom of z-axis).

        Returns
        -------
        (dim) numpy.ndarray of float
            origin location
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        # ensure the value is a numpy array
        value = torch.as_tensor(value)
        value = atleast_1d(value)
        if len(value) != self.dim:
            raise ValueError(
                f"origin and shape must be the same length, got {len(value)} and {self.dim}"
            )
        self._origin = value

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
            raise ValueError("CUDA device specified, but no CUDA-capable device is available.")

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
    def shape_cells(self):
        """Number of cells in each coordinate direction.

        For meshes of class :class:`~discretize.TensorMesh`,
        :class:`~discretize.CylindricalMesh` or :class:`~discretize.CurvilinearMesh`,
        **shape_cells** returns the number of cells along each coordinate axis direction.
        For mesh of class :class:`~discretize.TreeMesh`, *shape_cells* returns
        the number of underlying tensor mesh cells along each coordinate direction.

        Returns
        -------
        (dim) tuple of int
            the number of cells in each coordinate direcion

        Notes
        -----
        Property also accessible as using the shorthand **vnC**
        """
        return self._shape_cells

    @property
    def dim(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return len(self.shape_cells)

    @property
    def n_cells(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(torch.prod(torch.tensor(self.shape_cells, device=self.device)))

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(torch.prod([x + 1 for x in self.shape_cells], device=self.device))

    @property
    def n_edges_x(self):
        """Number of x-edges in the mesh.

        This property returns the number of edges that
        are parallel to the x-axis; i.e. x-edges.

        Returns
        -------
        int
            Number of x-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEx**
        """
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 1))], device=self.device))

    @property
    def n_edges_y(self):
        """Number of y-edges in the mesh.

        This property returns the number of edges that
        are parallel to the y-axis; i.e. y-edges.

        Returns
        -------
        int
            Number of y-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEy**
        """
        if self.dim < 2:
            return None
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 1))], device=self.device))

    @property
    def n_edges_z(self):
        """Number of z-edges in the mesh.

        This property returns the number of edges that
        are parallel to the z-axis; i.e. z-edges.

        Returns
        -------
        int
            Number of z-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEz**
        """
        if self.dim < 3:
            return None
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (1, 1, 0))], device=self.device))

    @property
    def n_edges_per_direction(self):
        """The number of edges in each direction.

        This property returns a tuple with the number of edges
        in each axis direction of the mesh. For a 3D mesh,
        *n_edges_per_direction* would return a tuple of the form
        (nEx, nEy, nEz). Thus the length of the
        tuple depends on the dimension of the mesh.

        Returns
        -------
        (dim) tuple of int
            Number of edges in each direction

        Notes
        -----
        Property also accessible as using the shorthand **vnE**

        Examples
        --------
        >>> import discretize
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(edges=True)
        >>> plt.show()
        """
        return tuple(
            x for x in [self.n_edges_x, self.n_edges_y, self.n_edges_z] if x is not None
        )

    @property
    def n_edges(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        n = self.n_edges_x
        if self.dim > 1:
            n += self.n_edges_y
        if self.dim > 2:
            n += self.n_edges_z
        return n

    @property
    def n_faces_x(self):
        """Number of x-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the x-axis; i.e. x-faces.

        Returns
        -------
        int
            Number of x-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFx**
        """
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 0))], device=self.device))

    @property
    def n_faces_y(self):
        """Number of y-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the y-axis; i.e. y-faces.

        Returns
        -------
        int
            Number of y-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFy**
        """
        if self.dim < 2:
            return None
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 0))], device=self.device))

    @property
    def n_faces_z(self):
        """Number of z-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the z-axis; i.e. z-faces.

        Returns
        -------
        int
            Number of z-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFz**
        """
        if self.dim < 3:
            return None
        return int(torch.prod([x + y for x, y in zip(self.shape_cells, (0, 0, 1))], device=self.device))

    @property
    def n_faces_per_direction(self):
        """The number of faces in each axis direction.

        This property returns a tuple with the number of faces
        in each axis direction of the mesh. For a 3D mesh,
        *n_faces_per_direction* would return a tuple of the form
        (nFx, nFy, nFz). Thus the length of the
        tuple depends on the dimension of the mesh.

        Returns
        -------
        (dim) tuple of int
            Number of faces in each axis direction

        Notes
        -----
        Property also accessible as using the shorthand **vnF**

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(faces=True)
        >>> plt.show()
        """
        return tuple(
            x for x in [self.n_faces_x, self.n_faces_y, self.n_faces_z] if x is not None
        )

    @property
    def n_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        n = self.n_faces_x
        if self.dim > 1:
            n += self.n_faces_y
        if self.dim > 2:
            n += self.n_faces_z
        return n

    @property
    def face_normals(self):

        if self.dim == 2:
            nX = torch.cat([
                torch.ones((self.n_faces_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_faces_x, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            nY = torch.cat([
                torch.zeros((self.n_faces_y, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_faces_y, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            return torch.cat([nX, nY], dim=0)

        elif self.dim == 3:
            nX = torch.cat([
                torch.ones((self.n_faces_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_faces_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_faces_x, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            nY = torch.cat([
                torch.zeros((self.n_faces_y, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_faces_y, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_faces_y, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            nZ = torch.cat([
                torch.zeros((self.n_faces_z, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_faces_z, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_faces_z, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            return torch.cat([nX, nY, nZ], dim=0)

    @property
    def edge_tangents(self):
        """Return unit vectors tangent to each edge, on the correct device and dtype."""
        if self.dim == 2:
            tX = torch.cat([
                torch.ones((self.n_edges_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_edges_x, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            tY = torch.cat([
                torch.zeros((self.n_edges_y, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_edges_y, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            return torch.cat([tX, tY], dim=0)

        elif self.dim == 3:
            tX = torch.cat([
                torch.ones((self.n_edges_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_edges_x, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_edges_x, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            tY = torch.cat([
                torch.zeros((self.n_edges_y, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_edges_y, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_edges_y, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            tZ = torch.cat([
                torch.zeros((self.n_edges_z, 1), dtype=self.dtype, device=self.device),
                torch.zeros((self.n_edges_z, 1), dtype=self.dtype, device=self.device),
                torch.ones((self.n_edges_z, 1), dtype=self.dtype, device=self.device)
            ], dim=1)
            return torch.cat([tX, tY, tZ], dim=0)

class BaseRectangularMesh(BaseRegularMesh):
    """Base rectangular mesh class for the ``discretize`` package.

    The ``BaseRectangularMesh`` class acts as an extension of the
    :class:`~discretize.base.BaseRegularMesh` classes with a regular structure.
    """

    @property
    def shape_nodes(self):
        """The number of nodes along each axis direction.

        This property returns a tuple containing the number of nodes along
        each axis direction. The length of the tuple is equal to the
        dimension of the mesh; i.e. 1, 2 or 3.

        Returns
        -------
        (dim) tuple of int
            Number of nodes along each axis direction

        Notes
        -----
        Property also accessible as using the shorthand **vnN**
        """
        return tuple(x + 1 for x in self.shape_cells)

    @property
    def shape_edges_x(self):
        """Number of x-edges along each axis direction.

        This property returns a tuple containing the number of x-edges
        along each axis direction. The length of the tuple is equal to the
        dimension of the mesh; i.e. 1, 2 or 3.

        Returns
        -------
        (dim) tuple of int
            Number of x-edges along each axis direction

            - *1D mesh:* `(n_cells_x)`
            - *2D mesh:* `(n_cells_x, n_nodes_y)`
            - *3D mesh:* `(n_cells_x, n_nodes_y, n_nodes_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEx**
        """
        return self.shape_cells[:1] + self.shape_nodes[1:]

    @property
    def shape_edges_y(self):
        """Number of y-edges along each axis direction.

        This property returns a tuple containing the number of y-edges
        along each axis direction. If `dim` is 1, there are no y-edges.

        Returns
        -------
        None or (dim) tuple of int
            Number of y-edges along each axis direction

            - *1D mesh: None*
            - *2D mesh:* `(n_nodes_x, n_cells_y)`
            - *3D mesh:* `(n_nodes_x, n_cells_y, n_nodes_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEy**
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sn[0], sc[1]) + sn[2:]  # conditionally added if dim == 3!

    @property
    def shape_edges_z(self):
        """Number of z-edges along each axis direction.

        This property returns a tuple containing the number of z-edges
        along each axis direction. There are only z-edges if `dim` is 3.

        Returns
        -------
        None or (dim) tuple of int
            Number of z-edges along each axis direction.

            - *1D mesh: None*
            - *2D mesh: None*
            - *3D mesh:* `(n_nodes_x, n_nodes_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEz**
        """
        if self.dim < 3:
            return None
        return self.shape_nodes[:2] + self.shape_cells[2:]

    @property
    def shape_faces_x(self):
        """Number of x-faces along each axis direction.

        This property returns a tuple containing the number of x-faces
        along each axis direction.

        Returns
        -------
        (dim) tuple of int
            Number of x-faces along each axis direction

            - *1D mesh:* `(n_nodes_x)`
            - *2D mesh:* `(n_nodes_x, n_cells_y)`
            - *3D mesh:* `(n_nodes_x, n_cells_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnFx**
        """
        return self.shape_nodes[:1] + self.shape_cells[1:]

    @property
    def shape_faces_y(self):
        """Number of y-faces along each axis direction.

        This property returns a tuple containing the number of y-faces
        along each axis direction. If `dim` is 1, there are no y-edges.

        Returns
        -------
        None or (dim) tuple of int
            Number of y-faces along each axis direction

            - *1D mesh: None*
            - *2D mesh:* `(n_cells_x, n_nodes_y)`
            - *3D mesh:* `(n_cells_x, n_nodes_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnFy**
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sc[0], sn[1]) + sc[2:]

    @property
    def shape_faces_z(self):
        """Number of z-faces along each axis direction.

        This property returns a tuple containing the number of z-faces
        along each axis direction. There are only z-faces if `dim` is 3.

        Returns
        -------
        None or (dim) tuple of int
            Number of z-faces along each axis direction.

                - *1D mesh: None*
                - *2D mesh: None*
                - *3D mesh:* (n_cells_x, n_cells_y, n_nodes_z)

        Notes
        -----
        Property also accessible as using the shorthand **vnFz**
        """
        if self.dim < 3:
            return None
        return self.shape_cells[:2] + self.shape_nodes[2:]

    ##################################
    # Redo the numbering so they are dependent of the shape tuples
    # these should all inherit the parent's docstrings
    ##################################

    @property
    def n_cells(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(torch.prod(torch.tensor(self.shape_cells, dtype=torch.int64, device=self.device)).item())

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(torch.prod(torch.tensor(self.shape_nodes, dtype=torch.int64, device=self.device)).item())

    @property
    def n_edges_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(torch.prod(torch.tensor(self.shape_edges_x, dtype=torch.int64, device=self.device)).item())

    @property
    def n_edges_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(torch.prod(torch.tensor(self.shape_edges_y, dtype=torch.int64, device=self.device)).item())

    @property
    def n_edges_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(torch.prod(torch.tensor(self.shape_edges_z, dtype=torch.int64, device=self.device)).item())

    @property
    def n_faces_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(torch.prod(torch.tensor(self.shape_faces_x, dtype=torch.int64, device=self.device)).item())

    @property
    def n_faces_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(torch.prod(torch.tensor(self.shape_faces_y, dtype=torch.int64, device=self.device)).item())

    @property
    def n_faces_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(torch.prod(torch.tensor(self.shape_faces_z, dtype=torch.int64, device=self.device)).item())
