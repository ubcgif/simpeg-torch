import torch
from .base_mesh import BaseMesh

from ..utils import atleast_1d, Identity, mkvc


class BaseRegularMesh(BaseMesh):

    _aliases = {
        **BaseMesh._aliases,
        "nEx": "n_edges_x",
        "nEy": "n_edges_y",
        "nEz": "n_edges_z",
        "vnE": "n_edges_per_direction",
        "nFx": "n_faces_x",
        "nFy": "n_faces_y",
        "nFz": "n_faces_z",
        "vnF": "n_faces_per_direction",
        "vnC": "shape_cells",
    }

    _items = {"shape_cells", "origin", "orientation", "reference_system"}

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
        self._dtype = dtype

        if device is None:
            device = "cpu"
        self._device = device

        if origin is None:
            origin = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self._origin = origin

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
    def orientation(self):
        """Rotation matrix defining mesh axes relative to Cartesian.

        This property returns a rotation matrix between the local coordinate
        axes of the mesh and the standard Cartesian axes. For a 3D mesh, this
        would define the x, y and z axes of the mesh relative to the Easting,
        Northing and elevation directions. The *orientation* property can
        be used to transform locations from a local coordinate
        system to a conventional Cartesian system. By default, *orientation*
        is an identity matrix of shape (mesh.dim, mesh.dim).

        Returns
        -------
        (dim, dim) numpy.ndarray of float
            Square rotation matrix defining orientation

        Examples
        --------
        For a visual example of this, please see the figure in the
        docs for :class:`~discretize.mixins.InterfaceVTK`.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):

        if isinstance(value, Identity):
            self._orientation = torch.eye(self.dim, dtype=torch.float64)
        else:
            R = torch.atleast_2d(torch.as_tensor(value, dtype=torch.float64))
            dim = self.dim
            if R.shape != (dim, dim):
                raise ValueError(
                    f"Orientation matrix must be square and of shape {(dim, dim)}, got {tuple(R.shape)}"
                )
            # Normalize each row
            R = R / torch.linalg.norm(R, dim=1, keepdim=True)

            # Check orthogonality: R @ R.T should be close to identity
            if not torch.allclose(
                R @ R.T, torch.eye(dim, dtype=torch.float64), rtol=1e-5, atol=1e-6
            ):
                raise ValueError("Orientation matrix is not orthogonal")

            self._orientation = R

    @property
    def reference_system(self):
        """Coordinate reference system.

        The type of coordinate reference frame. Will be one of the values "cartesian",
        "cylindrical", or "spherical".

        Returns
        -------
        str {'cartesian', 'cylindrical', 'spherical'}
            The coordinate system associated with the mesh.
        """
        return self._reference_system

    @reference_system.setter
    def reference_system(self, value):
        """Check if the reference system is of a known type."""
        choices = ["cartesian", "cylindrical", "spherical"]
        # Here are a few abbreviations that users can harnes
        abrevs = {
            "car": choices[0],
            "cart": choices[0],
            "cy": choices[1],
            "cyl": choices[1],
            "sph": choices[2],
        }
        # Get the name and fix it if it is abbreviated
        value = value.lower()
        value = abrevs.get(value, value)
        if value not in choices:
            raise ValueError(
                "Coordinate system ({}) unknown.".format(self.reference_system)
            )
        self._reference_system = value

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (0, 1, 1))], device=self.device
            )
        )

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (1, 0, 1))], device=self.device
            )
        )

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (1, 1, 0))], device=self.device
            )
        )

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (1, 0, 0))], device=self.device
            )
        )

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (0, 1, 0))], device=self.device
            )
        )

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
        return int(
            torch.prod(
                [x + y for x, y in zip(self.shape_cells, (0, 0, 1))], device=self.device
            )
        )

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
            nX = torch.cat(
                [
                    torch.ones(
                        (self.n_faces_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_faces_x, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            nY = torch.cat(
                [
                    torch.zeros(
                        (self.n_faces_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_faces_y, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            return torch.cat([nX, nY], dim=0)

        elif self.dim == 3:
            nX = torch.cat(
                [
                    torch.ones(
                        (self.n_faces_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_faces_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_faces_x, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            nY = torch.cat(
                [
                    torch.zeros(
                        (self.n_faces_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_faces_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_faces_y, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            nZ = torch.cat(
                [
                    torch.zeros(
                        (self.n_faces_z, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_faces_z, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_faces_z, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            return torch.cat([nX, nY, nZ], dim=0)

    @property
    def edge_tangents(self):
        """Return unit vectors tangent to each edge, on the correct device and dtype."""
        if self.dim == 2:
            tX = torch.cat(
                [
                    torch.ones(
                        (self.n_edges_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_edges_x, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            tY = torch.cat(
                [
                    torch.zeros(
                        (self.n_edges_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_edges_y, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            return torch.cat([tX, tY], dim=0)

        elif self.dim == 3:
            tX = torch.cat(
                [
                    torch.ones(
                        (self.n_edges_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_edges_x, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_edges_x, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            tY = torch.cat(
                [
                    torch.zeros(
                        (self.n_edges_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_edges_y, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_edges_y, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            tZ = torch.cat(
                [
                    torch.zeros(
                        (self.n_edges_z, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (self.n_edges_z, 1), dtype=self.dtype, device=self.device
                    ),
                    torch.ones(
                        (self.n_edges_z, 1), dtype=self.dtype, device=self.device
                    ),
                ],
                dim=1,
            )
            return torch.cat([tX, tY, tZ], dim=0)


class BaseRectangularMesh(BaseRegularMesh):
    """Base rectangular mesh class for the ``discretize`` package.

    The ``BaseRectangularMesh`` class acts as an extension of the
    :class:`~discretize.base.BaseRegularMesh` classes with a regular structure.
    """

    _aliases = {
        **BaseRegularMesh._aliases,
        **{
            "vnN": "shape_nodes",
            "vnEx": "shape_edges_x",
            "vnEy": "shape_edges_y",
            "vnEz": "shape_edges_z",
            "vnFx": "shape_faces_x",
            "vnFy": "shape_faces_y",
            "vnFz": "shape_faces_z",
        },
    }

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
        return int(
            torch.prod(
                torch.tensor(self.shape_cells, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(
            torch.prod(
                torch.tensor(self.shape_nodes, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_edges_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(
            torch.prod(
                torch.tensor(self.shape_edges_x, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_edges_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(
            torch.prod(
                torch.tensor(self.shape_edges_y, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_edges_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(
            torch.prod(
                torch.tensor(self.shape_edges_z, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_faces_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(
            torch.prod(
                torch.tensor(self.shape_faces_x, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_faces_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(
            torch.prod(
                torch.tensor(self.shape_faces_y, dtype=torch.int64, device=self.device)
            ).item()
        )

    @property
    def n_faces_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(
            torch.prod(
                torch.tensor(self.shape_faces_z, dtype=torch.int64, device=self.device)
            ).item()
        )

    def reshape(
        self,
        x,
        x_type="cell_centers",
        out_type="cell_centers",
        return_format="V",
        **kwargs,
    ):
        """Reshape tensor quantities.

        **Reshape** is a quick command that will do its best to reshape discrete
        quantities living on meshes than inherit the :class:`discretize.base_mesh.RectangularMesh`
        class. For example, you may have a 1D tensor defining a vector on mesh faces, and you would
        like to extract the x-component and reshaped it to a 3D matrix.

        Parameters
        ----------
        x : torch.Tensor or list of torch.Tensor
            The input quantity, tensor or a list
        x_type : {'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Defines the locations on the mesh where input parameter *x* lives.
        out_type : str
            Defines the output quantity. Choice depends on your input for *x_type*:

            - *x_type* = 'CC' ---> *out_type* = 'CC'
            - *x_type* = 'N' ---> *out_type* = 'N'
            - *x_type* = 'F' ---> *out_type* = {'F', 'Fx', 'Fy', 'Fz'}
            - *x_type* = 'E' ---> *out_type* = {'E', 'Ex', 'Ey', 'Ez'}
        return_format : str
            The dimensions of quantity being returned

            - *V:* return a vector (1D tensor) or a list of vectors
            - *M:* return matrix (nD tensor) or a list of matrices

        """
        if "xType" in kwargs:
            raise TypeError(
                "The xType keyword argument has been removed, please use x_type. "
                "This will be removed in discretize 1.0.0"
            )
        if "outType" in kwargs:
            raise TypeError(
                "The outType keyword argument has been removed, please use out_type. "
                "This will be removed in discretize 1.0.0",
            )
        if "format" in kwargs:
            raise TypeError(
                "The format keyword argument has been removed, please use return_format. "
                "This will be removed in discretize 1.0.0",
            )

        x_type = self._parse_location_type(x_type)
        out_type = self._parse_location_type(out_type)

        allowed_x_type = [
            "cell_centers",
            "nodes",
            "faces",
            "faces_x",
            "faces_y",
            "faces_z",
            "edges",
            "edges_x",
            "edges_y",
            "edges_z",
        ]
        if not (isinstance(x, list) or isinstance(x, torch.Tensor)):
            raise TypeError("x must be either a list or a torch.Tensor")
        if x_type not in allowed_x_type:
            raise ValueError(
                "x_type must be either '" + "', '".join(allowed_x_type) + "'"
            )
        if out_type not in allowed_x_type:
            raise ValueError(
                "out_type must be either '" + "', '".join(allowed_x_type) + "'"
            )
        if return_format not in ["M", "V"]:
            raise ValueError("return_format must be either 'M' or 'V'")
        if out_type[: len(x_type)] != x_type:
            raise ValueError("You cannot change types when reshaping.")
        if x_type not in out_type:
            raise ValueError("You cannot change type of components.")

        if isinstance(x, list):
            for i, xi in enumerate(x):
                if not isinstance(xi, torch.Tensor):
                    raise TypeError("x[{0:d}] must be a torch.Tensor".format(i))
                if xi.numel() != x[0].numel():
                    raise ValueError("Number of elements in list must not change.")

            x_tensor = torch.ones(
                (x[0].numel(), len(x)), dtype=self.dtype, device=self.device
            )
            # Unwrap it and put it in a tensor
            for i, xi in enumerate(x):
                x_tensor[:, i] = mkvc(xi)
            x = x_tensor

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")

        x = x.clone()  # make a copy.
        x_type_is_FE_xyz = (
            len(x_type) > 1
            and x_type[0] in ["f", "e"]
            and x_type[-1] in ["x", "y", "z"]
        )

        def outKernal(xx, nn):
            """Return xx as either a matrix (shape == nn) or a vector."""
            if return_format == "M":
                return xx.view(*nn)
            elif return_format == "V":
                return mkvc(xx)

        def switchKernal(xx):
            """Switch over the different options."""
            if x_type in ["cell_centers", "nodes"]:
                nn = self.shape_cells if x_type == "cell_centers" else self.shape_nodes
                if (
                    xx.numel()
                    != torch.prod(torch.tensor(nn, device=self.device)).item()
                ):
                    raise ValueError("Number of elements must not change.")
                return outKernal(xx, nn)
            elif x_type in ["faces", "edges"]:
                # This will only deal with components of fields,
                # not full 'F' or 'E'
                xx = mkvc(xx)  # unwrap it in case it is a matrix
                if x_type == "faces":
                    nn = (self.n_faces_x, self.n_faces_y, self.n_faces_z)[: self.dim]
                else:
                    nn = (self.n_edges_x, self.n_edges_y, self.n_edges_z)[: self.dim]
                nn = torch.tensor([0] + list(nn), device=self.device)

                nx = [None, None, None]
                nx[0] = self.shape_faces_x if x_type == "faces" else self.shape_edges_x
                nx[1] = self.shape_faces_y if x_type == "faces" else self.shape_edges_y
                nx[2] = self.shape_faces_z if x_type == "faces" else self.shape_edges_z

                for dim, dimName in enumerate(["x", "y", "z"]):
                    if dimName in out_type:
                        if self.dim <= dim:
                            raise ValueError(
                                "Dimensions of mesh not great enough for "
                                "{}_{}".format(x_type, dimName)
                            )
                        if xx.numel() != torch.sum(nn).item():
                            raise ValueError("Vector is not the right size.")
                        start = torch.sum(nn[: dim + 1]).item()
                        end = torch.sum(nn[: dim + 2]).item()
                        return outKernal(xx[start:end], nx[dim])

            elif x_type_is_FE_xyz:
                # This will deal with partial components (x, y or z)
                # lying on edges or faces
                if "x" in x_type:
                    nn = self.shape_faces_x if "f" in x_type else self.shape_edges_x
                elif "y" in x_type:
                    nn = self.shape_faces_y if "f" in x_type else self.shape_edges_y
                elif "z" in x_type:
                    nn = self.shape_faces_z if "f" in x_type else self.shape_edges_z
                if (
                    xx.numel()
                    != torch.prod(torch.tensor(nn, device=self.device)).item()
                ):
                    raise ValueError(
                        f"Vector is not the right size. Expected {torch.prod(torch.tensor(nn, device=self.device)).item()}, got {xx.numel()}"
                    )
                return outKernal(xx, nn)

        # Check if we are dealing with a vector quantity
        isVectorQuantity = len(x.shape) == 2 and x.shape[1] == self.dim

        if out_type in ["faces", "edges"]:
            if isVectorQuantity:
                raise ValueError("Not sure what to do with a vector vector quantity..")
            outTypeCopy = out_type
            out = ()
            for dirName in ["x", "y", "z"][: self.dim]:
                out_type = outTypeCopy + "_" + dirName
                out += (switchKernal(x),)
            return out
        elif isVectorQuantity:
            out = ()
            for ii in range(x.shape[1]):
                out += (switchKernal(x[:, ii]),)
            return out
        else:
            return switchKernal(x)
