import torch
from ..utils import (
    is_scalar,
    make_boundary_bool,
    unpack_widths,
    ndgrid,
    as_array_n_by_dim,
    interpolation_matrix,
)
from .base_regular_mesh import BaseRectangularMesh, BaseRegularMesh


class BaseTensorMesh(BaseRectangularMesh):
    """Base class for tensor-product style meshes.

    This class contains properites and methods that are common to Cartesian
    and cylindrical meshes. That is, meshes whose cell centers, nodes, faces
    and edges can be constructed with tensor-products of vectors.

    Do not use this class directly! Practical tensor meshes supported in
    discretize will inherit this class; i.e. :class:`discretize.TensorMesh`
    and :class:`~discretize.CylindricalMesh`. Inherit this class if you plan
    to develop a new tensor-style mesh class (e.g. a spherical mesh).

    Parameters
    ----------
    h : (dim) iterable of int, torch.tensor, or tuple
        Defines the cell widths along each axis. The length of the iterable object is
        equal to the dimension of the mesh (1, 2 or 3). For a 3D mesh, the list would
        have the form *[hx, hy, hz]* .

        Along each axis, the user has 3 choices for defining the cells widths:

        - :class:`int` -> A unit interval is equally discretized into `N` cells.
        - :class:`torch.tensor` -> The widths are explicity given for each cell
        - the widths are defined as a :class:`list` of :class:`tuple` of the form *(dh, nc, [npad])*
          where *dh* is the cell width, *nc* is the number of cells, and *npad* (optional)
          is a padding factor denoting exponential increase/decrease in the cell width
          for each cell; e.g. *[(2., 10, -1.3), (2., 50), (2., 10, 1.3)]*

    origin : (dim) iterable, default: 0
        Define the origin or 'anchor point' of the mesh; i.e. the bottom-left-frontmost
        corner. By default, the mesh is anchored such that its origin is at
        ``[0, 0, 0]``.

        For each dimension (x, y or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate along
          each axis is the first node location ('0'), in the center ('C') or the last
          node location ('N').

    See Also
    --------
    utils.unpack_widths :
        The function used to expand a ``list`` or ``tuple`` to generate widths.
    """

    _unitDimensions = [1, 1, 1]
    _items = {"h"} | BaseRegularMesh._items

    def __init__(self, h, origin=None, shape_cells=None, dtype=None, device=None):
        if dtype is None:
            dtype = torch.float64

        if device is None:
            device = "cpu"
        self._device = device

        self.dtype = dtype
        self.device = device
        try:
            h = list(h)  # ensure value is a list (and make a copy)
        except TypeError:
            raise TypeError("h must be an iterable object, not {}".format(type(h)))
        if len(h) == 0 or len(h) > 3:
            raise ValueError("h must be of dimension 1, 2, or 3 not {}".format(len(h)))
        # expand value
        for i, h_i in enumerate(h):
            if is_scalar(h_i) and not isinstance(h_i, torch.Tensor):
                # This gives you something over the unit cube.
                h_i = (
                    self._unitDimensions[i]
                    * torch.ones(int(h_i), device=self.device)
                    / int(h_i)
                )
            elif isinstance(h_i, (list, tuple)):
                h_i = unpack_widths(h_i, dtype=self.dtype, device=self.device)
            if not isinstance(h_i, torch.Tensor):
                try:
                    h_i = torch.tensor(h_i, device=self.device, dtype=self.dtype)
                except TypeError:
                    raise TypeError(
                        "h[{0:d}] cannot be cast to a torch tensor.".format(i)
                    )
            if len(h_i.shape) != 1:
                raise ValueError("h[{0:d}] must be a 1D array.".format(i))
            h[i] = h_i.to(
                device=self.device, dtype=self.dtype
            )  # make a copy and ensure correct device
        self._h = tuple(h)

        shape_cells = tuple([len(h_i) for h_i in h])
        super().__init__(
            shape_cells=shape_cells, dtype=dtype, device=device
        )  # do not pass origin here
        if origin is not None:
            if isinstance(origin, torch.Tensor):
                origin = origin.to(self.device)
            elif isinstance(origin, list):
                origin = torch.tensor(origin, device=self.device)
            self.origin = origin

    @property
    def h(self):
        r"""Cell widths along each axis direction.

        The widths of the cells along each axis direction are returned
        as a tuple of 1D arrays; e.g. (hx, hy, hz) for a 3D mesh.
        The lengths of the 1D arrays in the tuple are given by
        :py:attr:`~discretize.base.BaseRegularMesh.shape_cells`. Ordering
        begins at the bottom southwest corner. These are the
        cell widths used when creating the mesh.

        Returns
        -------
        (dim) tuple of torch.tensor
            Cell widths along each axis direction. This depends on the mesh class:

            - :class:`~discretize.TensorMesh`: cell widths along the *x* , [*y* and *z* ] directions
            - :class:`~discretize.CylindricalMesh`: cell widths along the *r*, :math:`\phi` and *z* directions
            - :class:`~discretize.TreeMesh`: cells widths of the *underlying tensor mesh* along the *x* , *y* [and *z* ] directions
        """
        return self._h

    @BaseRegularMesh.origin.setter
    def origin(self, value):  # NOQA D102
        # ensure value is a 1D array at all times
        try:
            value = list(value)
        except TypeError:
            raise TypeError("origin must be iterable")
        if len(value) != self.dim:
            raise ValueError("Dimension mismatch. len(origin) != len(h)")
        for i, (val, h_i) in enumerate(zip(value, self.h)):
            if val == "C":
                value[i] = -h_i.sum() * 0.5
            elif val == "N":
                value[i] = -h_i.sum()
            elif val == "0":
                value[i] = 0
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self._origin = value

    @property
    def nodes_x(self):
        """Return x-coordinates of the nodes along the x-direction.

        This property returns a vector containing the x-coordinate values of
        the nodes along the x-direction. For instances of
        :class:`~discretize.TensorMesh` or :class:`~discretize.CylindricalMesh`,
        this is equivalent to the node positions which define the tensor along
        the x-axis. For instances of :class:`~discretize.TreeMesh` however, this
        property returns the x-coordinate values of the nodes along the x-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_x) torch.tensor of float
            A 1D array containing the x-coordinates of the nodes along
            the x-direction.

        """
        print(f"Origin: {self.origin}")
        print(f"H: {self.h}")
        return torch.cat([self.origin[0].unsqueeze(0), self.h[0]]).cumsum(0)

    @property
    def nodes_y(self):
        """Return y-coordinates of the nodes along the y-direction.

        For 2D and 3D meshes, this property returns a vector
        containing the y-coordinate values of the nodes along the
        y-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the y-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the y-coordinate values of the nodes along the y-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_y) torch.tensor of float or None
            A 1D array containing the y-coordinates of the nodes along
            the y-direction. Returns *None* for 1D meshes.

        """
        return (
            None
            if self.dim < 2
            else torch.cat([self.origin[1].unsqueeze(0), self.h[1]]).cumsum(0)
        )

    @property
    def nodes_z(self):
        """Return z-coordinates of the nodes along the z-direction.

        For 3D meshes, this property returns a 1D vector
        containing the z-coordinate values of the nodes along the
        z-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the z-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the z-coordinate values of the nodes along the z-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_z) torch.tensor of float or None
            A 1D array containing the z-coordinates of the nodes along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        return (
            None
            if self.dim < 3
            else torch.cat([self.origin[2].unsqueeze(0), self.h[2]]).cumsum(0)
        )

    @property
    def cell_centers_x(self):
        """Return x-coordinates of the cell centers along the x-direction.

        For 1D, 2D and 3D meshes, this property returns a 1D vector
        containing the x-coordinate values of the cell centers along the
        x-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the x-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the x-coordinate values of the cell centers along the x-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_cells_x) torch.tensor of float
            A 1D array containing the x-coordinates of the cell centers along
            the x-direction.
        """
        nodes = self.nodes_x
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_y(self):
        """Return y-coordinates of the cell centers along the y-direction.

        For 2D and 3D meshes, this property returns a 1D vector
        containing the y-coordinate values of the cell centers along the
        y-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the y-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the y-coordinate values of the cell centers along the y-direction
        for the underlying tensor mesh .

        Returns
        -------
        (n_cells_y) torch.tensor of float or None
            A 1D array containing the y-coordinates of the cell centers along
            the y-direction. Returns *None* for 1D meshes.

        """
        if self.dim < 2:
            return None
        nodes = self.nodes_y
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_z(self):
        """Return z-coordinates of the cell centers along the z-direction.

        For 3D meshes, this property returns a 1D vector
        containing the z-coordinate values of the cell centers along the
        z-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the z-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the z-coordinate values of the cell centers along the z-direction
        for the underlying tensor mesh .

        Returns
        -------
        (n_cells_z) torch.tensor of float or None
            A 1D array containing the z-coordinates of the cell centers along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        if self.dim < 3:
            return None
        nodes = self.nodes_z
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._getTensorGrid("cell_centers")

    @property
    def nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._getTensorGrid("nodes")

    @property
    def boundary_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 1:
            return self.nodes_x[[0, -1]]
        return self.nodes[make_boundary_bool(self.shape_nodes)]

    @property
    def h_gridded(self):
        """Return dimensions of all mesh cells as staggered grid.

        This property returns a numpy array of shape (n_cells, dim)
        containing gridded x, (y and z) dimensions for all cells in the mesh.
        The first row corresponds to the bottom-front-leftmost cell.
        The cells are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_cells, dim) torch.tensor of float
            Dimensions of all mesh cells as staggered grid

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh_1D = TensorMesh([hx])
        >>> mesh_1D.h_gridded
        array([[1.],
               [1.],
               [1.],
               [1.],
               [1.]])

        The following is a 3D example.

        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh_3D = TensorMesh([hx, hy, hz])
        >>> mesh_3D.h_gridded
        array([[1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.]])

        """
        if self.dim == 1:
            return self.h[0][:, None]
        return ndgrid(*self.h, dtype=self.dtype, device=self.device)

    @property
    def faces_x(self):
        """Gridded x-face locations.

        This property returns a numpy array of shape (n_faces_x, dim)
        containing gridded locations for all x-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost x-face.
        The x-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_faces_x, dim) torch.tensor of float
            Gridded x-face locations
        """
        if self.n_faces_x == 0:
            return
        return self._getTensorGrid("faces_x")

    @property
    def faces_y(self):
        """Gridded y-face locations.

        This property returns a numpy array of shape (n_faces_y, dim)
        containing gridded locations for all y-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost y-face.
        The y-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        n_faces_y, dim) torch.tensor of float or None
            Gridded y-face locations for 2D and 3D mesh. Returns *None* for 1D meshes.
        """
        if self.n_faces_y == 0 or self.dim < 2:
            return
        return self._getTensorGrid("faces_y")

    @property
    def faces_z(self):
        """Gridded z-face locations.

        This property returns a numpy array of shape (n_faces_z, dim)
        containing gridded locations for all z-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost z-face.
        The z-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_faces_z, dim) torch.tensor of float or None
            Gridded z-face locations for 3D mesh. Returns *None* for 1D and 2D meshes.
        """
        if self.n_faces_z == 0 or self.dim < 3:
            return
        return self._getTensorGrid("faces_z")

    @property
    def faces(self):
        """Return all face indices stacked vertically (torch version)."""
        faces = []

        if self.faces_x is not None:
            faces.append(self.faces_x)
        else:
            faces.append(
                torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
            )

        if self.dim > 1 and self.faces_y is not None:
            faces.append(self.faces_y)

        if self.dim > 2 and self.faces_z is not None:
            faces.append(self.faces_z)

        return torch.cat(faces, dim=0)

    @property
    def boundary_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 1:
            return self.nodes_x[[0, -1]]
        if dim == 2:
            fx = ndgrid(self.nodes_x[[0, -1]], self.cell_centers_y)
            fy = ndgrid(self.cell_centers_x, self.nodes_y[[0, -1]])
            return torch.cat([fx, fy], dim=0)
        if dim == 3:
            fx = ndgrid(self.nodes_x[[0, -1]], self.cell_centers_y, self.cell_centers_z)
            fy = ndgrid(self.cell_centers_x, self.nodes_y[[0, -1]], self.cell_centers_z)
            fz = ndgrid(self.cell_centers_x, self.cell_centers_y, self.nodes_z[[0, -1]])
            return torch.cat([fx, fy, fz], dim=0)

    @property
    def boundary_face_outward_normals(self):
        dim = self.dim

        if dim == 1:
            return torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device)

        if dim == 2:
            nx = ndgrid(
                torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device),
                torch.zeros(self.shape_cells[1], dtype=self.dtype, device=self.device),
                dtype=self.dtype,
                device=self.device,
            )
            ny = ndgrid(
                torch.zeros(self.shape_cells[0], dtype=self.dtype, device=self.device),
                torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device),
                dtype=self.dtype,
                device=self.device,
            )
            return torch.cat([nx, ny], dim=0)

        if dim == 3:
            nx = ndgrid(
                torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device),
                torch.zeros(self.shape_cells[1], dtype=self.dtype, device=self.device),
                torch.zeros(self.shape_cells[2], dtype=self.dtype, device=self.device),
                dtype=self.dtype,
                device=self.device,
            )
            ny = ndgrid(
                torch.zeros(self.shape_cells[0], dtype=self.dtype, device=self.device),
                torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device),
                torch.zeros(self.shape_cells[2], dtype=self.dtype, device=self.device),
                dtype=self.dtype,
                device=self.device,
            )
            nz = ndgrid(
                torch.zeros(self.shape_cells[0], dtype=self.dtype, device=self.device),
                torch.zeros(self.shape_cells[1], dtype=self.dtype, device=self.device),
                torch.tensor([-1.0, 1.0], dtype=self.dtype, device=self.device),
                dtype=self.dtype,
                device=self.device,
            )
            return torch.cat([nx, ny, nz], dim=0)

    @property
    def edges_x(self):
        """Gridded x-edge locations.

        This property returns a numpy array of shape (n_edges_x, dim)
        containing gridded locations for all x-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost x-edge.
        The x-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_x, dim) torch.tensor of float or None
            Gridded x-edge locations. Returns *None* if `shape_edges_x[0]` is 0.
        """
        if self.n_edges_x == 0:
            return
        return self._getTensorGrid("edges_x")

    @property
    def edges_y(self):
        """Gridded y-edge locations.

        This property returns a numpy array of shape (n_edges_y, dim)
        containing gridded locations for all y-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost y-edge.
        The y-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_y, dim) torch.tensor of float
            Gridded y-edge locations. Returns *None* for 1D meshes.
        """
        if self.n_edges_y == 0 or self.dim < 2:
            return
        return self._getTensorGrid("edges_y")

    @property
    def edges_z(self):
        """Gridded z-edge locations.

        This property returns a numpy array of shape (n_edges_z, dim)
        containing gridded locations for all z-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost z-edge.
        The z-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_z, dim) torch.tensor of float
            Gridded z-edge locations. Returns *None* for 1D and 2D meshes.
        """
        if self.n_edges_z == 0 or self.dim < 3:
            return
        return self._getTensorGrid("edges_z")

    @property
    def edges(self):
        """Return all edge indices stacked vertically (PyTorch version)."""
        edges = []

        if self.edges_x is not None:
            edges.append(self.edges_x)
        else:
            edges.append(
                torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
            )

        if self.dim > 1 and self.edges_y is not None:
            edges.append(self.edges_y)

        if self.dim > 2 and self.edges_z is not None:
            edges.append(self.edges_z)

        return torch.cat(edges, dim=0)

    @property
    def boundary_edges(self):
        """Boundary edge locations (PyTorch version)."""

        if self.dim == 1:
            return None  # No boundary edges in 1D

        if self.dim == 2:
            ex = ndgrid(
                self.cell_centers_x,
                self.nodes_y[[0, -1]],
                dtype=self.dtype,
                device=self.device,
            )
            ey = ndgrid(
                self.nodes_x[[0, -1]],
                self.cell_centers_y,
                dtype=self.dtype,
                device=self.device,
            )
            return torch.cat([ex, ey], dim=0)

        if self.dim == 3:
            ex = self.edges_x[make_boundary_bool(self.shape_edges_x, bdir="yz")]
            ey = self.edges_y[make_boundary_bool(self.shape_edges_y, bdir="xz")]
            ez = self.edges_z[make_boundary_bool(self.shape_edges_z, bdir="xy")]
            return torch.cat([ex, ey, ez], dim=0)

    def _getTensorGrid(self, key):
        if getattr(self, "_" + key, None) is None:
            setattr(self, "_" + key, ndgrid(self.get_tensor(key)))
        return getattr(self, "_" + key)

    def get_tensor(self, key):
        """Return the base 1D arrays for a specified mesh tensor.

        The cell-centers, nodes, x-faces, z-edges, etc... of a tensor mesh
        can be constructed by applying tensor products to the set of base
        1D arrays; i.e. (vx, vy, vz). These 1D arrays define the gridded
        locations for the mesh tensor along each axis. For a given mesh tensor
        (i.e. cell centers, nodes, x/y/z faces or x/y/z edges),
        **get_tensor** returns a list containing the base 1D arrays.

        Parameters
        ----------
        key : str
            Specifies the tensor being returned. Please choose from::

                'CC', 'cell_centers' -> location of cell centers
                'N', 'nodes'         -> location of nodes
                'Fx', 'faces_x'      -> location of faces with an x normal
                'Fy', 'faces_y'      -> location of faces with an y normal
                'Fz', 'faces_z'      -> location of faces with an z normal
                'Ex', 'edges_x'      -> location of edges with an x tangent
                'Ey', 'edges_y'      -> location of edges with an y tangent
                'Ez', 'edges_z'      -> location of edges with an z tangent

        Returns
        -------
        (dim) list of 1D numpy.ndarray
            list of base 1D arrays for the tensor.

        """
        key = self._parse_location_type(key)

        if key == "faces_x":
            ten = [
                self.nodes_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "faces_y":
            ten = [
                self.cell_centers_x,
                self.nodes_y,
                self.cell_centers_z,
            ]
        elif key == "faces_z":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.nodes_z,
            ]
        elif key == "edges_x":
            ten = [self.cell_centers_x, self.nodes_y, self.nodes_z]
        elif key == "edges_y":
            ten = [self.nodes_x, self.cell_centers_y, self.nodes_z]
        elif key == "edges_z":
            ten = [self.nodes_x, self.nodes_y, self.cell_centers_z]
        elif key == "cell_centers":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "nodes":
            ten = [self.nodes_x, self.nodes_y, self.nodes_z]
        else:
            raise KeyError(r"Unrecognized key {key}")

        return [t for t in ten if t is not None]

    def is_inside(self, pts, location_type="nodes", tol=1e-10):
        """Determine which points lie within the mesh (PyTorch version)."""

        pts = as_array_n_by_dim(pts, self.dim)
        pts = torch.as_tensor(pts, dtype=self.dtype, device=self.device)

        tensors = self.get_tensor(location_type)

        if location_type[0].lower() == "n" and self._meshType == "CYL":
            # Add a node at the origin in R, and wrap-around in theta
            tensors[0] = torch.cat(
                [torch.tensor([0.0], dtype=self.dtype, device=self.device), tensors[0]]
            )
            tensors[1] = torch.cat(
                [
                    tensors[1],
                    torch.tensor(
                        [2.0 * torch.pi], dtype=self.dtype, device=self.device
                    ),
                ]
            )

        inside = torch.ones(pts.shape[0], dtype=torch.bool, device=self.device)

        for i, tensor in enumerate(tensors):
            diffs = torch.diff(tensor)
            TOL = diffs.min() * tol
            inside &= (pts[:, i] >= tensor.min() - TOL) & (
                pts[:, i] <= tensor.max() + TOL
            )

        return inside

    def _get_interpolation_matrix(
        self, loc, location_type="cell_centers", zeros_outside=False
    ):
        loc = as_array_n_by_dim(loc, self.dim)
        loc = torch.as_tensor(loc, dtype=self.dtype, device=self.device)

        if not zeros_outside:
            if not torch.all(self.is_inside(loc)):
                raise ValueError("Points outside of mesh")
        else:
            ind_zeros = ~self.is_inside(loc)
            loc = loc.clone()
            loc[ind_zeros] = torch.stack(
                [v.mean() for v in self.get_tensor("CC")]
            ).view(1, -1)

        location_type = self._parse_location_type(location_type)

        if location_type in [
            "faces_x",
            "faces_y",
            "faces_z",
            "edges_x",
            "edges_y",
            "edges_z",
        ]:
            ind = {"x": 0, "y": 1, "z": 2}[location_type[-1]]
            if self.dim <= ind:
                raise ValueError("mesh is not high enough dimension.")
            if "f" in location_type.lower():
                counts = (self.n_faces_x, self.n_faces_y, self.n_faces_z)[: self.dim]
            else:
                counts = (self.n_edges_x, self.n_edges_y, self.n_edges_z)[: self.dim]

            components = []
            for i, n in enumerate(counts):
                if i == ind:
                    interp = interpolation_matrix(
                        loc, *self.get_tensor(location_type)
                    )  # should return torch.sparse_coo_tensor
                else:
                    interp = torch.sparse_coo_tensor(
                        size=(loc.shape[0], n), dtype=self.dtype, device=self.device
                    )
                components.append(interp)

            # concatenate sparse tensors horizontally
            Q = torch.cat([comp.to_dense() for comp in components], dim=1).to_sparse()

        elif location_type in ["cell_centers", "nodes"]:
            Q = interpolation_matrix(loc, *self.get_tensor(location_type))

        elif location_type in ["cell_centers_x", "cell_centers_y", "cell_centers_z"]:
            Q_base = interpolation_matrix(loc, *self.get_tensor("CC"))
            Z = torch.zeros_like(Q_base)
            if location_type.endswith("x"):
                Q = torch.cat([Q_base, Z, Z], dim=1)
            elif location_type.endswith("y"):
                Q = torch.cat([Z, Q_base, Z], dim=1)
            elif location_type.endswith("z"):
                Q = torch.cat([Z, Z, Q_base], dim=1)
            Q = Q.to_sparse()

        else:
            raise NotImplementedError(
                f"get_interpolation_matrix: location_type={location_type} and mesh.dim={self.dim}"
            )

        if zeros_outside:
            # Remove contributions for points outside
            Q_dense = Q.to_dense()
            Q_dense[ind_zeros] = 0
            Q = Q_dense.to_sparse()

        return Q

    def get_interpolation_matrix(  # NOQA D102
        self, loc, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        return self._get_interpolation_matrix(loc, location_type, zeros_outside)
