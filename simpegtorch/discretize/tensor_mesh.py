import torch

from .base import BaseTensorMesh, BaseRectangularMesh

# from .operators import DiffOperators, InnerProducts
# from .mixins import InterfaceMixins, TensorMeshIO
from .utils import mkvc


class TensorMesh(
    # DiffOperators,
    # InnerProducts,
    BaseTensorMesh,
    BaseRectangularMesh,
    # TensorMeshIO,
    # InterfaceMixins,
):
    """
    Tensor mesh class.

    Tensor meshes are numerical grids whose cell centers, nodes, faces, edges, widths,
    volumes, etc... can be directly expressed as tensor products. The axes defining
    coordinates of the mesh are orthogonal. And cell properties along one axis do
    not vary with respect to the position along any other axis.

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
        corner. By default, the mesh is anchored such that its origin is at *[0, 0, 0]* .

        For each dimension (x, y or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate along
          each axis is the first node location ('0'), in the center ('C') or the last
          node location ('N') (see Examples).

    See Also
    --------
    utils.unpack_widths :
        The function used to expand a tuple to generate widths.

    Examples
    --------
    An example of a 2D tensor mesh is shown below. Here we use a list of tuple to
    define the discretization along the x-axis and a numpy array to define the
    discretization along the y-axis. We also use a string argument to center the
    x-axis about x = 0 and set the top of the mesh to y = 0.

    >>> from discretize import TensorMesh
    >>> import matplotlib.pyplot as plt

    >>> ncx = 10      # number of core mesh cells in x
    >>> dx = 5        # base cell width x
    >>> npad_x = 3    # number of padding cells in x
    >>> exp_x = 1.25  # expansion rate of padding cells in x
    >>> ncy = 24      # total number of mesh cells in y
    >>> dy = 5        # base cell width y

    >>> hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
    >>> hy = dy * np.ones(ncy)
    >>> mesh = TensorMesh([hx, hy], origin='CN')

    >>> fig = plt.figure(figsize=(5,5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_grid(ax=ax)
    >>> plt.show()
    """

    _meshType = "TENSOR"

    def __repr__(self):
        """Plain text representation."""
        fmt = "\n  {}: {:,} cells\n\n".format(type(self).__name__, self.n_cells)
        fmt += 22 * " " + "MESH EXTENT" + 13 * " " + "CELL WIDTH      FACTOR\n"
        fmt += "  dir    nC        min           max         min       max "
        fmt += "     max\n  ---   ---  " + 27 * "-" + "  " + 18 * "-" + "  ------\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "   {}".format(name)
            fmt += " {:6}".format(iattr["nC"])
            for p in ["min", "max"]:
                fmt += " {:13,.2f}".format(iattr[p])
            for p in ["h_min", "h_max"]:
                fmt += " {:9,.2f}".format(iattr[p])
            fmt += "{:8,.2f}".format(iattr["max_fact"])
            fmt += "\n"  # End row

        fmt += "\n"
        return fmt

    def _repr_html_(self):
        """HTML representation."""
        style = " style='padding: 5px 20px 5px 20px;'"

        fmt = "<table>\n"
        fmt += "  <tr>\n"
        fmt += "    <td style='font-weight: bold; font-size: 1.2em; text-align"
        fmt += ": center;' colspan='3'>{}</td>\n".format(type(self).__name__)
        fmt += "    <td style='font-size: 1.2em; text-align: center;'"
        fmt += "colspan='4'>{:,} cells</td>\n".format(self.n_cells)
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th colspan='2'" + style + ">MESH EXTENT</th>\n"
        fmt += "    <th colspan='2'" + style + ">CELL WIDTH</th>\n"
        fmt += "    <th" + style + ">FACTOR</th>\n"
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th" + style + ">dir</th>\n"
        fmt += "    <th" + style + ">nC</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "  </tr>\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "  <tr>\n"  # Start row
            fmt += "    <td" + style + ">{}</td>\n".format(name)
            fmt += "    <td" + style + ">{}</td>\n".format(iattr["nC"])
            for p in ["min", "max", "h_min", "h_max", "max_fact"]:
                fmt += "    <td" + style + ">{:,.2f}</td>\n".format(iattr[p])
            fmt += "  </tr>\n"  # End row

        fmt += "</table>\n"
        return fmt

    def __iter__(self):
        """Iterate over the cells."""
        iterator = (self[i] for i in range(len(self)))
        return iterator

    # --------------- Geometries ---------------------
    @property
    def cell_volumes(self):
        # Only compute if not already cached
        if getattr(self, "_cell_volumes", None) is None:
            vh = self.h
            if self.dim == 1:
                self._cell_volumes = mkvc(vh[0])
            elif self.dim == 2:
                # Outer product for 2D: (nx, 1) * (1, ny) → (nx, ny)
                vol = torch.outer(vh[0], vh[1])
                self._cell_volumes = mkvc(vol)
            elif self.dim == 3:
                # Outer product for 3D: (nx * ny, 1) * (1, nz)
                vol_xy = mkvc(torch.outer(vh[0], vh[1]))  # (nx*ny,)
                vol = torch.outer(vol_xy, vh[2])  # (nx*ny, nz)
                self._cell_volumes = mkvc(vol)
        return self._cell_volumes

    @property
    def face_x_areas(self):
        """Return the areas of the x-faces in a PyTorch-compatible way."""
        if getattr(self, "_face_x_areas", None) is None:
            vh = self.h  # list of 1D torch tensors
            n = (
                self.shape_cells
            )  # number of cells in each direction (tuple or list of ints)

            if self.dim == 1:
                areaFx = torch.ones(n[0] + 1, dtype=self.dtype, device=self.device)
            elif self.dim == 2:
                areaFx = torch.outer(
                    torch.ones(n[0] + 1, dtype=self.dtype, device=self.device), vh[1]
                )
            elif self.dim == 3:
                yz = mkvc(torch.outer(vh[1], vh[2]))
                areaFx = torch.outer(
                    torch.ones(n[0] + 1, dtype=self.dtype, device=self.device), yz
                )

            self._face_x_areas = mkvc(areaFx)

        return self._face_x_areas

    @property
    def face_y_areas(self):
        """Return the areas of the y-faces in a PyTorch-compatible way."""
        if getattr(self, "_face_y_areas", None) is None:
            vh = self.h  # list of 1D torch tensors
            n = self.shape_cells  # number of cells in each direction

            if self.dim == 1:
                raise Exception("1D meshes do not have y-faces")
            elif self.dim == 2:
                areaFy = torch.outer(
                    vh[0], torch.ones(n[1] + 1, dtype=self.dtype, device=self.device)
                )
            elif self.dim == 3:
                yz = mkvc(
                    torch.outer(
                        torch.ones(n[1] + 1, dtype=self.dtype, device=self.device),
                        vh[2],
                    )
                )
                areaFy = torch.outer(vh[0], yz)

            self._face_y_areas = mkvc(areaFy)

        return self._face_y_areas

    @property
    def face_z_areas(self):
        """Return the areas of the z-faces (PyTorch version)."""
        if getattr(self, "_face_z_areas", None) is None:
            vh = self.h  # list of 1D torch tensors: [hx, hy, hz]
            n = self.shape_cells  # list of number of cells in each direction

            if self.dim in [1, 2]:
                raise Exception(f"{self.dim}D meshes do not have z-Faces")
            elif self.dim == 3:
                yz = mkvc(
                    torch.outer(
                        vh[1],
                        torch.ones(n[2] + 1, dtype=self.dtype, device=self.device),
                    )
                )
                areaFz = torch.outer(vh[0], yz)

            self._face_z_areas = mkvc(areaFz)

        return self._face_z_areas

    @property
    def face_areas(self):  # NOQA D102
        # Return total face areas based on dimension
        if self.dim == 1:
            return self.face_x_areas
        elif self.dim == 2:
            return torch.cat([self.face_x_areas, self.face_y_areas])
        elif self.dim == 3:
            return torch.cat([self.face_x_areas, self.face_y_areas, self.face_z_areas])

    @property
    def edge_x_lengths(self):
        """Return the x-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the x-axis.

        Returns
        -------
        (n_edges_x) torch.Tensor
            X-edge lengths
        """
        if getattr(self, "_edge_x_lengths", None) is None:
            vh = self.h  # assumes list of 1D torch tensors
            n = self.shape_cells  # assumes list or tuple of ints

            if self.dim == 1:
                edgeEx = vh[0]
            elif self.dim == 2:
                edgeEx = torch.outer(
                    vh[0], torch.ones(n[1] + 1, dtype=self.dtype, device=self.device)
                )
            elif self.dim == 3:
                edgeEx = torch.outer(
                    vh[0],
                    mkvc(
                        torch.outer(
                            torch.ones(n[1] + 1, dtype=self.dtype, device=self.device),
                            torch.ones(n[2] + 1, dtype=self.dtype, device=self.device),
                        )
                    ),
                )

            self._edge_x_lengths = mkvc(edgeEx)
        return self._edge_x_lengths

    @property
    def edge_y_lengths(self):
        """Return the y-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the y-axis.

        Returns
        -------
        (n_edges_y) torch.Tensor
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* Raises an Exception (no y-edges)
            - *2D:* Returns y-edge lengths
            - *3D:* Returns y-edge lengths
        """
        if getattr(self, "_edge_y_lengths", None) is None:
            vh = self.h  # list of 1D torch tensors
            n = self.shape_cells  # list or tuple of ints

            if self.dim == 1:
                raise Exception("1D meshes do not have y-edges")
            elif self.dim == 2:
                edgeEy = torch.outer(
                    torch.ones(n[0] + 1, dtype=self.dtype, device=self.device), vh[1]
                )
            elif self.dim == 3:
                inner = torch.outer(
                    vh[1], torch.ones(n[2] + 1, dtype=self.dtype, device=self.device)
                )
                edgeEy = torch.outer(
                    torch.ones(n[0] + 1, dtype=self.dtype, device=self.device),
                    mkvc(inner),
                )

            self._edge_y_lengths = mkvc(edgeEy)
        return self._edge_y_lengths

    @property
    def edge_z_lengths(self):
        """Return the z-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the z-axis.

        Returns
        -------
        (n_edges_z) torch.Tensor
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* N/A since 1D meshes do not have z-edges
            - *2D:* N/A since 2D meshes do not have z-edges
            - *3D:* Returns z-edge lengths
        """
        if getattr(self, "_edge_z_lengths", None) is None:
            vh = self.h  # list of 1D torch tensors
            n = self.shape_cells  # list or tuple of ints

            if self.dim == 1 or self.dim == 2:
                raise Exception(f"{self.dim}D meshes do not have z-edges")
            elif self.dim == 3:
                inner = torch.outer(
                    torch.ones(n[1] + 1, dtype=self.dtype, device=self.device), vh[2]
                )
                edgeEz = torch.outer(
                    torch.ones(n[0] + 1, dtype=self.dtype, device=self.device),
                    mkvc(inner),
                )

            self._edge_z_lengths = mkvc(edgeEz)
        return self._edge_z_lengths

    @property
    def edge_lengths(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 1:
            return self.edge_x_lengths
        elif self.dim == 2:
            return torch.cat([self.edge_x_lengths, self.edge_y_lengths])
        elif self.dim == 3:
            return torch.cat(
                [self.edge_x_lengths, self.edge_y_lengths, self.edge_z_lengths]
            )
        raise ValueError(f"Unsupported mesh dimension: {self.dim}")

    @property
    def face_boundary_indices(self):
        """Return the indices of the x, (y and z) boundary faces.

        For x, (y and z) faces, this property returns the indices of the faces
        on the boundaries. That is, the property returns the indices of the x-faces
        that lie on the x-boundary; likewise for y and z. Note that each
        Cartesian direction will have both a lower and upper boundary,
        and the property will return the indices corresponding to the lower
        and upper boundaries separately.

        E.g. for a 2D domain, there are 2 x-boundaries and 2 y-boundaries (4 in total).
        In this case, the return is a list of length 4 organized
        [ind_Bx1, ind_Bx2, ind_By1, ind_By2]::

                       By2
                + ------------- +
                |               |
                |               |
            Bx1 |               | Bx2
                |               |
                |               |
                + ------------- +
                       By1


        Returns
        -------
        (dim * 2) list of numpy.ndarray of bool
            The length of list returned depends on the dimension of the mesh.
            And the length of each array containing the indices depends on the
            number of faces in each direction. For 1D, 2D and 3D
            tensor meshes, the returns take the following form:

            - *1D:* returns [ind_Bx1, ind_Bx2]
            - *2D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2]
            - *3D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2, ind_Bz1, ind_Bz2]

        Examples
        --------
        Here, we construct a 4 by 3 cell 2D tensor mesh and return the indices
        of the x and y-boundary faces. In this case there are 3 x-faces on each
        x-boundary, and there are 4 y-faces on each y-boundary.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> hx = [1, 1, 1, 1]
        >>> hy = [2, 2, 2]
        >>> mesh = TensorMesh([hx, hy])
        >>> ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.face_boundary_indices

        >>> ax = plt.subplot(111)
        >>> mesh.plot_grid(ax=ax)
        >>> ax.scatter(*mesh.faces_x[ind_Bx1].T)
        >>> plt.show()
        """
        if self.dim == 1:
            indxd = self.gridFx == min(self.gridFx)
            indxu = self.gridFx == max(self.gridFx)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            indzd = self.gridFz[:, 2] == min(self.gridFz[:, 2])
            indzu = self.gridFz[:, 2] == max(self.gridFz[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    @property
    def cell_bounds(self):
        """The bounds of each cell.

        Return a 2D array with the coordinates that define the bounds of each
        cell in the mesh. Each row of the array contains the bounds for
        a particular cell in the following order: ``x1``, ``x2``, ``y1``,
        ``y2``, ``z1``, ``z2``, where ``x1 < x2``, ``y1 < y2`` and ``z1 < z2``.
        """
        nodes = self.nodes.view(*self.shape_nodes, -1).permute(
            *range(self.dim), self.dim
        )

        if self.dim == 1:
            min_nodes = nodes[:-1]
            max_nodes = nodes[1:]
        elif self.dim == 2:
            min_nodes = nodes[:-1, :-1].reshape(self.n_cells, -1)
            max_nodes = nodes[1:, 1:].reshape(self.n_cells, -1)
        else:
            min_nodes = nodes[:-1, :-1, :-1].reshape(self.n_cells, -1)
            max_nodes = nodes[1:, 1:, 1:].reshape(self.n_cells, -1)

        cell_bounds = torch.stack((min_nodes, max_nodes), dim=-1)
        cell_bounds = cell_bounds.view(self.n_cells, -1)
        return cell_bounds

    @property
    def cell_nodes(self):
        """The index of all nodes for each cell (torch-compatible with Fortran order)."""
        shape = self.shape_nodes
        nodes_indices = (
            torch.arange(self.n_nodes, dtype=torch.long)
            .reshape(*shape)
            .permute(*reversed(range(self.dim)))
        )

        if self.dim == 1:
            cell_nodes = torch.stack(
                [mkvc(nodes_indices[:-1]), mkvc(nodes_indices[1:])], dim=-1
            )

        elif self.dim == 2:
            cell_nodes = torch.stack(
                [
                    mkvc(nodes_indices[:-1, :-1]),
                    mkvc(nodes_indices[1:, :-1]),
                    mkvc(nodes_indices[:-1, 1:]),
                    mkvc(nodes_indices[1:, 1:]),
                ],
                dim=-1,
            )

        elif self.dim == 3:
            cell_nodes = torch.stack(
                [
                    mkvc(nodes_indices[:-1, :-1, :-1]),
                    mkvc(nodes_indices[1:, :-1, :-1]),
                    mkvc(nodes_indices[:-1, 1:, :-1]),
                    mkvc(nodes_indices[1:, 1:, :-1]),
                    mkvc(nodes_indices[:-1, :-1, 1:]),
                    mkvc(nodes_indices[1:, :-1, 1:]),
                    mkvc(nodes_indices[:-1, 1:, 1:]),
                    mkvc(nodes_indices[1:, 1:, 1:]),
                ],
                dim=-1,
            )

        else:
            raise ValueError("Unsupported mesh dimension")

        return cell_nodes

    @property
    def cell_boundary_indices(self):
        """Return the indices of the x, (y and z) boundary cells.

        This property returns the indices of the cells on the x, (y and z)
        boundaries, respectively. Note that each axis direction will
        have both a lower and upper boundary. The property will
        return the indices corresponding to the lower and upper
        boundaries separately.

        E.g. for a 2D domain, there are 2 x-boundaries and 2 y-boundaries (4 in total).
        In this case, the return is a list of length 4 organized
        [ind_Bx1, ind_Bx2, ind_By1, ind_By2]::

                       By2
                + ------------- +
                |               |
                |               |
            Bx1 |               | Bx2
                |               |
                |               |
                + ------------- +
                       By1


        Returns
        -------
        (2 * dim) list of numpy.ndarray of bool
            The length of list returned depends on the dimension of the mesh (= 2 x dim).
            And the length of each array containing the indices is equal to
            the number of cells in the mesh. For 1D, 2D and 3D
            tensor meshes, the returns take the following form:

            - *1D:* returns [ind_Bx1, ind_Bx2]
            - *2D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2]
            - *3D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2, ind_Bz1, ind_Bz2]

        Examples
        --------
        Here, we construct a 4 by 3 cell 2D tensor mesh and return the indices
        of the x and y-boundary cells. In this case there are 3 cells touching
        each x-boundary, and there are 4 cells touching each y-boundary.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> hx = [1, 1, 1, 1]
        >>> hy = [2, 2, 2]
        >>> mesh = TensorMesh([hx, hy])
        >>> ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.cell_boundary_indices

        >>> ax = plt.subplot(111)
        >>> mesh.plot_grid(ax=ax)
        >>> ax.scatter(*mesh.cell_centers[ind_Bx1].T)
        >>> plt.show()
        """
        if self.dim == 1:
            indxd = self.gridCC == min(self.gridCC)
            indxu = self.gridCC == max(self.gridCC)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            indzd = self.gridCC[:, 2] == min(self.gridCC[:, 2])
            indzu = self.gridCC[:, 2] == max(self.gridCC[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    def _repr_attributes(self):
        """Represent attributes of the mesh."""
        attrs = {}
        attrs["names"] = ["x", "y", "z"][: self.dim]

        for i in range(self.dim):
            name = attrs["names"][i]
            attrs[name] = {}

            # Get min/max node
            n_vector = getattr(self, "nodes_" + name)
            attrs[name]["min"] = torch.min(n_vector)
            attrs[name]["max"] = torch.max(n_vector)

            # Get min/max cell width
            h_vector = self.h[i]
            attrs[name]["h_min"] = torch.min(h_vector)
            attrs[name]["h_max"] = torch.max(h_vector)

            # Max stretching factor
            if len(h_vector) < 2:
                attrs[name]["max_fact"] = 1.0
            else:
                r1 = h_vector[:-1] / h_vector[1:]
                r2 = h_vector[1:] / h_vector[:-1]
                attrs[name]["max_fact"] = torch.max(torch.cat([r1, r2]))

            # Number of cells
            attrs[name]["nC"] = self.shape_cells[i]

        return attrs

    def _slice_to_index(index_slice, end):
        """Generate indices from a slice.

        Parameters
        ----------
        index_slice : slice
            Slice for cell indices along a single dimension
        end : int
            End of the slice. Will use this value as the stop in case the
            `index_slice.stop` is None.

        Returns
        -------
        Generator
        """
        if (start := index_slice.start) is None:
            start = 0
        if (stop := index_slice.stop) is None:
            stop = end
        if (step := index_slice.step) is None:
            step = 1
        if start < 0:
            start += end
        if stop < 0:
            stop += end
        if step < 0:
            return reversed(range(start, stop, abs(step)))
        return range(start, stop, step)


def _slice_to_index(index_slice, end):
    """Generate indices from a slice.

    Parameters
    ----------
    index_slice : slice
        Slice for cell indices along a single dimension
    end : int
        End of the slice. Will use this value as the stop in case the
        `index_slice.stop` is None.

    Returns
    -------
    Generator
    """
    if (start := index_slice.start) is None:
        start = 0
    if (stop := index_slice.stop) is None:
        stop = end
    if (step := index_slice.step) is None:
        step = 1
    if start < 0:
        start += end
    if stop < 0:
        stop += end
    if step < 0:
        return reversed(range(start, stop, abs(step)))
    return range(start, stop, step)
