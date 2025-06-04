"""Differential operators for tensor meshes in PyTorch.

This module provides differential and averaging operators for finite volume
computations on tensor meshes, implemented using PyTorch for automatic
differentiation and GPU acceleration.
"""

import torch
from simpegtorch.discretize.base import BaseMesh
from simpegtorch.discretize.utils import (
    sdiag,
    kron,
    kron3,
    speye,
    spzeros,
    ddx,
    av,
    av_extrap,
    make_boundary_bool,
    torch_blockdiag,
    cross2d,
)


def _validate_BC(bc):
    """Check if boundary condition 'bc' is valid.

    Each bc must be either 'dirichlet' or 'neumann'
    """
    if isinstance(bc, str):
        bc = [bc, bc]
    if not isinstance(bc, list):
        raise TypeError("bc must be a single string or list of strings")
    if not len(bc) == 2:
        raise TypeError("bc list must have two elements, one for each side")

    for bc_i in bc:
        if not isinstance(bc_i, str):
            raise TypeError("each bc must be a string")
        if bc_i not in ["dirichlet", "neumann"]:
            raise ValueError("each bc must be either, 'dirichlet' or 'neumann'")
    return bc


def _ddxCellGrad(n, bc, device=None, dtype=None):
    """Create 1D derivative operator from cell-centers to nodes.

    This means we go from n to n+1

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

            u_g       u_1      u_2
             *    |    *   |    *     ...
                  ^
                  0

        u_g = - u_1
        grad = 2*u1/dx
        negative on the other side.

    For Cell-Centered **Neumann**, use a ghost point::

        (u_1 - u_g)/hf = 0

            u_g       u_1      u_2
             *    |    *   |    *     ...

        u_g = u_1
        grad = 0;  put a zero in.
    """
    bc = _validate_BC(bc)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64

    # Create sparse difference operator
    indices = torch.zeros((2, (n + 1) * 2), dtype=torch.long, device=device)
    values = torch.zeros((n + 1) * 2, dtype=dtype, device=device)

    # Fill indices and values for the sparse matrix
    row_idx = 0
    for i in range(n + 1):
        if i > 0:
            indices[0, row_idx] = i
            indices[1, row_idx] = i - 1
            values[row_idx] = -1.0
            row_idx += 1
        if i < n:
            indices[0, row_idx] = i
            indices[1, row_idx] = i
            values[row_idx] = 1.0
            row_idx += 1

    # Trim to actual size
    indices = indices[:, :row_idx]
    values = values[:row_idx]

    D = torch.sparse_coo_tensor(indices, values, (n + 1, n), device=device, dtype=dtype)
    D = D.coalesce()

    # Set boundary conditions by creating new sparse tensor with modified values
    D_dense = D.to_dense()
    if bc[0] == "dirichlet":
        D_dense[0, 0] = 2
    elif bc[0] == "neumann":
        D_dense[0, 0] = 0
    # Set the second side
    if bc[1] == "dirichlet":
        D_dense[-1, -1] = -2
    elif bc[1] == "neumann":
        D_dense[-1, -1] = 0

    return D_dense.to_sparse()


def _ddxCellGradBC(n, bc, device=None, dtype=None):
    """Create 1D derivative operator from cell-centers to nodes.

    This means we go from n to n+1.

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

         u_g       u_1      u_2
          *    |    *   |    *     ...
               ^
              u_b

    We know the value at the boundary (u_b)::

        (u_g+u_1)/2 = u_b               (the average)
        u_g = 2*u_b - u_1

        So plug in to gradient:

        (u_1 - (2*u_b - u_1))/hf = grad
        2*(u_1-u_b)/hf = grad

    Separate, because BC are known (and can move to RHS later)::

        ( 2/hf )*u_1 + ( -2/hf )*u_b = grad

                       (   ^   ) JUST RETURN THIS
    """

    bc = _validate_BC(bc)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64

    # Create the sparse matrix using COO format
    indices = []
    values = []

    if bc[0] == "dirichlet":
        indices.append([0, 0])
        values.append(-2.0)
    elif bc[0] == "neumann":
        indices.append([0, 0])
        values.append(0.0)

    if bc[1] == "dirichlet":
        indices.append([n, 1])
        values.append(2.0)
    elif bc[1] == "neumann":
        indices.append([n, 1])
        values.append(0.0)

    if indices:
        indices = torch.tensor(indices, dtype=torch.long, device=device).T
        values = torch.tensor(values, dtype=dtype, device=device)
        D = torch.sparse_coo_tensor(
            indices, values, (n + 1, 2), device=device, dtype=dtype
        )
        D = D.coalesce()
    else:
        D = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            (n + 1, 2),
            device=device,
            dtype=dtype,
        )

    return D


class DiffOperators(BaseMesh):
    """Class used for creating differential and averaging operators.

    ``DiffOperators`` is a class for managing the construction of
    differential and averaging operators at the highest level.
    The ``DiffOperator`` class is inherited by every ``discretize``
    mesh class. In practice, differential and averaging operators are
    not constructed by creating instances of ``DiffOperators``.
    Instead, the operators are constructed (and sometimes stored)
    when called as a property of the mesh.
    """

    _aliases = {
        **BaseMesh._aliases,
        "aveFx2CC": "average_face_x_to_cell",
        "aveFy2CC": "average_face_y_to_cell",
        "aveFz2CC": "average_face_z_to_cell",
        "aveEx2CC": "average_edge_x_to_cell",
        "aveEy2CC": "average_edge_y_to_cell",
        "aveEz2CC": "average_edge_z_to_cell",
        "aveCC2F": "average_cell_to_face",
        "nFx": "n_faces_x",
        "nFy": "n_faces_y",
        "nFz": "n_faces_z",
        "nEx": "n_edges_x",
        "nEy": "n_edges_y",
        "nEz": "n_edges_z",
    }

    ###########################################################################
    #                                                                         #
    #                             Face Divergence                             #
    #                                                                         #
    ###########################################################################

    @property
    def _face_x_divergence_stencil(self):
        """Stencil for face divergence operator in the x-direction (x-faces to cell centers)."""
        if self.dim == 1:
            Dx = ddx(self.shape_cells[0], dtype=self.dtype, device=self.device)
        elif self.dim == 2:
            Dx = kron(
                speye(self.shape_cells[1], dtype=self.dtype, device=self.device),
                ddx(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        elif self.dim == 3:
            Dx = kron3(
                speye(self.shape_cells[2]),
                speye(self.shape_cells[1]),
                ddx(self.shape_cells[0]),
            )
        return Dx

    @property
    def _face_y_divergence_stencil(self):
        """Stencil for face divergence operator in the y-direction (y-faces to cell centers)."""
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Dy = kron(
                ddx(self.shape_cells[1], dtype=self.dtype, device=self.device),
                speye(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        elif self.dim == 3:
            Dy = kron3(
                speye(self.shape_cells[2], dtype=self.dtype, device=self.device),
                ddx(self.shape_cells[1], dtype=self.dtype, device=self.device),
                speye(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        return Dy

    @property
    def _face_z_divergence_stencil(self):
        """Stencil for face divergence operator in the z-direction (z-faces to cell centers)."""
        if self.dim == 1 or self.dim == 2:
            return None
        elif self.dim == 3:
            Dz = kron3(
                ddx(self.shape_cells[2], dtype=self.dtype, device=self.device),
                speye(self.shape_cells[1], dtype=self.dtype, device=self.device),
                speye(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        return Dz

    @property
    def _face_divergence_stencil(self):
        """Stencil for face divergence operator (all faces to cell centers)."""
        if self.dim == 1:
            D = self._face_x_divergence_stencil
        elif self.dim == 2:
            D = torch.hstack(
                (self._face_x_divergence_stencil, self._face_y_divergence_stencil)
            )
        elif self.dim == 3:
            D = torch.hstack(
                (
                    self._face_x_divergence_stencil,
                    self._face_y_divergence_stencil,
                    self._face_z_divergence_stencil,
                )
            )
        return D

    @property
    def face_divergence(self):
        """Face divergence operator (faces to cell centers).

        Returns
        -------
        torch.sparse.Tensor
            The divergence operator matrix that maps from faces to cell centers
        """
        if getattr(self, "_face_divergence", None) is None:
            # Get the stencil of +1, -1's
            D = self._face_divergence_stencil
            # Compute areas of cell faces & volumes
            S = self.face_areas
            V = self.cell_volumes
            self._face_divergence = (
                sdiag(1 / V, device=self.device, dtype=self.dtype)
                @ D
                @ sdiag(S, device=self.device, dtype=self.dtype)
            )
        return self._face_divergence

    @property
    def face_x_divergence(self):
        """X-derivative operator (x-faces to cell-centres).

        This property constructs a 2nd order x-derivative operator which maps
        from x-faces to cell centers. The operator is a sparse matrix
        that can be applied as a matrix-vector product to a discrete scalar
        quantity that lives on x-faces.

        Returns
        -------
        torch.sparse.Tensor
            The numerical x-derivative operator from x-faces to cell centers
        """
        S = self.reshape(self.face_areas, "F", "Fx", "V")
        V = self.cell_volumes
        return (
            sdiag(1 / V, device=self.device, dtype=self.dtype)
            @ self._face_x_divergence_stencil
            @ sdiag(S, device=self.device, dtype=self.dtype)
        )

    @property
    def face_y_divergence(self):
        """Y-derivative operator (y-faces to cell-centres).

        This property constructs a 2nd order y-derivative operator which maps
        from y-faces to cell centers. The operator is a sparse matrix
        that can be applied as a matrix-vector product to a discrete scalar
        quantity that lives on y-faces.

        Returns
        -------
        torch.sparse.Tensor
            The numerical y-derivative operator from y-faces to cell centers
        """
        if self.dim < 2:
            return None
        S = self.reshape(self.face_areas, "F", "Fy", "V")
        # Compute areas of cell faces & volumes
        V = self.cell_volumes
        return (
            sdiag(1 / V, device=self.device, dtype=self.dtype)
            @ self._face_y_divergence_stencil
            @ sdiag(S, device=self.device, dtype=self.dtype)
        )

    @property
    def face_z_divergence(self):
        """Z-derivative operator (z-faces to cell-centres).

        This property constructs a 2nd order z-derivative operator which maps
        from z-faces to cell centers. The operator is a sparse matrix
        that can be applied as a matrix-vector product to a discrete scalar
        quantity that lives on z-faces.

        Returns
        -------
        torch.sparse.Tensor
            The numerical z-derivative operator from z-faces to cell centers
        """
        if self.dim < 3:
            return None
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fz", "V")
        V = self.cell_volumes
        return (
            sdiag(1 / V, device=self.device, dtype=self.dtype)
            @ self._face_z_divergence_stencil
            @ sdiag(S, device=self.device, dtype=self.dtype)
        )

    ###########################################################################
    #                                                                         #
    #                            Nodal Gradient                              #
    #                                                                         #
    ###########################################################################

    @property
    def _nodal_gradient_x_stencil(self):
        """Stencil for nodal gradient operator in the x-direction."""
        if self.dim == 1:
            Gx = ddx(self.shape_cells[0], dtype=self.dtype, device=self.device)
        elif self.dim == 2:
            Gx = kron(
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
                ddx(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        elif self.dim == 3:
            Gx = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
                ddx(self.shape_cells[0], dtype=self.dtype, device=self.device),
            )
        return Gx

    @property
    def _nodal_gradient_y_stencil(self):
        """Stencil for nodal gradient operator in the y-direction."""
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Gy = kron(
                ddx(self.shape_cells[1], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
            )
        elif self.dim == 3:
            Gy = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                ddx(self.shape_cells[1], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
            )
        return Gy

    @property
    def _nodal_gradient_z_stencil(self):
        """Stencil for nodal gradient operator in the z-direction."""
        if self.dim == 1 or self.dim == 2:
            return None
        else:
            Gz = kron3(
                ddx(self.shape_cells[2], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
            )
        return Gz

    @property
    def _nodal_gradient_stencil(self):
        """Stencil for nodal gradient operator."""
        if self.dim == 1:
            G = self._nodal_gradient_x_stencil
        elif self.dim == 2:
            G = torch.vstack(
                (self._nodal_gradient_x_stencil, self._nodal_gradient_y_stencil),
            )
        elif self.dim == 3:
            G = torch.vstack(
                (
                    self._nodal_gradient_x_stencil,
                    self._nodal_gradient_y_stencil,
                    self._nodal_gradient_z_stencil,
                ),
            )
        return G

    @property
    def nodal_gradient(self):
        """Nodal gradient operator (nodes to edges).

        Returns
        -------
        torch.sparse.Tensor
            The gradient operator matrix that maps from nodes to edges
        """
        if getattr(self, "_nodal_gradient", None) is None:
            G = self._nodal_gradient_stencil
            L = self.edge_lengths
            self._nodal_gradient = (
                sdiag(1 / L, dtype=self.dtype, device=self.device) @ G
            )
        return self._nodal_gradient

    ###########################################################################
    #                                                                         #
    #                            Nodal Laplacian                             #
    #                                                                         #
    ###########################################################################

    @property
    def _nodal_laplacian_x_stencil(self):
        """Stencil for nodal laplacian operator in the x-direction."""
        Dx = ddx(self.shape_cells[0], dtype=self.dtype, device=self.device)
        Lx = -Dx.T * Dx

        if self.dim == 2:
            Lx = kron(speye(self.shape_nodes[1]), Lx)
        elif self.dim == 3:
            Lx = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
                Lx,
            )
        return Lx

    @property
    def _nodal_laplacian_y_stencil(self):
        """Stencil for nodal laplacian operator in the y-direction."""
        if self.dim == 1:
            return None

        Dy = ddx(self.shape_cells[1], dtype=self.dtype, device=self.device)
        Ly = -Dy.T * Dy

        if self.dim == 2:
            Ly = kron(Ly, speye(self.shape_nodes[0]))
        elif self.dim == 3:
            Ly = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                Ly,
                speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
            )
        return Ly

    @property
    def _nodal_laplacian_z_stencil(self):
        """Stencil for nodal laplacian operator in the z-direction."""
        if self.dim == 1 or self.dim == 2:
            return None

        Dz = ddx(self.shape_cells[2], dtype=self.dtype, device=self.device)
        Lz = -Dz.T * Dz
        return kron3(
            Lz,
            speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
            speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
        )

    @property
    def _nodal_laplacian_x(self):
        """Nodal laplacian operator in the x-direction."""
        Hx = sdiag(1.0 / self.h[0], dtype=self.dtype, device=self.device)
        if self.dim == 2:
            Hx = kron(
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device), Hx
            )
        elif self.dim == 3:
            Hx = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
                Hx,
            )
        return Hx.T * self._nodal_gradient_x_stencil * Hx

    @property
    def _nodal_laplacian_y(self):
        """Nodal laplacian operator in the y-direction."""
        Hy = sdiag(1.0 / self.h[1], dtype=self.dtype, device=self.device)
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Hy = kron(
                Hy, speye(self.shape_nodes[0], dtype=self.dtype, device=self.device)
            )
        elif self.dim == 3:
            Hy = kron3(
                speye(self.shape_nodes[2], dtype=self.dtype, device=self.device),
                Hy,
                speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
            )
        return Hy.T * self._nodal_gradient_y_stencil * Hy

    @property
    def _nodal_laplacian_z(self):
        """Nodal laplacian operator in the z-direction."""
        if self.dim == 1 or self.dim == 2:
            return None
        Hz = sdiag(1.0 / self.h[2], dtype=self.dtype, device=self.device)
        Hz = kron3(
            Hz,
            speye(self.shape_nodes[1], dtype=self.dtype, device=self.device),
            speye(self.shape_nodes[0], dtype=self.dtype, device=self.device),
        )
        return Hz.T * self._nodal_laplacian_z_stencil * Hz

    @property
    def nodal_laplacian(self):
        """Nodal laplacian operator (nodes to nodes).

        Returns
        -------
        torch.sparse.Tensor
            The laplacian operator matrix that maps from nodes to nodes
        """
        if getattr(self, "_nodal_laplacian", None) is None:
            # Compute divergence operator on faces
            if self.dim == 1:
                self._nodal_laplacian = self._nodal_laplacian_x
            elif self.dim == 2:
                self._nodal_laplacian = (
                    self._nodal_laplacian_x + self._nodal_laplacian_y
                )
            elif self.dim == 3:
                self._nodal_laplacian = (
                    self._nodal_laplacian_x
                    + self._nodal_laplacian_y
                    + self._nodal_laplacian_z
                )
        return self._nodal_laplacian

    ###########################################################################
    #                                                                         #
    #                              Weak Forms                                #
    #                                                                         #
    ###########################################################################

    def edge_divergence_weak_form_robin(self, alpha=0.0, beta=1.0, gamma=0.0):
        """Weak form edge divergence operator with Robin boundary conditions.

        Parameters
        ----------
        alpha : float, optional
            Robin boundary condition parameter, by default 0.0
        beta : float, optional
            Robin boundary condition parameter, by default 1.0
        gamma : float, optional
            Robin boundary condition parameter, by default 0.0

        Returns
        -------
        tuple
            Tuple containing the weak form operator and boundary terms
        """
        alpha = torch.atleast_1d(alpha)
        beta = torch.atleast_1d(beta)
        gamma = torch.atleast_1d(gamma)

        if torch.any(beta == 0.0):
            raise ValueError("beta cannot have a zero value")

        Pbn = self.project_node_to_boundary_node
        Pbf = self.project_face_to_boundary_face

        n_boundary_faces = Pbf.shape[0]
        n_boundary_nodes = Pbn.shape[0]

        if len(alpha) == 1:
            if len(beta) != 1:
                alpha = torch.full(len(beta), alpha[0])
            elif len(gamma) != 1:
                alpha = torch.full(len(gamma), alpha[0])
            else:
                alpha = torch.full(n_boundary_faces, alpha[0])
        if len(beta) == 1:
            if len(alpha) != 1:
                beta = torch.full(len(alpha), beta[0])
        if len(gamma) == 1:
            if len(alpha) != 1:
                gamma = torch.full(len(alpha), gamma[0])

        if len(alpha) != len(beta) or len(beta) != len(gamma):
            raise ValueError("alpha, beta, and gamma must have the same length")

        if len(alpha) not in [n_boundary_faces, n_boundary_nodes]:
            raise ValueError(
                "The arrays must be of length n_boundary_faces or n_boundary_nodes"
            )

        AveN2F = self.average_node_to_face
        boundary_areas = Pbf @ self.face_areas
        AveBN2Bf = Pbf @ AveN2F @ Pbn.T

        # at the boundary, we have that u dot n = (gamma - alpha * phi)/beta
        if len(alpha) == n_boundary_faces:
            if gamma.ndim == 2:
                b = Pbn.T @ (
                    AveBN2Bf.T @ (gamma / beta[:, None] * boundary_areas[:, None])
                )
            else:
                b = Pbn.T @ (AveBN2Bf.T @ (gamma / beta * boundary_areas))
            B = sdiag(Pbn.T @ (AveBN2Bf.T @ (-alpha / beta * boundary_areas)))
        else:
            if gamma.ndim == 2:
                b = Pbn.T @ (
                    gamma / beta[:, None] * (AveBN2Bf.T @ boundary_areas)[:, None]
                )
            else:
                b = Pbn.T @ (gamma / beta * (AveBN2Bf.T @ boundary_areas))
            B = sdiag(Pbn.T @ (-alpha / beta * (AveBN2Bf.T @ boundary_areas)))
        return B, b

    ###########################################################################
    #                                                                         #
    #                             Cell Gradient                              #
    #                                                                         #
    ###########################################################################

    def set_cell_gradient_BC(self, BC):
        """Set boundary conditions for cell gradient operators.

        Parameters
        ----------
        BC : str or list
            Boundary condition specification
        """
        if isinstance(BC, str):
            BC = [BC] * self.dim
        if isinstance(BC, list):
            if len(BC) != self.dim:
                raise ValueError("BC list must be the size of your mesh")
        else:
            raise TypeError("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = _validate_BC(bc_i)

        # ensure we create a new gradient next time we call it
        self._cell_gradient = None
        self._cell_gradient_BC = None
        self._cell_gradient_BC_list = BC
        return BC

    @property
    def stencil_cell_gradient_x(self):
        """Stencil for cell gradient operator in the x-direction.

        This property constructs a differencing operator along the x-axis
        that acts on cell centered quantities; i.e. the stencil for the
        x-component of the cell gradient.

        Returns
        -------
        torch.sparse.Tensor
            The stencil for the x-component of the cell gradient
        """
        BC = ["neumann", "neumann"]
        if self.dim == 1:
            G1 = _ddxCellGrad(
                self.shape_cells[0], BC, device=self.device, dtype=self.dtype
            )
        elif self.dim == 2:
            G1 = kron(
                speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                _ddxCellGrad(
                    self.shape_cells[0], BC, device=self.device, dtype=self.dtype
                ),
            )
        elif self.dim == 3:
            G1 = kron3(
                speye(self.shape_cells[2], device=self.device, dtype=self.dtype),
                speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                _ddxCellGrad(
                    self.shape_cells[0], BC, device=self.device, dtype=self.dtype
                ),
            )
        return G1

    @property
    def stencil_cell_gradient_y(self):
        """Stencil for cell gradient operator in the y-direction.

        This property constructs a differencing operator along the y-axis
        that acts on cell centered quantities; i.e. the stencil for the
        y-component of the cell gradient.

        Returns
        -------
        torch.sparse.Tensor
            The stencil for the y-component of the cell gradient
        """
        if self.dim < 2:
            return None
        BC = ["neumann", "neumann"]
        n = self.shape_cells
        if self.dim == 2:
            G2 = kron(
                _ddxCellGrad(n[1], BC, device=self.device, dtype=self.dtype),
                speye(n[0], device=self.device, dtype=self.dtype),
            )
        elif self.dim == 3:
            G2 = kron3(
                speye(n[2], device=self.device, dtype=self.dtype),
                _ddxCellGrad(n[1], BC, device=self.device, dtype=self.dtype),
                speye(n[0], device=self.device, dtype=self.dtype),
            )
        return G2

    @property
    def stencil_cell_gradient_z(self):
        """Stencil for cell gradient operator in the z-direction.

        This property constructs a differencing operator along the z-axis
        that acts on cell centered quantities; i.e. the stencil for the
        z-component of the cell gradient.

        Returns
        -------
        torch.sparse.Tensor
            The stencil for the z-component of the cell gradient
        """
        if self.dim < 3:
            return None
        BC = ["neumann", "neumann"]
        n = self.shape_cells
        G3 = kron3(
            _ddxCellGrad(n[2], BC, device=self.device, dtype=self.dtype),
            speye(n[1], device=self.device, dtype=self.dtype),
            speye(n[0], device=self.device, dtype=self.dtype),
        )
        return G3

    @property
    def stencil_cell_gradient(self):
        """Stencil for cell gradient operator."""
        if getattr(self, "_stencil_cell_gradient", None) is None:
            if self.dim == 1:
                self._stencil_cell_gradient = self.stencil_cell_gradient_x
            elif self.dim == 2:
                G1 = self.stencil_cell_gradient_x
                G2 = self.stencil_cell_gradient_y
                self._stencil_cell_gradient = torch.cat([G1, G2], dim=0)
            elif self.dim == 3:
                G1 = self.stencil_cell_gradient_x
                G2 = self.stencil_cell_gradient_y
                G3 = self.stencil_cell_gradient_z
                self._stencil_cell_gradient = torch.cat([G1, G2, G3], dim=0)
        return self._stencil_cell_gradient

    @property
    def cell_gradient(self):
        """Cell gradient operator (cell centers to faces).

        Returns
        -------
        torch.sparse.Tensor
            The gradient operator matrix that maps from cell centers to faces
        """
        if getattr(self, "_cell_gradient", None) is None:
            G = self.stencil_cell_gradient
            S = self.face_areas  # Compute areas of cell faces
            V = (
                self.aveCC2F @ self.cell_volumes
            )  # Average volume between adjacent cells
            self._cell_gradient = sdiag(S / V, device=self.device, dtype=self.dtype) @ G
        return self._cell_gradient

    def cell_gradient_weak_form_robin(self, alpha=0.0, beta=1.0, gamma=0.0):
        """Weak form cell gradient operator with Robin boundary conditions.

        Parameters
        ----------
        alpha : float, optional
            Robin boundary condition parameter, by default 0.0
        beta : float, optional
            Robin boundary condition parameter, by default 1.0
        gamma : float, optional
            Robin boundary condition parameter, by default 0.0

        Returns
        -------
        tuple
            Tuple containing the weak form operator and boundary terms
        """
        raise NotImplementedError("cell_gradient_weak_form_robin not yet implemented")

    @property
    def cell_gradient_BC(self):
        """Cell gradient boundary condition operator (Deprecated)."""
        import warnings

        warnings.warn(
            "cell_gradient_BC is deprecated and is no longer used. See cell_gradient",
            DeprecationWarning,
            stacklevel=2,
        )
        return None  # This property is deprecated and not functional

    @property
    def cell_gradient_x(self):
        """X-derivative operator (cell centers to x-faces).

        This property constructs an x-derivative operator that acts on
        discrete scalar quantities defined at cell centers.
        When applied, the x-derivative is mapped to x-faces.

        Returns
        -------
        torch.sparse.Tensor
            X-derivative operator (x-component of the cell gradient)
        """
        if getattr(self, "_cell_gradient_x", None) is None:
            G1 = self.stencil_cell_gradient_x
            # Compute areas of cell faces & volumes
            V = self.aveCC2F @ self.cell_volumes
            # For x-faces, we need only the x-face areas
            S_x = self.face_areas[: self.nFx]
            V_x = V[: self.nFx]
            L = S_x / V_x
            self._cell_gradient_x = sdiag(L, device=self.device, dtype=self.dtype) @ G1
        return self._cell_gradient_x

    @property
    def cell_gradient_y(self):
        """Y-derivative operator (cell centers to y-faces).

        This property constructs a y-derivative operator that acts on
        discrete scalar quantities defined at cell centers.
        When applied, the y-derivative is mapped to y-faces.

        Returns
        -------
        torch.sparse.Tensor
            Y-derivative operator (y-component of the cell gradient)
        """
        if getattr(self, "_cell_gradient_y", None) is None:
            if self.dim < 2:
                raise RuntimeError(
                    "Y-derivative operator only exists for meshes with dim >= 2"
                )
            G2 = self.stencil_cell_gradient_y
            # Compute areas of cell faces & volumes
            V = self.aveCC2F @ self.cell_volumes
            # For y-faces, we need only the y-face areas
            S_y = self.face_areas[self.nFx : self.nFx + self.nFy]
            V_y = V[self.nFx : self.nFx + self.nFy]
            L = S_y / V_y
            self._cell_gradient_y = sdiag(L, device=self.device, dtype=self.dtype) @ G2
        return self._cell_gradient_y

    @property
    def cell_gradient_z(self):
        """Z-derivative operator (cell centers to z-faces).

        This property constructs an z-derivative operator that acts on
        discrete scalar quantities defined at cell centers.
        When applied, the z-derivative is mapped to z-faces.

        Returns
        -------
        torch.sparse.Tensor
            Z-derivative operator (z-component of the cell gradient)
        """
        if getattr(self, "_cell_gradient_z", None) is None:
            if self.dim < 3:
                raise RuntimeError(
                    "Z-derivative operator only exists for meshes with dim >= 3"
                )
            G3 = self.stencil_cell_gradient_z
            # Compute areas of cell faces & volumes
            V = self.aveCC2F @ self.cell_volumes
            # For z-faces, we need only the z-face areas
            S_z = self.face_areas[self.nFx + self.nFy :]
            V_z = V[self.nFx + self.nFy :]
            L = S_z / V_z
            self._cell_gradient_z = sdiag(L, device=self.device, dtype=self.dtype) @ G3
        return self._cell_gradient_z

    ###########################################################################
    #                                                                         #
    #                               Edge Curl                                #
    #                                                                         #
    ###########################################################################

    @property
    def _edge_x_curl_stencil(self):
        """Stencil for edge curl operator in the x-direction."""
        if self.dim < 3:
            raise NotImplementedError("Edge x-curl only programmed for 3D")
        n = self.shape_cells  # The number of cell centers in each direction

        D32 = kron3(
            ddx(n[2], device=self.device, dtype=self.dtype),
            speye(n[1], device=self.device, dtype=self.dtype),
            speye(n[0] + 1, device=self.device, dtype=self.dtype),
        )
        D23 = kron3(
            speye(n[2], device=self.device, dtype=self.dtype),
            ddx(n[1], device=self.device, dtype=self.dtype),
            speye(n[0] + 1, device=self.device, dtype=self.dtype),
        )
        O1 = spzeros(
            (n[0] + 1) * n[1] * n[2],
            n[0] * (n[1] + 1) * (n[2] + 1),
            device=self.device,
            dtype=self.dtype,
        )

        return torch.hstack((O1, -D32, D23))

    @property
    def _edge_y_curl_stencil(self):
        """Stencil for edge curl operator in the y-direction."""
        if self.dim < 3:
            raise NotImplementedError("Edge y-curl only programmed for 3D")
        n = self.shape_cells  # The number of cell centers in each direction

        D31 = kron3(
            ddx(n[2], device=self.device, dtype=self.dtype),
            speye(n[1] + 1, device=self.device, dtype=self.dtype),
            speye(n[0], device=self.device, dtype=self.dtype),
        )
        D13 = kron3(
            speye(n[2], device=self.device, dtype=self.dtype),
            speye(n[1] + 1, device=self.device, dtype=self.dtype),
            ddx(n[0], device=self.device, dtype=self.dtype),
        )
        O2 = spzeros(
            n[0] * (n[1] + 1) * n[2],
            (n[0] + 1) * n[1] * (n[2] + 1),
            device=self.device,
            dtype=self.dtype,
        )

        return torch.hstack((D31, O2, -D13))

    @property
    def _edge_z_curl_stencil(self):
        """Stencil for edge curl operator in the z-direction."""
        if self.dim < 3:
            raise NotImplementedError("Edge z-curl only programmed for 3D")
        n = self.shape_cells  # The number of cell centers in each direction

        D21 = kron3(
            speye(n[2] + 1, device=self.device, dtype=self.dtype),
            ddx(n[1], device=self.device, dtype=self.dtype),
            speye(n[0], device=self.device, dtype=self.dtype),
        )
        D12 = kron3(
            speye(n[2] + 1, device=self.device, dtype=self.dtype),
            speye(n[1], device=self.device, dtype=self.dtype),
            ddx(n[0], device=self.device, dtype=self.dtype),
        )
        O3 = spzeros(
            n[0] * n[1] * (n[2] + 1),
            (n[0] + 1) * (n[1] + 1) * n[2],
            device=self.device,
            dtype=self.dtype,
        )

        return torch.hstack((-D21, D12, O3))

    @property
    def _edge_curl_stencil(self):
        """Stencil for edge curl operator."""
        if self.dim <= 1:
            raise NotImplementedError("Edge Curl only programmed for 2 or 3D.")

        if self.dim == 2:
            n = self.shape_cells  # The number of cell centers in each direction

            D21 = kron(
                ddx(n[1], device=self.device, dtype=self.dtype),
                speye(n[0], device=self.device, dtype=self.dtype),
            )
            D12 = kron(
                speye(n[1], device=self.device, dtype=self.dtype),
                ddx(n[0], device=self.device, dtype=self.dtype),
            )
            C = torch.hstack((-D21, D12))
            return C

        elif self.dim == 3:
            C = torch.vstack(
                (
                    self._edge_x_curl_stencil,
                    self._edge_y_curl_stencil,
                    self._edge_z_curl_stencil,
                )
            )
            return C

    @property
    def edge_curl(self):
        """Edge curl operator (edges to faces).

        Returns
        -------
        torch.sparse.Tensor
            The curl operator matrix that maps from edges to faces
        """
        if getattr(self, "_edge_curl", None) is None:
            if self.dim <= 1:
                raise NotImplementedError("Edge Curl only programmed for 2 or 3D.")
            L = self.edge_lengths  # Compute lengths of cell edges
            if self.dim == 2:
                S = self.cell_volumes
            elif self.dim == 3:
                S = self.face_areas
            self._edge_curl = (
                sdiag(1 / S, device=self.device, dtype=self.dtype)
                @ self._edge_curl_stencil
                @ sdiag(L, device=self.device, dtype=self.dtype)
            )
        return self._edge_curl

    ###########################################################################
    #                                                                         #
    #                            Boundary Integrals                          #
    #                                                                         #
    ###########################################################################

    @property
    def boundary_face_scalar_integral(self):
        """Boundary face scalar integral operator.

        Returns
        -------
        torch.sparse.Tensor
            Operator for computing scalar integrals over boundary faces
        """
        if self.dim == 1:
            indices = torch.tensor([[0, self.n_faces_x - 1], [0, 1]])
            values = torch.tensor([-1, 1], device=self.device, dtype=self.dtype)
            return torch.sparse_coo_tensor(
                indices, values, (self.n_faces_x, 2)
            ).coalesce()
        P = self.project_face_to_boundary_face

        w_h_dot_normal = torch.sum(
            (P @ self.face_normals) * self.boundary_face_outward_normals, axis=-1
        )
        A = sdiag(self.face_areas) @ P.T @ sdiag(w_h_dot_normal)
        return A

    @property
    def boundary_edge_vector_integral(self):
        """Boundary edge vector integral operator.

        Returns
        -------
        torch.sparse.Tensor
            Operator for computing vector integrals over boundary edges
        """
        Pe = self.project_edge_to_boundary_edge
        Pf = self.project_face_to_boundary_face
        # Handle potential dimension mismatch in matrix-vector multiplication
        face_areas_proj = Pf @ self.face_areas
        if face_areas_proj.dim() == 0:
            face_areas_proj = face_areas_proj.unsqueeze(0)
        dA = self.boundary_face_outward_normals * face_areas_proj[:, None]
        w = Pe @ self.edge_tangents

        n_boundary_edges = len(w)

        Av = Pf @ self.average_edge_to_face @ Pe.T
        if self.dim > 2:
            Av *= 2

        av_da = Av.T @ dA

        if self.dim == 2:
            w_cross_n = cross2d(av_da, w)
        else:
            w_cross_n = torch.cross(av_da, w, dim=-1)

        if self.dim == 2:
            return Pe.T @ sdiag(w_cross_n)
        # Create sparse diagonal matrix equivalent to sp.diags for 3D case
        diags = w_cross_n.T  # Shape should be (3, n_boundary_edges)
        offsets = n_boundary_edges * torch.arange(3, device=self.device)

        row_indices = []
        col_indices = []
        values = []

        for i, offset in enumerate(offsets):
            for j in range(n_boundary_edges):
                row_idx = j
                col_idx = j + offset.item()
                if 0 <= col_idx < 3 * n_boundary_edges:
                    row_indices.append(row_idx)
                    col_indices.append(col_idx)
                    values.append(diags[i, j])

        if len(row_indices) > 0:
            indices = torch.stack(
                [
                    torch.tensor(row_indices, device=self.device),
                    torch.tensor(col_indices, device=self.device),
                ]
            )
            values_tensor = torch.tensor(values, device=self.device, dtype=self.dtype)
            diag_matrix = torch.sparse_coo_tensor(
                indices,
                values_tensor,
                (n_boundary_edges, 3 * n_boundary_edges),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()
        else:
            # Empty sparse matrix
            diag_matrix = torch.sparse_coo_tensor(
                torch.zeros((2, 0), device=self.device),
                torch.zeros(0, device=self.device, dtype=self.dtype),
                (n_boundary_edges, 3 * n_boundary_edges),
                device=self.device,
                dtype=self.dtype,
            )

        return Pe.T @ diag_matrix

    @property
    def boundary_node_vector_integral(self):
        """Boundary node vector integral operator.

        Returns
        -------
        torch.sparse.Tensor
            Operator for computing vector integrals over boundary nodes
        """
        if self.dim == 1:
            indices = torch.tensor([[0, self.shape_nodes[0] - 1], [0, 1]])
            values = torch.tensor([-1, 1], device=self.device, dtype=self.dtype)
            return torch.sparse_coo_tensor(
                indices, values, (self.shape_nodes[0], 2)
            ).coalesce()

        Pn = self.project_node_to_boundary_node
        Pf = self.project_face_to_boundary_face
        n_boundary_nodes = Pn.shape[0]

        # Handle potential dimension mismatch in matrix-vector multiplication
        face_areas_proj = Pf @ self.face_areas
        if face_areas_proj.dim() == 0:
            face_areas_proj = face_areas_proj.unsqueeze(0)
        dA = self.boundary_face_outward_normals * face_areas_proj[:, None]

        Av = Pf @ self.average_node_to_face @ Pn.T

        u_dot_ds = Av.T @ dA
        diags = u_dot_ds.T
        offsets = n_boundary_nodes * torch.arange(self.dim)

        # Create sparse diagonal matrix equivalent to sp.diags
        # diags should be shape (self.dim, n_boundary_nodes)
        # offsets should be offsets for each diagonal
        row_indices = []
        col_indices = []
        values = []

        for i, offset in enumerate(offsets):
            for j in range(n_boundary_nodes):
                row_idx = j
                col_idx = j + offset.item()
                if 0 <= col_idx < self.dim * n_boundary_nodes:
                    row_indices.append(row_idx)
                    col_indices.append(col_idx)
                    values.append(diags[i, j])

        if len(row_indices) > 0:
            indices = torch.stack(
                [
                    torch.tensor(row_indices, device=self.device),
                    torch.tensor(col_indices, device=self.device),
                ]
            )
            values_tensor = torch.tensor(values, device=self.device, dtype=self.dtype)
            diag_matrix = torch.sparse_coo_tensor(
                indices,
                values_tensor,
                (n_boundary_nodes, self.dim * n_boundary_nodes),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()
        else:
            # Empty sparse matrix
            diag_matrix = torch.sparse_coo_tensor(
                torch.zeros((2, 0), device=self.device),
                torch.zeros(0, device=self.device, dtype=self.dtype),
                (n_boundary_nodes, self.dim * n_boundary_nodes),
                device=self.device,
                dtype=self.dtype,
            )

        return Pn.T @ diag_matrix

    ###########################################################################
    #                                                                         #
    #                         Boundary Conditions                            #
    #                                                                         #
    ###########################################################################

    def get_BC_projections(self, BC, discretization="CC"):
        """Create the weak form boundary condition projection matrices.

        Examples
        --------
        .. code:: python

            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain, Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """
        if discretization != "CC":
            raise NotImplementedError(
                "Boundary conditions only implemented for CC discretization."
            )

        if isinstance(BC, str):
            BC = [BC for _ in self.vnC]  # Repeat the str self.dim times
        elif isinstance(BC, list):
            if len(BC) != self.dim:
                raise ValueError("BC list must be the size of your mesh")
        else:
            raise TypeError("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = _validate_BC(bc_i)

        # Get device and dtype from self attributes
        device = self.face_areas.device
        dtype = self.face_areas.dtype

        def projDirichlet(n, bc):
            bc = _validate_BC(bc)
            # Create sparse COO matrix
            indices = torch.tensor([[0, n], [0, 1]], dtype=torch.long, device=device)
            vals = torch.zeros(2, dtype=dtype, device=device)
            if bc[0] == "dirichlet":
                vals[0] = -1.0
            if bc[1] == "dirichlet":
                vals[1] = 1.0
            return torch.sparse_coo_tensor(
                indices, vals, size=(n + 1, 2), dtype=dtype, device=device
            ).coalesce()

        def projNeumannIn(n, bc):
            bc = _validate_BC(bc)
            P = speye(n + 1)  # Identity matrix in COO format
            if bc[0] == "neumann":
                # Remove first row: P[1:, :]
                mask = torch.arange(P.size(0), device=device) >= 1
                row_indices = P.indices()[0]
                valid_mask = mask[row_indices]
                new_indices = P.indices()[:, valid_mask]
                new_indices[0] -= 1  # Adjust row indices
                new_values = P.values()[valid_mask]
                P = torch.sparse_coo_tensor(
                    new_indices,
                    new_values,
                    size=(n, P.size(1)),
                    dtype=dtype,
                    device=device,
                ).coalesce()
            if bc[1] == "neumann":
                # Remove last row: P[:-1, :]
                mask = torch.arange(P.size(0), device=device) < P.size(0) - 1
                row_indices = P.indices()[0]
                valid_mask = mask[row_indices]
                new_indices = P.indices()[:, valid_mask]
                new_values = P.values()[valid_mask]
                P = torch.sparse_coo_tensor(
                    new_indices,
                    new_values,
                    size=(P.size(0) - 1, P.size(1)),
                    dtype=dtype,
                    device=device,
                ).coalesce()
            return P

        def projNeumannOut(n, bc):
            bc = _validate_BC(bc)
            indices = torch.tensor([[0, 1], [0, n]], dtype=torch.long, device=device)
            vals = torch.zeros(2, dtype=dtype, device=device)
            if bc[0] == "neumann":
                vals[0] = 1.0
            if bc[1] == "neumann":
                vals[1] = 1.0
            return torch.sparse_coo_tensor(
                indices, vals, size=(2, n + 1), dtype=dtype, device=device
            ).coalesce()

        n = self.vnC
        indF = self.face_boundary_indices

        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            indF = indF[0] | indF[1]
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

            Pin = projNeumannIn(n[0], BC[0])
            Pout = projNeumannOut(n[0], BC[0])

        elif self.dim == 2:
            Pbc1 = kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = torch_blockdiag([Pbc1, Pbc2])
            indF = torch.cat([indF[0] | indF[1], indF[2] | indF[3]])
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

            P1 = kron(speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = kron(projNeumannIn(n[1], BC[1]), speye(n[0]))
            Pin = torch_blockdiag([P1, P2])

            P1 = kron(speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = kron(projNeumannOut(n[1], BC[1]), speye(n[0]))
            Pout = torch_blockdiag([P1, P2])

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = torch_blockdiag([Pbc1, Pbc2, Pbc3])
            indF = torch.cat([indF[0] | indF[1], indF[2] | indF[3], indF[4] | indF[5]])
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannIn(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannIn(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pin = torch_blockdiag([P1, P2, P3])

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannOut(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannOut(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pout = torch_blockdiag([P1, P2, P3])

        return Pbc, Pin, Pout

    def get_BC_projections_simple(self, discretization="CC"):
        """Create weak form boundary condition projection matrices for mixed boundary condition."""
        if discretization != "CC":
            raise NotImplementedError(
                "Boundary conditions only implemented for CC discretization."
            )

        # Get device and dtype from self attributes
        device = self.face_areas.device
        dtype = self.face_areas.dtype

        def projBC(n):
            indices = torch.tensor([[0, n], [0, 1]], dtype=torch.long, device=device)
            vals = torch.ones(2, dtype=dtype, device=device)
            return torch.sparse_coo_tensor(
                indices, vals, size=(n + 1, 2), dtype=dtype, device=device
            ).coalesce()

        def projDirichlet(n, bc):
            bc = _validate_BC(bc)
            indices = torch.tensor([[0, n], [0, 1]], dtype=torch.long, device=device)
            vals = torch.zeros(2, dtype=dtype, device=device)
            if bc[0] == "dirichlet":
                vals[0] = -1.0
            if bc[1] == "dirichlet":
                vals[1] = 1.0
            return torch.sparse_coo_tensor(
                indices, vals, size=(n + 1, 2), dtype=dtype, device=device
            ).coalesce()

        BC = [
            ["dirichlet", "dirichlet"],
            ["dirichlet", "dirichlet"],
            ["dirichlet", "dirichlet"],
        ]
        n = self.vnC
        indF = self.face_boundary_indices

        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            B = projBC(n[0])
            indF = indF[0] | indF[1]
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

        elif self.dim == 2:
            Pbc1 = kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = torch_blockdiag([Pbc1, Pbc2])
            B1 = kron(speye(n[1]), projBC(n[0]))
            B2 = kron(projBC(n[1]), speye(n[0]))
            B = torch_blockdiag([B1, B2])
            indF = torch.cat([indF[0] | indF[1], indF[2] | indF[3]])
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = torch_blockdiag([Pbc1, Pbc2, Pbc3])
            B1 = kron3(speye(n[2]), speye(n[1]), projBC(n[0]))
            B2 = kron3(speye(n[2]), projBC(n[1]), speye(n[0]))
            B3 = kron3(projBC(n[2]), speye(n[1]), speye(n[0]))
            B = torch_blockdiag([B1, B2, B3])
            indF = torch.cat([indF[0] | indF[1], indF[2] | indF[3], indF[4] | indF[5]])
            Pbc = torch.sparse.mm(Pbc, sdiag(self.face_areas[indF]))

        return Pbc, B.t()  # .t() is transpose for PyTorch tensors

    ###########################################################################
    #                                                                         #
    #                                Averaging                                #
    #                                                                         #
    ###########################################################################

    @property
    def average_face_to_cell(self):
        """Averaging operator from faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages face values to cell centers
        """
        if getattr(self, "_average_face_to_cell", None) is None:
            if self.dim == 1:
                self._average_face_to_cell = self.average_face_x_to_cell
            elif self.dim == 2:
                self._average_face_to_cell = (0.5) * torch.cat(
                    [self.average_face_x_to_cell, self.average_face_y_to_cell], dim=1
                )
            elif self.dim == 3:
                self._average_face_to_cell = (1.0 / 3.0) * torch.cat(
                    [
                        self.average_face_x_to_cell,
                        self.average_face_y_to_cell,
                        self.average_face_z_to_cell,
                    ],
                    dim=1,
                )
        return self._average_face_to_cell

    @property
    def average_face_to_cell_vector(self):
        """Vector averaging operator from faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages face vector values to cell centers
        """
        if getattr(self, "_average_face_to_cell_vector", None) is None:
            if self.dim == 1:
                self._average_face_to_cell_vector = self.average_face_x_to_cell
            elif self.dim == 2:
                # Create block diagonal matrix using torch_blockdiag
                afx = self.average_face_x_to_cell
                afy = self.average_face_y_to_cell
                self._average_face_to_cell_vector = torch_blockdiag([afx, afy])
            elif self.dim == 3:
                # Create block diagonal matrix using torch_blockdiag
                afx = self.average_face_x_to_cell
                afy = self.average_face_y_to_cell
                afz = self.average_face_z_to_cell
                self._average_face_to_cell_vector = torch_blockdiag([afx, afy, afz])
        return self._average_face_to_cell_vector

    @property
    def average_face_x_to_cell(self):
        """Averaging operator from x-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages x-face values to cell centers
        """
        if getattr(self, "_average_face_x_to_cell", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                self._average_face_x_to_cell = av(
                    n[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                self._average_face_x_to_cell = kron(
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_face_x_to_cell = kron3(
                    speye(n[2], device=self.device, dtype=self.dtype),
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_face_x_to_cell

    @property
    def average_face_y_to_cell(self):
        """Averaging operator from y-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages y-face values to cell centers
        """
        if self.dim < 2:
            return None
        if getattr(self, "_average_face_y_to_cell", None) is None:
            n = self.shape_cells
            if self.dim == 2:
                self._average_face_y_to_cell = kron(
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_face_y_to_cell = kron3(
                    speye(n[2], device=self.device, dtype=self.dtype),
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_face_y_to_cell

    @property
    def average_face_z_to_cell(self):
        """Averaging operator from z-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages z-face values to cell centers
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_face_z_to_cell", None) is None:
            n = self.shape_cells
            self._average_face_z_to_cell = kron3(
                av(n[2], device=self.device, dtype=self.dtype),
                speye(n[1], device=self.device, dtype=self.dtype),
                speye(n[0], device=self.device, dtype=self.dtype),
            )
        return self._average_face_z_to_cell

    @property
    def average_cell_to_face(self):
        """Averaging operator from cell centers to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to faces
        """
        if getattr(self, "_average_cell_to_face", None) is None:
            if self.dim == 1:
                self._average_cell_to_face = av_extrap(
                    self.shape_cells[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                G1 = kron(
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[0], device=self.device, dtype=self.dtype
                    ),
                )
                G2 = kron(
                    av_extrap(
                        self.shape_cells[1], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                self._average_cell_to_face = torch.cat([G1, G2], dim=0)
            elif self.dim == 3:
                G1 = kron3(
                    speye(self.shape_cells[2], device=self.device, dtype=self.dtype),
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[0], device=self.device, dtype=self.dtype
                    ),
                )
                G2 = kron3(
                    speye(self.shape_cells[2], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[1], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                G3 = kron3(
                    av_extrap(
                        self.shape_cells[2], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                self._average_cell_to_face = torch.cat([G1, G2, G3], dim=0)
        return self._average_cell_to_face

    @property
    def average_cell_vector_to_face(self):
        """Vector averaging operator from cell centers to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center vector values to faces
        """
        if getattr(self, "_average_cell_vector_to_face", None) is None:
            if self.dim == 1:
                self._average_cell_vector_to_face = self.average_cell_to_face
            elif self.dim == 2:
                # Create averaging operators for each direction
                aveCCV2Fx = kron(
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[0], device=self.device, dtype=self.dtype
                    ),
                )
                aveCC2VFy = kron(
                    av_extrap(
                        self.shape_cells[1], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                self._average_cell_vector_to_face = torch_blockdiag(
                    [aveCCV2Fx, aveCC2VFy]
                )
            elif self.dim == 3:
                # Create averaging operators for each direction
                aveCCV2Fx = kron3(
                    speye(self.shape_cells[2], device=self.device, dtype=self.dtype),
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[0], device=self.device, dtype=self.dtype
                    ),
                )
                aveCC2VFy = kron3(
                    speye(self.shape_cells[2], device=self.device, dtype=self.dtype),
                    av_extrap(
                        self.shape_cells[1], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                aveCC2BFz = kron3(
                    av_extrap(
                        self.shape_cells[2], device=self.device, dtype=self.dtype
                    ),
                    speye(self.shape_cells[1], device=self.device, dtype=self.dtype),
                    speye(self.shape_cells[0], device=self.device, dtype=self.dtype),
                )
                self._average_cell_vector_to_face = torch_blockdiag(
                    [aveCCV2Fx, aveCC2VFy, aveCC2BFz]
                )
        return self._average_cell_vector_to_face

    @property
    def average_cell_to_edge(self):
        """Averaging operator from cell centers to edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to edges
        """
        if getattr(self, "_average_cell_to_edge", None) is None:
            if self.dim == 1:
                self._average_cell_to_edge = self.average_cell_to_edge_x
            elif self.dim == 2:
                self._average_cell_to_edge = torch.cat(
                    [self.average_cell_to_edge_x, self.average_cell_to_edge_y], dim=0
                )
            elif self.dim == 3:
                self._average_cell_to_edge = torch.cat(
                    [
                        self.average_cell_to_edge_x,
                        self.average_cell_to_edge_y,
                        self.average_cell_to_edge_z,
                    ],
                    dim=0,
                )
        return self._average_cell_to_edge

    @property
    def average_cell_to_edge_x(self):
        """Averaging operator from cell centers to x-edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to x-edges
        """
        if getattr(self, "_average_cell_to_edge_x", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                # In 1D: cells and edges are the same, use identity
                self._average_cell_to_edge_x = speye(
                    n[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                self._average_cell_to_edge_x = kron(
                    av_extrap(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_cell_to_edge_x = kron3(
                    av_extrap(n[2], device=self.device, dtype=self.dtype),
                    av_extrap(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_cell_to_edge_x

    @property
    def average_cell_to_edge_y(self):
        """Averaging operator from cell centers to y-edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to y-edges
        """
        if self.dim < 2:
            return None
        if getattr(self, "_average_cell_to_edge_y", None) is None:
            n = self.shape_cells
            if self.dim == 2:
                self._average_cell_to_edge_y = kron(
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av_extrap(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_cell_to_edge_y = kron3(
                    av_extrap(n[2], device=self.device, dtype=self.dtype),
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av_extrap(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_cell_to_edge_y

    @property
    def average_cell_to_edge_z(self):
        """Averaging operator from cell centers to z-edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to z-edges
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_cell_to_edge_z", None) is None:
            n = self.shape_cells
            self._average_cell_to_edge_z = kron3(
                speye(n[2], device=self.device, dtype=self.dtype),
                av_extrap(n[1], device=self.device, dtype=self.dtype),
                av_extrap(n[0], device=self.device, dtype=self.dtype),
            )
        return self._average_cell_to_edge_z

    @property
    def average_edge_to_cell(self):
        """Averaging operator from edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge values to cell centers
        """
        if getattr(self, "_average_edge_to_cell", None) is None:
            if self.dim == 1:
                self._average_edge_to_cell = self.average_edge_x_to_cell
            elif self.dim == 2:
                self._average_edge_to_cell = 0.5 * torch.cat(
                    [self.average_edge_x_to_cell, self.average_edge_y_to_cell], dim=1
                )
            elif self.dim == 3:
                self._average_edge_to_cell = (1.0 / 3.0) * torch.cat(
                    [
                        self.average_edge_x_to_cell,
                        self.average_edge_y_to_cell,
                        self.average_edge_z_to_cell,
                    ],
                    dim=1,
                )
        return self._average_edge_to_cell

    @property
    def average_edge_to_cell_vector(self):
        """Vector averaging operator from edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge vector values to cell centers
        """
        if getattr(self, "_average_edge_to_cell_vector", None) is None:
            if self.dim == 1:
                self._average_edge_to_cell_vector = self.average_edge_x_to_cell
            elif self.dim == 2:
                # Create block diagonal matrix using torch_blockdiag
                aex = self.average_edge_x_to_cell
                aey = self.average_edge_y_to_cell
                self._average_edge_to_cell_vector = torch_blockdiag([aex, aey])
            elif self.dim == 3:
                # Create block diagonal matrix using torch_blockdiag
                aex = self.average_edge_x_to_cell
                aey = self.average_edge_y_to_cell
                aez = self.average_edge_z_to_cell
                self._average_edge_to_cell_vector = torch_blockdiag([aex, aey, aez])
        return self._average_edge_to_cell_vector

    @property
    def average_edge_x_to_cell(self):
        """Averaging operator from x-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages x-edge values to cell centers
        """
        if getattr(self, "_average_edge_x_to_cell", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                self._average_edge_x_to_cell = speye(
                    n[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                self._average_edge_x_to_cell = kron(
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_edge_x_to_cell = kron3(
                    av(n[2], device=self.device, dtype=self.dtype),
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_edge_x_to_cell

    @property
    def average_edge_y_to_cell(self):
        """Averaging operator from y-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages y-edge values to cell centers
        """
        if self.dim < 2:
            return None
        if getattr(self, "_average_edge_y_to_cell", None) is None:
            n = self.shape_cells
            if self.dim == 2:
                self._average_edge_y_to_cell = kron(
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                self._average_edge_y_to_cell = kron3(
                    av(n[2], device=self.device, dtype=self.dtype),
                    speye(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_edge_y_to_cell

    @property
    def average_edge_z_to_cell(self):
        """Averaging operator from z-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages z-edge values to cell centers
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_edge_z_to_cell", None) is None:
            n = self.shape_cells
            self._average_edge_z_to_cell = kron3(
                speye(n[2], device=self.device, dtype=self.dtype),
                av(n[1], device=self.device, dtype=self.dtype),
                av(n[0], device=self.device, dtype=self.dtype),
            )
        return self._average_edge_z_to_cell

    @property
    def average_edge_to_face(self):
        """Averaging operator from edges to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge values to faces
        """
        if getattr(self, "_average_edge_to_face", None) is None:
            if self.dim == 1:
                # In 1D: edges are cell centers, use cell-to-face averaging
                self._average_edge_to_face = av_extrap(
                    self.shape_cells[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                # In 2D: this is a complex mapping. For simplicity, implement
                # a basic version that just does block averaging
                # x-faces from y-edges, y-faces from x-edges

                # For x-faces: average from the corresponding y-edges
                # Each x-face corresponds to edges in the y-direction at the same location
                zero_xy = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros(0, dtype=self.dtype, device=self.device),
                    (self.n_faces_x, self.n_edges_x),
                    device=self.device,
                )

                # Simple identity mapping for cross-component averaging (assuming same counts)
                if self.n_faces_x == self.n_edges_y:
                    ave_y_to_fx = speye(
                        self.n_faces_x, device=self.device, dtype=self.dtype
                    )
                else:
                    # If sizes don't match, create a rectangular mapping
                    indices = torch.arange(
                        min(self.n_faces_x, self.n_edges_y), device=self.device
                    )
                    ave_y_to_fx = torch.sparse_coo_tensor(
                        torch.stack([indices, indices]),
                        torch.ones(len(indices), dtype=self.dtype, device=self.device),
                        (self.n_faces_x, self.n_edges_y),
                        device=self.device,
                    )

                zero_yx = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros(0, dtype=self.dtype, device=self.device),
                    (self.n_faces_y, self.n_edges_y),
                    device=self.device,
                )

                if self.n_faces_y == self.n_edges_x:
                    ave_x_to_fy = speye(
                        self.n_faces_y, device=self.device, dtype=self.dtype
                    )
                else:
                    # If sizes don't match, create a rectangular mapping
                    indices = torch.arange(
                        min(self.n_faces_y, self.n_edges_x), device=self.device
                    )
                    ave_x_to_fy = torch.sparse_coo_tensor(
                        torch.stack([indices, indices]),
                        torch.ones(len(indices), dtype=self.dtype, device=self.device),
                        (self.n_faces_y, self.n_edges_x),
                        device=self.device,
                    )

                # Create block matrix: [0, I; I, 0] pattern
                top_row = torch.cat([zero_xy, ave_y_to_fx], dim=1)
                bottom_row = torch.cat([ave_x_to_fy, zero_yx], dim=1)
                self._average_edge_to_face = torch.cat([top_row, bottom_row], dim=0)

            elif self.dim == 3:
                # In 3D: more complex averaging relationships
                # x-faces: average from y-edges and z-edges
                # y-faces: average from x-edges and z-edges
                # z-faces: average from x-edges and y-edges

                # 3D edge to face averaging based on original SimPEG implementation
                n1, n2, n3 = self.shape_cells
                ex_to_fy = kron3(
                    av(n3, device=self.device, dtype=self.dtype),
                    speye(n2 + 1, device=self.device, dtype=self.dtype),
                    speye(n1, device=self.device, dtype=self.dtype),
                )
                ex_to_fz = kron3(
                    speye(n3 + 1, device=self.device, dtype=self.dtype),
                    av(n2, device=self.device, dtype=self.dtype),
                    speye(n1, device=self.device, dtype=self.dtype),
                )

                ey_to_fx = kron3(
                    av(n3, device=self.device, dtype=self.dtype),
                    speye(n2, device=self.device, dtype=self.dtype),
                    speye(n1 + 1, device=self.device, dtype=self.dtype),
                )
                ey_to_fz = kron3(
                    speye(n3 + 1, device=self.device, dtype=self.dtype),
                    speye(n2, device=self.device, dtype=self.dtype),
                    av(n1, device=self.device, dtype=self.dtype),
                )

                ez_to_fx = kron3(
                    speye(n3, device=self.device, dtype=self.dtype),
                    av(n2, device=self.device, dtype=self.dtype),
                    speye(n1 + 1, device=self.device, dtype=self.dtype),
                )
                ez_to_fy = kron3(
                    speye(n3, device=self.device, dtype=self.dtype),
                    speye(n2 + 1, device=self.device, dtype=self.dtype),
                    av(n1, device=self.device, dtype=self.dtype),
                )

                # Create block matrix [None, ey_to_fx, ez_to_fx; ex_to_fy, None, ez_to_fy; ex_to_fz, ey_to_fz, None]
                zeros_x = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), device=self.device),
                    torch.zeros(0, device=self.device, dtype=self.dtype),
                    (self.n_faces_x, self.n_edges_x),
                    device=self.device,
                    dtype=self.dtype,
                )
                zeros_y = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), device=self.device),
                    torch.zeros(0, device=self.device, dtype=self.dtype),
                    (self.n_faces_y, self.n_edges_y),
                    device=self.device,
                    dtype=self.dtype,
                )
                zeros_z = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), device=self.device),
                    torch.zeros(0, device=self.device, dtype=self.dtype),
                    (self.n_faces_z, self.n_edges_z),
                    device=self.device,
                    dtype=self.dtype,
                )

                # Build the block matrix row by row
                row1 = torch.cat([zeros_x, ey_to_fx, ez_to_fx], dim=1)
                row2 = torch.cat([ex_to_fy, zeros_y, ez_to_fy], dim=1)
                row3 = torch.cat([ex_to_fz, ey_to_fz, zeros_z], dim=1)

                self._average_edge_to_face = 0.5 * torch.cat([row1, row2, row3], dim=0)

        return self._average_edge_to_face

    @property
    def average_node_to_cell(self):
        """Averaging operator from nodes to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to cell centers
        """
        if getattr(self, "_average_node_to_cell", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                # In 1D: average 2 nodes to get 1 cell center
                self._average_node_to_cell = av(
                    n[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                # In 2D: average 4 nodes to get 1 cell center
                self._average_node_to_cell = kron(
                    av(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                # In 3D: average 8 nodes to get 1 cell center
                self._average_node_to_cell = kron3(
                    av(n[2], device=self.device, dtype=self.dtype),
                    av(n[1], device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
        return self._average_node_to_cell

    @property
    def _average_node_to_edge_x(self):
        """Internal averaging operator from nodes to x-edges."""
        if getattr(self, "__average_node_to_edge_x", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                # In 1D: average nodes to edges (cell centers)
                self.__average_node_to_edge_x = av(
                    self.shape_cells[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                # In 2D: x-edges span in x-direction, average nodes in x-direction
                self.__average_node_to_edge_x = kron(
                    speye(n[1] + 1, device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                # In 3D: x-edges span in x-direction
                self.__average_node_to_edge_x = kron3(
                    speye(n[2] + 1, device=self.device, dtype=self.dtype),
                    speye(n[1] + 1, device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
        return self.__average_node_to_edge_x

    @property
    def _average_node_to_edge_y(self):
        """Internal averaging operator from nodes to y-edges."""
        if self.dim < 2:
            return None
        if getattr(self, "__average_node_to_edge_y", None) is None:
            n = self.shape_cells
            if self.dim == 2:
                # In 2D: y-edges span in y-direction, average nodes in y-direction
                self.__average_node_to_edge_y = kron(
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0] + 1, device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                # In 3D: y-edges span in y-direction
                self.__average_node_to_edge_y = kron3(
                    speye(n[2] + 1, device=self.device, dtype=self.dtype),
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0] + 1, device=self.device, dtype=self.dtype),
                )
        return self.__average_node_to_edge_y

    @property
    def _average_node_to_edge_z(self):
        """Internal averaging operator from nodes to z-edges."""
        if self.dim < 3:
            return None
        if getattr(self, "__average_node_to_edge_z", None) is None:
            n = self.shape_cells
            # In 3D: z-edges span in z-direction, average nodes in z-direction
            self.__average_node_to_edge_z = kron3(
                av(n[2], device=self.device, dtype=self.dtype),
                speye(n[1] + 1, device=self.device, dtype=self.dtype),
                speye(n[0] + 1, device=self.device, dtype=self.dtype),
            )
        return self.__average_node_to_edge_z

    @property
    def average_node_to_edge(self):
        """Averaging operator from nodes to edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to edges
        """
        if getattr(self, "_average_node_to_edge", None) is None:
            if self.dim == 1:
                self._average_node_to_edge = self._average_node_to_edge_x
            elif self.dim == 2:
                self._average_node_to_edge = torch.cat(
                    [self._average_node_to_edge_x, self._average_node_to_edge_y], dim=0
                )
            elif self.dim == 3:
                self._average_node_to_edge = torch.cat(
                    [
                        self._average_node_to_edge_x,
                        self._average_node_to_edge_y,
                        self._average_node_to_edge_z,
                    ],
                    dim=0,
                )
        return self._average_node_to_edge

    @property
    def _average_node_to_face_x(self):
        """Internal averaging operator from nodes to x-faces."""
        if getattr(self, "__average_node_to_face_x", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                # In 1D: x-faces are just node locations (identity)
                self.__average_node_to_face_x = speye(
                    self.shape_nodes[0], device=self.device, dtype=self.dtype
                )
            elif self.dim == 2:
                # In 2D: x-faces are normal to x-direction, average nodes in y-direction
                self.__average_node_to_face_x = kron(
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0] + 1, device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                # In 3D: x-faces are normal to x-direction, average nodes in y,z directions
                self.__average_node_to_face_x = kron3(
                    av(n[2], device=self.device, dtype=self.dtype),
                    av(n[1], device=self.device, dtype=self.dtype),
                    speye(n[0] + 1, device=self.device, dtype=self.dtype),
                )
        return self.__average_node_to_face_x

    @property
    def _average_node_to_face_y(self):
        """Internal averaging operator from nodes to y-faces."""
        if self.dim < 2:
            return None
        if getattr(self, "__average_node_to_face_y", None) is None:
            n = self.shape_cells
            if self.dim == 2:
                # In 2D: y-faces are normal to y-direction, average nodes in x-direction
                self.__average_node_to_face_y = kron(
                    speye(n[1] + 1, device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
            elif self.dim == 3:
                # In 3D: y-faces are normal to y-direction, average nodes in x,z directions
                self.__average_node_to_face_y = kron3(
                    av(n[2], device=self.device, dtype=self.dtype),
                    speye(n[1] + 1, device=self.device, dtype=self.dtype),
                    av(n[0], device=self.device, dtype=self.dtype),
                )
        return self.__average_node_to_face_y

    @property
    def _average_node_to_face_z(self):
        """Internal averaging operator from nodes to z-faces."""
        if self.dim < 3:
            return None
        if getattr(self, "__average_node_to_face_z", None) is None:
            n = self.shape_cells
            # In 3D: z-faces are normal to z-direction, average nodes in x,y directions
            self.__average_node_to_face_z = kron3(
                speye(n[2] + 1, device=self.device, dtype=self.dtype),
                av(n[1], device=self.device, dtype=self.dtype),
                av(n[0], device=self.device, dtype=self.dtype),
            )
        return self.__average_node_to_face_z

    @property
    def average_node_to_face(self):
        """Averaging operator from nodes to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to faces
        """
        if getattr(self, "_average_node_to_face", None) is None:
            if self.dim == 1:
                self._average_node_to_face = self._average_node_to_face_x
            elif self.dim == 2:
                self._average_node_to_face = torch.cat(
                    [self._average_node_to_face_x, self._average_node_to_face_y], dim=0
                )
            elif self.dim == 3:
                self._average_node_to_face = torch.cat(
                    [
                        self._average_node_to_face_x,
                        self._average_node_to_face_y,
                        self._average_node_to_face_z,
                    ],
                    dim=0,
                )
        return self._average_node_to_face

    ###########################################################################
    #                                                                         #
    #                              Projections                               #
    #                                                                         #
    ###########################################################################

    @property
    def project_face_to_boundary_face(self):
        """Projection operator from faces to boundary faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that projects face values to boundary faces
        """
        is_b = make_boundary_bool(self.shape_faces_x, bdir="x")
        if self.dim > 1:
            is_b = torch.cat([is_b, make_boundary_bool(self.shape_faces_y, bdir="y")])
        if self.dim == 3:
            is_b = torch.cat([is_b, make_boundary_bool(self.shape_faces_z, bdir="z")])

        boundary_indices = torch.nonzero(is_b, as_tuple=False).squeeze(1)
        n_boundary = boundary_indices.numel()

        row_indices = torch.arange(n_boundary, device=is_b.device)
        col_indices = boundary_indices
        values = torch.ones(n_boundary, dtype=torch.float64, device=is_b.device)

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (n_boundary, self.n_faces)
        ).coalesce()

    @property
    def project_edge_to_boundary_edge(self):
        """Projection operator from edges to boundary edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that projects edge values to boundary edges
        """
        if self.dim == 1:
            return None  # No edges are on the boundary in 1D

        is_b = torch.cat(
            [
                make_boundary_bool(self.shape_edges_x, bdir="yz"),
                make_boundary_bool(self.shape_edges_y, bdir="xz"),
            ]
        )
        if self.dim == 3:
            is_b = torch.cat([is_b, make_boundary_bool(self.shape_edges_z, bdir="xy")])

        boundary_indices = torch.nonzero(is_b, as_tuple=False).squeeze(1)
        n_boundary = boundary_indices.numel()

        row_indices = torch.arange(n_boundary, device=is_b.device)
        col_indices = boundary_indices
        values = torch.ones(n_boundary, dtype=torch.float64, device=is_b.device)

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (n_boundary, self.n_edges)
        ).coalesce()

    @property
    def project_node_to_boundary_node(self):
        """Projection operator from nodes to boundary nodes.

        Returns
        -------
        torch.sparse.Tensor
            Operator that projects node values to boundary nodes
        """
        is_b = make_boundary_bool(self.shape_nodes)
        boundary_indices = torch.nonzero(is_b, as_tuple=False).squeeze(1)
        n_boundary = boundary_indices.numel()

        row_indices = torch.arange(n_boundary, device=is_b.device)
        col_indices = boundary_indices
        values = torch.ones(n_boundary, dtype=torch.float64, device=is_b.device)

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (n_boundary, self.n_nodes)
        ).coalesce()
