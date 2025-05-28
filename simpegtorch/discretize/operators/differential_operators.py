"""Differential operators for tensor meshes in PyTorch.

This module provides differential and averaging operators for finite volume
computations on tensor meshes, implemented using PyTorch for automatic
differentiation and GPU acceleration.
"""

import torch
from simpegtorch.discretize.base import BaseMesh


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

    # Set boundary conditions
    # TODO: Implement proper boundary condition handling
    # This is a placeholder implementation

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
        "aveFx2CC": "average_face_x_to_cell",
        "aveFy2CC": "average_face_y_to_cell",
        "aveFz2CC": "average_face_z_to_cell",
        "aveEx2CC": "average_edge_x_to_cell",
        "aveEy2CC": "average_edge_y_to_cell",
        "aveEz2CC": "average_edge_z_to_cell",
    }

    ###########################################################################
    #                                                                         #
    #                             Face Divergence                             #
    #                                                                         #
    ###########################################################################

    @property
    def _face_x_divergence_stencil(self):
        """Stencil for face divergence operator in the x-direction (x-faces to cell centers)."""
        raise NotImplementedError("_face_x_divergence_stencil not yet implemented")

    @property
    def _face_y_divergence_stencil(self):
        """Stencil for face divergence operator in the y-direction (y-faces to cell centers)."""
        raise NotImplementedError("_face_y_divergence_stencil not yet implemented")

    @property
    def _face_z_divergence_stencil(self):
        """Stencil for face divergence operator in the z-direction (z-faces to cell centers)."""
        raise NotImplementedError("_face_z_divergence_stencil not yet implemented")

    @property
    def _face_divergence_stencil(self):
        """Stencil for face divergence operator (all faces to cell centers)."""
        raise NotImplementedError("_face_divergence_stencil not yet implemented")

    @property
    def face_divergence(self):
        """Face divergence operator (faces to cell centers).

        Returns
        -------
        torch.sparse.Tensor
            The divergence operator matrix that maps from faces to cell centers
        """
        raise NotImplementedError("face_divergence not yet implemented")

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
        raise NotImplementedError("face_x_divergence not yet implemented")

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
        raise NotImplementedError("face_y_divergence not yet implemented")

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
        raise NotImplementedError("face_z_divergence not yet implemented")

    ###########################################################################
    #                                                                         #
    #                            Nodal Gradient                              #
    #                                                                         #
    ###########################################################################

    @property
    def _nodal_gradient_x_stencil(self):
        """Stencil for nodal gradient operator in the x-direction."""
        raise NotImplementedError("_nodal_gradient_x_stencil not yet implemented")

    @property
    def _nodal_gradient_y_stencil(self):
        """Stencil for nodal gradient operator in the y-direction."""
        raise NotImplementedError("_nodal_gradient_y_stencil not yet implemented")

    @property
    def _nodal_gradient_z_stencil(self):
        """Stencil for nodal gradient operator in the z-direction."""
        raise NotImplementedError("_nodal_gradient_z_stencil not yet implemented")

    @property
    def _nodal_gradient_stencil(self):
        """Stencil for nodal gradient operator."""
        raise NotImplementedError("_nodal_gradient_stencil not yet implemented")

    @property
    def nodal_gradient(self):
        """Nodal gradient operator (nodes to edges).

        Returns
        -------
        torch.sparse.Tensor
            The gradient operator matrix that maps from nodes to edges
        """
        raise NotImplementedError("nodal_gradient not yet implemented")

    ###########################################################################
    #                                                                         #
    #                            Nodal Laplacian                             #
    #                                                                         #
    ###########################################################################

    @property
    def _nodal_laplacian_x_stencil(self):
        """Stencil for nodal laplacian operator in the x-direction."""
        raise NotImplementedError("_nodal_laplacian_x_stencil not yet implemented")

    @property
    def _nodal_laplacian_y_stencil(self):
        """Stencil for nodal laplacian operator in the y-direction."""
        raise NotImplementedError("_nodal_laplacian_y_stencil not yet implemented")

    @property
    def _nodal_laplacian_z_stencil(self):
        """Stencil for nodal laplacian operator in the z-direction."""
        raise NotImplementedError("_nodal_laplacian_z_stencil not yet implemented")

    @property
    def _nodal_laplacian_x(self):
        """Nodal laplacian operator in the x-direction."""
        raise NotImplementedError("_nodal_laplacian_x not yet implemented")

    @property
    def _nodal_laplacian_y(self):
        """Nodal laplacian operator in the y-direction."""
        raise NotImplementedError("_nodal_laplacian_y not yet implemented")

    @property
    def _nodal_laplacian_z(self):
        """Nodal laplacian operator in the z-direction."""
        raise NotImplementedError("_nodal_laplacian_z not yet implemented")

    @property
    def nodal_laplacian(self):
        """Nodal laplacian operator (nodes to nodes).

        Returns
        -------
        torch.sparse.Tensor
            The laplacian operator matrix that maps from nodes to nodes
        """
        raise NotImplementedError("nodal_laplacian not yet implemented")

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
        raise NotImplementedError("edge_divergence_weak_form_robin not yet implemented")

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
        raise NotImplementedError("set_cell_gradient_BC not yet implemented")

    @property
    def stencil_cell_gradient_x(self):
        """Stencil for cell gradient operator in the x-direction."""
        raise NotImplementedError("stencil_cell_gradient_x not yet implemented")

    @property
    def stencil_cell_gradient_y(self):
        """Stencil for cell gradient operator in the y-direction."""
        raise NotImplementedError("stencil_cell_gradient_y not yet implemented")

    @property
    def stencil_cell_gradient_z(self):
        """Stencil for cell gradient operator in the z-direction."""
        raise NotImplementedError("stencil_cell_gradient_z not yet implemented")

    @property
    def stencil_cell_gradient(self):
        """Stencil for cell gradient operator."""
        raise NotImplementedError("stencil_cell_gradient not yet implemented")

    @property
    def cell_gradient(self):
        """Cell gradient operator (cell centers to faces).

        Returns
        -------
        torch.sparse.Tensor
            The gradient operator matrix that maps from cell centers to faces
        """
        raise NotImplementedError("cell_gradient not yet implemented")

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
        """Cell gradient boundary condition operator."""
        raise NotImplementedError("cell_gradient_BC not yet implemented")

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
        raise NotImplementedError("cell_gradient_x not yet implemented")

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
        raise NotImplementedError("cell_gradient_y not yet implemented")

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
        raise NotImplementedError("cell_gradient_z not yet implemented")

    ###########################################################################
    #                                                                         #
    #                               Edge Curl                                #
    #                                                                         #
    ###########################################################################

    @property
    def _edge_x_curl_stencil(self):
        """Stencil for edge curl operator in the x-direction."""
        raise NotImplementedError("_edge_x_curl_stencil not yet implemented")

    @property
    def _edge_y_curl_stencil(self):
        """Stencil for edge curl operator in the y-direction."""
        raise NotImplementedError("_edge_y_curl_stencil not yet implemented")

    @property
    def _edge_z_curl_stencil(self):
        """Stencil for edge curl operator in the z-direction."""
        raise NotImplementedError("_edge_z_curl_stencil not yet implemented")

    @property
    def _edge_curl_stencil(self):
        """Stencil for edge curl operator."""
        raise NotImplementedError("_edge_curl_stencil not yet implemented")

    @property
    def edge_curl(self):
        """Edge curl operator (edges to faces).

        Returns
        -------
        torch.sparse.Tensor
            The curl operator matrix that maps from edges to faces
        """
        raise NotImplementedError("edge_curl not yet implemented")

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
        raise NotImplementedError("boundary_face_scalar_integral not yet implemented")

    @property
    def boundary_edge_vector_integral(self):
        """Boundary edge vector integral operator.

        Returns
        -------
        torch.sparse.Tensor
            Operator for computing vector integrals over boundary edges
        """
        raise NotImplementedError("boundary_edge_vector_integral not yet implemented")

    @property
    def boundary_node_vector_integral(self):
        """Boundary node vector integral operator.

        Returns
        -------
        torch.sparse.Tensor
            Operator for computing vector integrals over boundary nodes
        """
        raise NotImplementedError("boundary_node_vector_integral not yet implemented")

    ###########################################################################
    #                                                                         #
    #                         Boundary Conditions                            #
    #                                                                         #
    ###########################################################################

    def get_BC_projections(self, BC, discretization="CC"):
        """Get boundary condition projection matrices.

        Parameters
        ----------
        BC : dict
            Boundary condition specification
        discretization : str, optional
            Discretization type, by default "CC"

        Returns
        -------
        tuple
            Boundary condition projection matrices
        """
        raise NotImplementedError("get_BC_projections not yet implemented")

    def get_BC_projections_simple(self, discretization="CC"):
        """Get simplified boundary condition projection matrices.

        Parameters
        ----------
        discretization : str, optional
            Discretization type, by default "CC"

        Returns
        -------
        tuple
            Simplified boundary condition projection matrices
        """
        raise NotImplementedError("get_BC_projections_simple not yet implemented")

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
        raise NotImplementedError("average_face_to_cell not yet implemented")

    @property
    def average_face_to_cell_vector(self):
        """Vector averaging operator from faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages face vector values to cell centers
        """
        raise NotImplementedError("average_face_to_cell_vector not yet implemented")

    @property
    def average_face_x_to_cell(self):
        """Averaging operator from x-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages x-face values to cell centers
        """
        raise NotImplementedError("average_face_x_to_cell not yet implemented")

    @property
    def average_face_y_to_cell(self):
        """Averaging operator from y-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages y-face values to cell centers
        """
        raise NotImplementedError("average_face_y_to_cell not yet implemented")

    @property
    def average_face_z_to_cell(self):
        """Averaging operator from z-faces to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages z-face values to cell centers
        """
        raise NotImplementedError("average_face_z_to_cell not yet implemented")

    @property
    def average_cell_to_face(self):
        """Averaging operator from cell centers to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to faces
        """
        raise NotImplementedError("average_cell_to_face not yet implemented")

    @property
    def average_cell_vector_to_face(self):
        """Vector averaging operator from cell centers to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center vector values to faces
        """
        raise NotImplementedError("average_cell_vector_to_face not yet implemented")

    @property
    def average_cell_to_edge(self):
        """Averaging operator from cell centers to edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages cell center values to edges
        """
        raise NotImplementedError("average_cell_to_edge not yet implemented")

    @property
    def average_edge_to_cell(self):
        """Averaging operator from edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge values to cell centers
        """
        raise NotImplementedError("average_edge_to_cell not yet implemented")

    @property
    def average_edge_to_cell_vector(self):
        """Vector averaging operator from edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge vector values to cell centers
        """
        raise NotImplementedError("average_edge_to_cell_vector not yet implemented")

    @property
    def average_edge_x_to_cell(self):
        """Averaging operator from x-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages x-edge values to cell centers
        """
        raise NotImplementedError("average_edge_x_to_cell not yet implemented")

    @property
    def average_edge_y_to_cell(self):
        """Averaging operator from y-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages y-edge values to cell centers
        """
        raise NotImplementedError("average_edge_y_to_cell not yet implemented")

    @property
    def average_edge_z_to_cell(self):
        """Averaging operator from z-edges to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages z-edge values to cell centers
        """
        raise NotImplementedError("average_edge_z_to_cell not yet implemented")

    @property
    def average_edge_to_face(self):
        """Averaging operator from edges to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages edge values to faces
        """
        raise NotImplementedError("average_edge_to_face not yet implemented")

    @property
    def average_node_to_cell(self):
        """Averaging operator from nodes to cell centers.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to cell centers
        """
        raise NotImplementedError("average_node_to_cell not yet implemented")

    @property
    def _average_node_to_edge_x(self):
        """Internal averaging operator from nodes to x-edges."""
        raise NotImplementedError("_average_node_to_edge_x not yet implemented")

    @property
    def _average_node_to_edge_y(self):
        """Internal averaging operator from nodes to y-edges."""
        raise NotImplementedError("_average_node_to_edge_y not yet implemented")

    @property
    def _average_node_to_edge_z(self):
        """Internal averaging operator from nodes to z-edges."""
        raise NotImplementedError("_average_node_to_edge_z not yet implemented")

    @property
    def average_node_to_edge(self):
        """Averaging operator from nodes to edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to edges
        """
        raise NotImplementedError("average_node_to_edge not yet implemented")

    @property
    def _average_node_to_face_x(self):
        """Internal averaging operator from nodes to x-faces."""
        raise NotImplementedError("_average_node_to_face_x not yet implemented")

    @property
    def _average_node_to_face_y(self):
        """Internal averaging operator from nodes to y-faces."""
        raise NotImplementedError("_average_node_to_face_y not yet implemented")

    @property
    def _average_node_to_face_z(self):
        """Internal averaging operator from nodes to z-faces."""
        raise NotImplementedError("_average_node_to_face_z not yet implemented")

    @property
    def average_node_to_face(self):
        """Averaging operator from nodes to faces.

        Returns
        -------
        torch.sparse.Tensor
            Operator that averages node values to faces
        """
        raise NotImplementedError("average_node_to_face not yet implemented")

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
        raise NotImplementedError("project_face_to_boundary_face not yet implemented")

    @property
    def project_edge_to_boundary_edge(self):
        """Projection operator from edges to boundary edges.

        Returns
        -------
        torch.sparse.Tensor
            Operator that projects edge values to boundary edges
        """
        raise NotImplementedError("project_edge_to_boundary_edge not yet implemented")

    @property
    def project_node_to_boundary_node(self):
        """Projection operator from nodes to boundary nodes.

        Returns
        -------
        torch.sparse.Tensor
            Operator that projects node values to boundary nodes
        """
        raise NotImplementedError("project_node_to_boundary_node not yet implemented")
