from simpegtorch.discretize.base import BaseMesh
from simpegtorch.discretize.utils import (
    sub2ind,
    sdiag,
    kron,
    speye,
    inverse_property_tensor,
    TensorType,
    make_property_tensor,
    ndgrid,
    # inverse_2x2_block_diagonal,
    # get_subarray,
    # inverse_3x3_block_diagonal,
    sdinv,
    mkvc,
    is_scalar,
)

import torch


class InnerProducts(BaseMesh):
    """Class for constructing inner product matrices.

    ``InnerProducts`` is a mixin class for constructing inner product matrices,
    their inverses and their derivatives with respect to model parameters.
    The ``InnerProducts`` class is inherited by all ``discretize`` mesh classes.
    In practice, we don't create instances of the ``InnerProducts`` class in
    order to construct inner product matrices, their inverses or their derivatives.
    These quantities are instead constructed from instances of ``discretize``
    meshes using the appropriate method.
    """

    def get_face_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
    ):
        return self._getInnerProduct(
            "F",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def get_edge_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
    ):
        return self._getInnerProduct(
            "E",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def _getInnerProduct(
        self,
        projection_type,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=False,
    ):
        """Get the inner product matrix.

        Parameters
        ----------
        str : projection_type
            'F' for faces 'E' for edges
        torch.tensor : model
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
        bool : invert_model
            inverts the material property
        bool : invert_matrix
            inverts the matrix
        bool : do_fast
            do a faster implementation if available.

        Returns
        -------
        torch.tensor
            Inner product matrix of shape (nF, nF) for faces or (nE, nE) for edges.
        """

        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        fast = None
        if hasattr(self, "_fastInnerProduct") and do_fast:
            fast = self._fastInnerProduct(
                projection_type,
                model=model,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )
        if fast is not None:
            return fast

        if invert_model:
            model = inverse_property_tensor(self, model)

        tensorType = TensorType(self, model)

        Mu = make_property_tensor(self, model, sparse_type="coo", device=self.device)
        # Uses COO format for the inner product matrix as CSR does not support batch operations or additions
        # This is platform dependent, so we use COO format for consistency
        Ps = self._getInnerProductProjectionMatrices(projection_type, tensorType)
        # Manually sum sparse tensors since torch.sum doesn't work on lists of sparse tensors
        terms = [torch.sparse.mm(torch.sparse.mm(P.T, Mu), P) for P in Ps]
        A = terms[0]
        for term in terms[1:]:
            A = A + term

        if invert_matrix and tensorType < 3:
            A = sdinv(A)
        elif invert_matrix and tensorType == 3:
            raise Exception("Solver needed to invert A.")

        return A

    def _getInnerProductProjectionMatrices(self, projection_type, tensorType):
        """Get the inner product projection matrices.

        Parameters
        ----------
        projection_type : str
            'F' for faces 'E' for edges
        tensorType : TensorType
            type of the tensor: TensorType(mesh, sigma)

        Returns
        -------
        torch.tensor
            batch of inner product projection matrices of shape (nF, nC, nC) for faces
            or (nE, nC, nC) for edges.
        """
        if not isinstance(tensorType, TensorType):
            raise TypeError("tensorType must be an instance of TensorType.")
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = kron(
            speye(d),
            sdiag(torch.sqrt((2 ** (-d)) * self.cell_volumes)),
            sparse_type="coo",
        ).to(self.device)

        nodes = ["000", "100", "010", "110", "001", "101", "011", "111"][: 2**d]

        if projection_type == "F":
            locs = {
                "000": [("fXm",), ("fXm", "fYm"), ("fXm", "fYm", "fZm")],
                "100": [("fXp",), ("fXp", "fYm"), ("fXp", "fYm", "fZm")],
                "010": [None, ("fXm", "fYp"), ("fXm", "fYp", "fZm")],
                "110": [None, ("fXp", "fYp"), ("fXp", "fYp", "fZm")],
                "001": [None, None, ("fXm", "fYm", "fZp")],
                "101": [None, None, ("fXp", "fYm", "fZp")],
                "011": [None, None, ("fXm", "fYp", "fZp")],
                "111": [None, None, ("fXp", "fYp", "fZp")],
            }
            proj = getattr(self, "_getFaceP" + ("x" * d))()

        elif projection_type == "E":
            locs = {
                "000": [("eX0",), ("eX0", "eY0"), ("eX0", "eY0", "eZ0")],
                "100": [("eX0",), ("eX0", "eY1"), ("eX0", "eY1", "eZ1")],
                "010": [None, ("eX1", "eY0"), ("eX1", "eY0", "eZ2")],
                "110": [None, ("eX1", "eY1"), ("eX1", "eY1", "eZ3")],
                "001": [None, None, ("eX2", "eY2", "eZ0")],
                "101": [None, None, ("eX2", "eY3", "eZ1")],
                "011": [None, None, ("eX3", "eY2", "eZ2")],
                "111": [None, None, ("eX3", "eY3", "eZ3")],
            }
            proj = getattr(self, "_getEdgeP" + ("x" * d))()

        return [torch.sparse.mm(V, proj(*locs[node][d - 1])) for node in nodes]

    def _getFacePx(M):
        """Return a function for creating face projection matrices in 1D."""
        ii = torch.arange(M.shape_cells[0], device=M.device, dtype=torch.long)

        def Px(xFace):
            # xFace is 'fXp' or 'fXm'
            posFx = 0 if xFace == "fXm" else 1
            IND = ii + posFx
            PX = torch.sparse_coo_tensor(
                torch.vstack(
                    [torch.arange(M.n_cells, device=M.device, dtype=torch.long), IND]
                ),
                torch.ones(M.n_cells, device=M.device, dtype=M.dtype),
                (M.n_cells, M.n_faces),
                device=M.device,
            )
            return PX

        return Px

    def _getFacePxx(M):
        """Return a function for creating face projection matrices in 2D."""
        # returns a function for creating projection matrices
        #
        # Mats takes you from faces a subset of all faces on only the
        # faces that you ask for.
        #
        # These are centered around a single nodes.
        #
        # For example, if this was your entire mesh:
        #
        #                 f3(Yp)
        #           2_______________3
        #           |               |
        #           |               |
        #           |               |
        #   f0(Xm)  |       x       |  f1(Xp)
        #           |               |
        #           |               |
        #           |_______________|
        #           0               1
        #                 f2(Ym)
        #
        # Pxx('fXm','fYm') = | 1, 0, 0, 0 |
        #                    | 0, 0, 1, 0 |
        #
        # Pxx('fXp','fYm') = | 0, 1, 0, 0 |
        #                    | 0, 0, 1, 0 |
        i, j = torch.arange(M.shape_cells[0], device=M.device), torch.arange(
            M.shape_cells[1], device=M.device
        )

        iijj = ndgrid(i, j, vector=True, device=M.device)
        ii, jj = iijj[:, 0], iijj[:, 1]

        def Pxx(xFace, yFace):
            # xFace is 'fXp' or 'fXm'
            # yFace is 'fYp' or 'fYm'

            # no | node      | f1     | f2
            # 00 | i  ,j     | i  , j | i, j
            # 10 | i+1,j     | i+1, j | i, j
            # 01 | i  ,j+1   | i  , j | i, j+1
            # 11 | i+1,j+1   | i+1, j | i, j+1

            posFx = 0 if xFace == "fXm" else 1
            posFy = 0 if yFace == "fYm" else 1

            ind1 = sub2ind(M.vnFx, torch.column_stack([ii + posFx, jj]))
            ind2 = sub2ind(M.vnFy, torch.column_stack([ii, jj + posFy])) + M.nFx

            IND = torch.cat([ind1, ind2])

            PXX = torch.sparse_coo_tensor(
                torch.vstack(
                    [
                        torch.arange(2 * M.n_cells, device=M.device, dtype=torch.long),
                        IND,
                    ]
                ),
                torch.ones(2 * M.n_cells, device=M.device, dtype=M.dtype),
                (2 * M.n_cells, M.n_faces),
                device=M.device,
            )

            return PXX

        return Pxx

    def _getFacePxxx(M):
        """Return a function for creating face projection matrices in 3D.

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.
        """
        i, j, k = (
            torch.arange(M.shape_cells[0], device=M.device),
            torch.arange(M.shape_cells[1], device=M.device),
            torch.arange(M.shape_cells[2], device=M.device),
        )

        iijjkk = ndgrid(i, j, k, vector=True, device=M.device)
        ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

        def Pxxx(xFace, yFace, zFace):
            # xFace is 'fXp' or 'fXm'
            # yFace is 'fYp' or 'fYm'
            # zFace is 'fZp' or 'fZm'

            # no  | node        | f1        | f2        | f3
            # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
            # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
            # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
            # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
            # 001 | i  ,j  ,k+1 | i  , j, k | i, j  , k | i, j, k+1
            # 101 | i+1,j  ,k+1 | i+1, j, k | i, j  , k | i, j, k+1
            # 011 | i  ,j+1,k+1 | i  , j, k | i, j+1, k | i, j, k+1
            # 111 | i+1,j+1,k+1 | i+1, j, k | i, j+1, k | i, j, k+1

            posX = 0 if xFace == "fXm" else 1
            posY = 0 if yFace == "fYm" else 1
            posZ = 0 if zFace == "fZm" else 1

            ind1 = sub2ind(M.vnFx, torch.column_stack([ii + posX, jj, kk]))
            ind2 = sub2ind(M.vnFy, torch.column_stack([ii, jj + posY, kk])) + M.nFx
            ind3 = (
                sub2ind(M.vnFz, torch.column_stack([ii, jj, kk + posZ])) + M.nFx + M.nFy
            )

            IND = torch.cat([ind1, ind2, ind3])

            PXXX = torch.sparse_coo_tensor(
                torch.vstack(
                    [
                        torch.arange(3 * M.n_cells, device=M.device, dtype=torch.long),
                        IND,
                    ]
                ),
                torch.ones(3 * M.n_cells, device=M.device, dtype=M.dtype),
                (3 * M.n_cells, M.n_faces),
                device=M.device,
            )

            return PXXX

        return Pxxx

    def get_edge_inner_product_surface(
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
    ):
        """Generate the edge inner product surface matrix or its inverse.

        This method generates the inner product surface matrix (or its inverse)
        when discrete variables are defined on mesh edges. It constructs
        the inner product surface matrix when diagnostic properties
        (e.g. conductance) are defined on mesh edges.

        Parameters
        ----------
        model : None or torch.Tensor
            Parameters defining the diagnostic properties for every edge in the mesh.
            Inner product surface matrices can be constructed for the following cases:

            - *None* : returns the basic inner product surface matrix
            - *(n_edges)* torch.Tensor : returns inner product surface matrix
              for an isotropic model. The array contains a scalar diagnostic property value
              for each edge.

        invert_model : bool, optional
            The inverse of *model* is used as the diagnostic property.
        invert_matrix : bool, optional
            Returns the inverse of the inner product surface matrix.

        Returns
        -------
        torch.sparse.COOTensor
            Inner product surface matrix of shape (n_edges, n_edges)
        """

        edge_lengths = self.edge_lengths
        if model is None:
            model = torch.ones(self.n_edges, device=self.device, dtype=self.dtype)

        if invert_model:
            model = 1.0 / model

        if is_scalar(model):
            model = model * torch.ones(
                self.n_edges, device=self.device, dtype=self.dtype
            )

        # Isotropic case only
        if model.numel() != self.n_edges:
            raise ValueError(
                f"Unexpected shape of tensor: {model.shape}. "
                f"Must be scalar or have length equal to total number of edges ({self.n_edges})."
            )

        M = sdiag(edge_lengths * mkvc(model), sparse_type="coo")

        if invert_matrix:
            return sdinv(M)
        else:
            return M

    def _getEdgePx(M):
        """Return a function for creating edge projection matrices in 1D."""

        def Px(xEdge):
            if xEdge != "eX0":
                raise TypeError("xEdge = {0!s}, not eX0".format(xEdge))
            return speye(M.n_cells, device=M.device, dtype=M.dtype, sparse_type="coo")

        return Px

    def _getEdgePxx(M):
        """Return a function for creating edge projection matrices in 2D."""
        i, j = torch.arange(M.shape_cells[0], device=M.device), torch.arange(
            M.shape_cells[1], device=M.device
        )

        iijj = ndgrid(i, j, vector=True, device=M.device)
        ii, jj = iijj[:, 0], iijj[:, 1]

        def Pxx(xEdge, yEdge):
            # no | node      | e1      | e2
            # 00 | i  ,j     | i  ,j   | i  ,j
            # 10 | i+1,j     | i  ,j   | i+1,j
            # 01 | i  ,j+1   | i  ,j+1 | i  ,j
            # 11 | i+1,j+1   | i  ,j+1 | i+1,j
            posX = 0 if xEdge == "eX0" else 1
            posY = 0 if yEdge == "eY0" else 1

            ind1 = sub2ind(M.vnEx, torch.column_stack([ii, jj + posX]))
            ind2 = sub2ind(M.vnEy, torch.column_stack([ii + posY, jj])) + M.nEx

            IND = torch.cat([ind1, ind2])

            PXX = torch.sparse_coo_tensor(
                torch.vstack(
                    [
                        torch.arange(2 * M.n_cells, device=M.device, dtype=torch.long),
                        IND,
                    ]
                ),
                torch.ones(2 * M.n_cells, device=M.device, dtype=M.dtype),
                (2 * M.n_cells, M.n_edges),
                device=M.device,
            )

            return PXX

        return Pxx

    def _getEdgePxxx(M):
        """Return a function for creating edge projection matrices in 3D."""
        i, j, k = (
            torch.arange(M.shape_cells[0], device=M.device),
            torch.arange(M.shape_cells[1], device=M.device),
            torch.arange(M.shape_cells[2], device=M.device),
        )

        iijjkk = ndgrid(i, j, k, vector=True, device=M.device)
        ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

        def Pxxx(xEdge, yEdge, zEdge):
            # no  | node        | e1          | e2          | e3
            # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
            # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
            # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
            # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
            # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
            # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
            # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
            # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k

            posX = (
                [0, 0]
                if xEdge == "eX0"
                else [1, 0] if xEdge == "eX1" else [0, 1] if xEdge == "eX2" else [1, 1]
            )
            posY = (
                [0, 0]
                if yEdge == "eY0"
                else [1, 0] if yEdge == "eY1" else [0, 1] if yEdge == "eY2" else [1, 1]
            )
            posZ = (
                [0, 0]
                if zEdge == "eZ0"
                else [1, 0] if zEdge == "eZ1" else [0, 1] if zEdge == "eZ2" else [1, 1]
            )

            ind1 = sub2ind(M.vnEx, torch.column_stack([ii, jj + posX[0], kk + posX[1]]))
            ind2 = (
                sub2ind(M.vnEy, torch.column_stack([ii + posY[0], jj, kk + posY[1]]))
                + M.nEx
            )
            ind3 = (
                sub2ind(M.vnEz, torch.column_stack([ii + posZ[0], jj + posZ[1], kk]))
                + M.nEx
                + M.nEy
            )

            IND = torch.cat([ind1, ind2, ind3])

            PXXX = torch.sparse_coo_tensor(
                torch.vstack(
                    [
                        torch.arange(3 * M.n_cells, device=M.device, dtype=torch.long),
                        IND,
                    ]
                ),
                torch.ones(3 * M.n_cells, device=M.device, dtype=M.dtype),
                (3 * M.n_cells, M.n_edges),
                device=M.device,
            )

            return PXXX

        return Pxxx
