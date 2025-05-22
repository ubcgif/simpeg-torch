from discretize.base import BaseMesh
from discretize.utils import (
    # sub2ind,
    sdiag,
    # kron,
    inverse_property_tensor,
    TensorType,
    make_property_tensor,
    # ndgrid,
    # inverse_2x2_block_diagonal,
    # get_subarray,
    # inverse_3x3_block_diagonal,
    # spzeros,
    sdinv,
    # mkvc,
    # is_scalar,
)

# import torch
import torch.sparse as sp
import numpy as np


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

    def _getInnerProduct(
        self,
        projection_type,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs,
    ):
        """Get the inner product matrix.

        Parameters
        ----------
        str : projection_type
            'F' for faces 'E' for edges
        numpy.ndarray : model
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
        bool : invert_model
            inverts the material property
        bool : invert_matrix
            inverts the matrix
        bool : do_fast
            do a faster implementation if available.

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the inner product matrix (nE, nE)
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

        Mu = make_property_tensor(self, model)
        Ps = self._getInnerProductProjectionMatrices(projection_type, tensorType)
        A = np.sum([P.T * Mu * P for P in Ps])

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
        scipy.sparse.csr_matrix
        """
        if not isinstance(tensorType, TensorType):
            raise TypeError("tensorType must be an instance of TensorType.")
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = sp.kron(sp.identity(d), sdiag(np.sqrt((2 ** (-d)) * self.cell_volumes)))

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

        return [V * proj(*locs[node][d - 1]) for node in nodes]
