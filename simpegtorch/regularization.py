"""
PyTorch-based regularization framework for SimPEG-Torch

This module provides PyTorch implementations of regularization functions using
automatic differentiation for geophysical inversions.
"""

from typing import Optional
import torch
import torch.nn as nn
from abc import abstractmethod


class BaseRegularization(nn.Module):
    """
    Base regularization class using PyTorch autograd.

    All regularization functions inherit from this class and implement
    the forward() method to compute the regularization objective.

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object with differential operators
    mapping : callable
        Mapping function that outputs the model parameters.
    reference_model : torch.Tensor, optional
        Reference model for regularization (in mesh space)
    device : str, optional
        PyTorch device ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for computations
    """

    def __init__(
        self,
        mesh,
        mapping,
        reference_model: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.mesh = mesh
        self.device = device
        self.dtype = dtype
        self.mapping = mapping

        # Convert and register mesh operators as sparse tensors
        self._setup_operators()

        # Get number of cells in mesh space
        n_cells = mesh.nC if hasattr(mesh, "nC") else len(mesh.cell_centers)

        # Set reference model as a non-learned parameter (in mesh space)
        if reference_model is not None:
            self.register_buffer(
                "reference_model", reference_model.to(dtype=dtype, device=device)
            )
        else:
            self.register_buffer(
                "reference_model", torch.zeros(n_cells, dtype=dtype, device=device)
            )

    def _setup_operators(self):
        """Setup mesh differential operators as PyTorch sparse tensors"""
        # Cell gradient operators (already PyTorch sparse tensors)
        if hasattr(self.mesh, "cell_gradient_x"):
            self.register_buffer("grad_x", self.mesh.cell_gradient_x)
        if hasattr(self.mesh, "cell_gradient_y") and self.mesh.dim >= 2:
            self.register_buffer("grad_y", self.mesh.cell_gradient_y)
        if hasattr(self.mesh, "cell_gradient_z") and self.mesh.dim >= 3:
            self.register_buffer("grad_z", self.mesh.cell_gradient_z)

        # Volume weighting
        if hasattr(self.mesh, "cell_volumes"):
            vol_sqrt = torch.sqrt(
                self.mesh.cell_volumes.to(dtype=self.dtype, device=self.device)
            )
            self.register_buffer("vol_sqrt", vol_sqrt)

    def _delta_m(self) -> torch.Tensor:
        """
        Compute model difference from reference: m_mapped - m_ref

        Returns
        -------
        torch.Tensor
            Mapped model difference from reference (in mesh space)
        """
        # Apply mapping if provided (e.g., active cell mapping or log mapping)

        model = self.mapping.forward()

        return model - self.reference_model

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Compute regularization objective function"""
        pass


class Smallness(BaseRegularization):
    """
    Smallness regularization: φ_s = ||W(m - m_ref)||²

    Penalizes deviations from the reference model.

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object
    alpha : float, optional
        Regularization parameter (default: 1.0)
    weights : torch.Tensor, optional
        Custom weighting (default: volume weighting)
    **kwargs
        Additional arguments passed to BaseRegularization
    """

    def __init__(
        self, mesh, alpha: float = 1.0, weights: Optional[torch.Tensor] = None, **kwargs
    ):
        super().__init__(mesh, **kwargs)
        self.alpha = alpha

        # Set up weighting
        if weights is not None:
            self.register_buffer(
                "weights", weights.to(dtype=self.dtype, device=self.device)
            )
        else:
            # Use volume weighting by default
            if hasattr(self.mesh, "cell_volumes"):
                vol_sqrt = torch.sqrt(
                    self.mesh.cell_volumes.to(dtype=self.dtype, device=self.device)
                )
                self.register_buffer("weights", vol_sqrt)
            else:
                # Fallback to uniform weighting
                self.register_buffer(
                    "weights",
                    torch.ones(self.mesh.nC, dtype=self.dtype, device=self.device),
                )

    def forward(self) -> torch.Tensor:
        """
        Compute smallness regularization: α × ||W(m - m_ref)||²

        Returns
        -------
        torch.Tensor
            Regularization objective value
        """
        dm = self._delta_m()

        weighted_dm = self.weights * dm
        return self.alpha * torch.sum(weighted_dm**2)


class SmoothnessFirstOrder(BaseRegularization):
    """
    First-order smoothness regularization: φ_x = ||W∇_x m||²

    Penalizes gradients in the specified direction.

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object
    orientation : str
        Direction for smoothness ('x', 'y', or 'z')
    alpha : float, optional
        Regularization parameter (default: 1.0)
    reference_model_in_smooth : bool, optional
        Include reference model in smoothness (default: False)
    **kwargs
        Additional arguments passed to BaseRegularization
    """

    def __init__(
        self,
        mesh,
        orientation: str = "x",
        alpha: float = 1.0,
        reference_model_in_smooth: bool = False,
        **kwargs,
    ):
        super().__init__(mesh, **kwargs)
        self.orientation = orientation
        self.alpha = alpha
        self.reference_model_in_smooth = reference_model_in_smooth

        # Get gradient operator for specified direction
        if orientation == "x":
            self.grad_op = self.grad_x
        elif orientation == "y" and hasattr(self, "grad_y"):
            self.grad_op = self.grad_y
        elif orientation == "z" and hasattr(self, "grad_z"):
            self.grad_op = self.grad_z
        else:
            raise ValueError(
                f"Invalid orientation '{orientation}' or operator not available"
            )

        # Face volume weights (projected to faces)
        if hasattr(self.mesh, "average_cell_to_face") and hasattr(
            self.mesh, "cell_volumes"
        ):
            # Use proper face volumes - average cell volumes to faces
            mesh_vol = self.mesh.cell_volumes.to(dtype=self.dtype, device=self.device)
            face_vols = torch.sparse.mm(
                self.mesh.average_cell_to_face, mesh_vol.unsqueeze(1)
            ).squeeze()

            # Select the appropriate face volumes for this orientation
            if orientation == "x":
                n_faces_dir = self.mesh.n_faces_x
                face_vols_dir = face_vols[:n_faces_dir]
            elif orientation == "y" and self.mesh.dim >= 2:
                n_faces_x = self.mesh.n_faces_x
                n_faces_dir = self.mesh.n_faces_y
                face_vols_dir = face_vols[n_faces_x : n_faces_x + n_faces_dir]
            elif orientation == "z" and self.mesh.dim >= 3:
                n_faces_x = self.mesh.n_faces_x
                n_faces_y = self.mesh.n_faces_y if self.mesh.dim >= 2 else 0
                n_faces_dir = self.mesh.n_faces_z
                face_vols_dir = face_vols[n_faces_x + n_faces_y :]
            else:
                # Fallback for invalid orientation
                n_faces = self.grad_op.shape[0]
                face_vols_dir = torch.ones(
                    n_faces, dtype=self.dtype, device=self.device
                )

            self.register_buffer("face_weights", torch.sqrt(face_vols_dir))
        else:
            # Fallback to uniform weighting
            n_faces = self.grad_op.shape[0]
            self.register_buffer(
                "face_weights",
                torch.ones(n_faces, dtype=self.dtype, device=self.device),
            )

    def forward(self) -> torch.Tensor:
        """
        Compute first-order smoothness: α × ||W∇_dir m||²

        Returns
        -------
        torch.Tensor
            Regularization objective value
        """
        if self.reference_model_in_smooth:
            m_smooth = self._delta_m()
        else:
            # Apply mapping to transform model to mesh space
            m_smooth = self.mapping.forward()

        # Apply gradient operator: ∇m
        grad_m = torch.sparse.mm(self.grad_op, m_smooth.unsqueeze(1)).squeeze()

        # Apply weighting and compute norm
        weighted_grad = self.face_weights * grad_m
        return self.alpha * torch.sum(weighted_grad**2)


class SmoothnessSecondOrder(BaseRegularization):
    """
    Second-order smoothness regularization: φ_xx = ||W∇²_x m||²

    Penalizes curvature (second derivatives) in the specified direction.

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object
    orientation : str
        Direction for smoothness ('x', 'y', or 'z')
    alpha : float, optional
        Regularization parameter (default: 1.0)
    reference_model_in_smooth : bool, optional
        Include reference model in smoothness (default: False)
    **kwargs
        Additional arguments passed to BaseRegularization
    """

    def __init__(
        self,
        mesh,
        orientation: str = "x",
        alpha: float = 1.0,
        reference_model_in_smooth: bool = False,
        **kwargs,
    ):
        super().__init__(mesh, **kwargs)
        self.orientation = orientation
        self.alpha = alpha
        self.reference_model_in_smooth = reference_model_in_smooth

        # Get gradient operator
        if orientation == "x":
            self.grad_op = self.grad_x
        elif orientation == "y" and hasattr(self, "grad_y"):
            self.grad_op = self.grad_y
        elif orientation == "z" and hasattr(self, "grad_z"):
            self.grad_op = self.grad_z
        else:
            raise ValueError(
                f"Invalid orientation '{orientation}' or operator not available"
            )

        # Cell volume weights
        if hasattr(self.mesh, "vol"):
            vol_sqrt = torch.sqrt(
                self.mesh.vol.to(dtype=self.dtype, device=self.device)
            )
            self.register_buffer("cell_weights", vol_sqrt)
        else:
            # Fallback to uniform weighting
            n_cells = self.mesh.nC
            self.register_buffer(
                "cell_weights",
                torch.ones(n_cells, dtype=self.dtype, device=self.device),
            )

    def forward(self) -> torch.Tensor:
        """
        Compute second-order smoothness: α × ||W∇²_dir m||²

        Returns
        -------
        torch.Tensor
            Regularization objective value
        """
        if self.reference_model_in_smooth:
            m_smooth = self._delta_m()
        else:
            # Apply mapping to transform model to mesh space
            m_smooth = self.mapping.forward()

        # Apply gradient operator twice: ∇²m = ∇ᵀ∇m
        grad_m = torch.sparse.mm(self.grad_op, m_smooth.unsqueeze(1)).squeeze()
        laplace_m = torch.sparse.mm(self.grad_op.t(), grad_m.unsqueeze(1)).squeeze()

        # Apply weighting and compute norm
        weighted_laplace = self.cell_weights * laplace_m
        return self.alpha * torch.sum(weighted_laplace**2)


class WeightedLeastSquares(nn.Module):
    """
    Combined weighted least-squares regularization.

    Implements: φ_m = α_s φ_s + α_x φ_x + α_y φ_y + α_z φ_z + α_xx φ_xx + α_yy φ_yy + α_zz φ_zz

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object
    mapping : callable, optional
        Mapping function that transforms input model to mesh model space.
        If None, identity mapping is used.
    alpha_s : float, optional
        Smallness weight (default: 1.0)
    alpha_x, alpha_y, alpha_z : float, optional
        First-order smoothness weights (default: 1.0)
    alpha_xx, alpha_yy, alpha_zz : float, optional
        Second-order smoothness weights (default: 0.0)
    reference_model_in_smooth : bool, optional
        Include reference model in smoothness terms (default: False)
    **kwargs
        Additional arguments passed to individual regularization terms
    """

    def __init__(
        self,
        mesh,
        mapping: Optional[callable] = None,
        alpha_s: float = 1.0,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        alpha_z: float = 1.0,
        alpha_xx: float = 0.0,
        alpha_yy: float = 0.0,
        alpha_zz: float = 0.0,
        reference_model_in_smooth: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mesh = mesh

        # Store alpha values
        self.alpha_s = alpha_s
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z
        self.alpha_xx = alpha_xx
        self.alpha_yy = alpha_yy
        self.alpha_zz = alpha_zz

        # Create regularization terms
        self.terms = nn.ModuleList()

        # Smallness term
        if alpha_s > 0:
            self.terms.append(
                Smallness(
                    mesh,
                    alpha=alpha_s,
                    mapping=mapping,
                    **kwargs,
                )
            )

        # First-order smoothness terms
        if alpha_x > 0:
            self.terms.append(
                SmoothnessFirstOrder(
                    mesh,
                    orientation="x",
                    alpha=alpha_x,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )
        if alpha_y > 0 and mesh.dim >= 2:
            self.terms.append(
                SmoothnessFirstOrder(
                    mesh,
                    orientation="y",
                    alpha=alpha_y,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )
        if alpha_z > 0 and mesh.dim >= 3:
            self.terms.append(
                SmoothnessFirstOrder(
                    mesh,
                    orientation="z",
                    alpha=alpha_z,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )

        # Second-order smoothness terms
        if alpha_xx > 0:
            self.terms.append(
                SmoothnessSecondOrder(
                    mesh,
                    orientation="x",
                    alpha=alpha_xx,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )
        if alpha_yy > 0 and mesh.dim >= 2:
            self.terms.append(
                SmoothnessSecondOrder(
                    mesh,
                    orientation="y",
                    alpha=alpha_yy,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )
        if alpha_zz > 0 and mesh.dim >= 3:
            self.terms.append(
                SmoothnessSecondOrder(
                    mesh,
                    orientation="z",
                    alpha=alpha_zz,
                    mapping=mapping,
                    reference_model_in_smooth=reference_model_in_smooth,
                    **kwargs,
                )
            )

    def forward(self) -> torch.Tensor:
        """
        Compute total regularization objective.

        Returns
        -------
        torch.Tensor
            Total regularization objective value
        """
        total = None

        for term in self.terms:
            term_val = term()
            if total is None:
                total = term_val
            else:
                total = total + term_val

        return total if total is not None else torch.tensor(0.0)

    def set_reference_model(self, reference_model: torch.Tensor):
        """Set reference model for all terms"""
        for term in self.terms:
            if hasattr(term, "reference_model"):
                term.reference_model = reference_model.to(
                    dtype=term.reference_model.dtype, device=term.reference_model.device
                )


class Sparse(BaseRegularization):
    """
    Sparse regularization using L0/L1 norms.

    Promotes sparse (blocky) models using iteratively reweighted least squares (IRLS).

    Parameters
    ----------
    mesh : TensorMesh
        Mesh object
    norm : float, optional
        Norm for sparsity (0 < norm <= 2, default: 1.0)
    alpha : float, optional
        Regularization parameter (default: 1.0)
    epsilon : float, optional
        Small value to avoid division by zero in IRLS (default: 1e-8)
    **kwargs
        Additional arguments passed to BaseRegularization
    """

    def __init__(
        self,
        mesh,
        norm: float = 1.0,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(mesh, **kwargs)
        self.norm = norm
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize IRLS weights
        n_cells = mesh.nC if hasattr(mesh, "nC") else len(mesh.cell_centers)
        self.register_buffer(
            "irls_weights", torch.ones(n_cells, dtype=self.dtype, device=self.device)
        )

    def forward(self) -> torch.Tensor:
        """
        Compute sparse regularization with IRLS weighting.

        Returns
        -------
        torch.Tensor
            Regularization objective value
        """
        dm = self._delta_m()

        # Apply IRLS weighting
        weighted_dm = self.irls_weights * dm

        # Compute weighted norm
        if self.norm == 2.0:
            return self.alpha * torch.sum(weighted_dm**2)
        elif self.norm == 1.0:
            return self.alpha * torch.sum(torch.abs(weighted_dm))
        else:
            return self.alpha * torch.sum(torch.abs(weighted_dm) ** self.norm)

    def update_irls_weights(self):
        """Update IRLS weights based on current model"""
        with torch.no_grad():
            dm = self._delta_m()
            weights = (torch.abs(dm) + self.epsilon) ** (self.norm / 2.0 - 1.0)
            self.irls_weights.copy_(weights)
