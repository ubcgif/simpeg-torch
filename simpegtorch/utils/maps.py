"""
Simple mapping utilities for simpeg-torch.

This module provides basic mapping functionality for transforming model parameters.
For more advanced mapping capabilities, refer to the full SimPEG implementation.
"""

import torch
import numpy as np
from typing import Union


class InjectActiveCells:
    """
    Simple mapping for injecting active cell values into full mesh.

    This mapping takes a model defined on active cells and expands it to the full mesh,
    filling inactive cells with a specified value.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object
    active_cells : torch.Tensor
        Boolean tensor indicating active cells
    valInactive : float, default=0.0
        Value to assign to inactive cells

    Examples
    --------
    >>> import torch
    >>> from simpegtorch.discretize import TensorMesh
    >>> from simpegtorch.utils.maps import InjectActiveCells
    >>>
    >>> # Create mesh and active cells
    >>> h = torch.ones(10) * 20.0
    >>> mesh = TensorMesh([h, h, h])
    >>> active_cells = mesh.cell_centers[:, 2] < 0  # Below z=0
    >>>
    >>> # Create mapping
    >>> mapping = InjectActiveCells(mesh, active_cells, valInactive=1e8)
    >>>
    >>> # Create model on active cells
    >>> n_active = torch.sum(active_cells)
    >>> active_model = torch.ones(n_active) * 100.0  # 100 ohm-m
    >>>
    >>> # Map to full mesh
    >>> full_model = mapping * active_model
    >>> print(f"Full model shape: {full_model.shape}")
    """

    def __init__(self, mesh, active_cells: torch.Tensor, valInactive: float = 0.0):
        self.mesh = mesh
        self.active_cells = active_cells.bool()  # Ensure boolean
        self.valInactive = valInactive
        self.n_active = torch.sum(self.active_cells).item()

        print(
            f"InjectActiveCells mapping: {self.n_active} active out of {mesh.nC} total cells"
        )

    def __mul__(self, active_model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Apply the mapping: active model -> full model.

        Parameters
        ----------
        active_model : torch.Tensor or numpy.ndarray
            Model values on active cells

        Returns
        -------
        torch.Tensor
            Model values on full mesh
        """
        return self.forward(active_model)

    def forward(self, active_model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Apply the mapping: active model -> full model.

        Parameters
        ----------
        active_model : torch.Tensor or numpy.ndarray
            Model values on active cells

        Returns
        -------
        torch.Tensor
            Model values on full mesh
        """
        # Convert to torch tensor if needed
        if not isinstance(active_model, torch.Tensor):
            active_model = torch.tensor(active_model, dtype=torch.float64)

        # Validate input size
        if active_model.numel() != self.n_active:
            raise ValueError(
                f"Active model size ({active_model.numel()}) must match "
                f"number of active cells ({self.n_active})"
            )

        # Create full model array
        full_model = torch.full(
            (self.mesh.nC,),
            self.valInactive,
            dtype=torch.float64,
            device=active_model.device,
        )

        # Inject active values
        full_model[self.active_cells] = active_model.flatten()

        return full_model

    def inverse(self, full_model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Apply the inverse mapping: full model -> active model.

        Parameters
        ----------
        full_model : torch.Tensor or numpy.ndarray
            Model values on full mesh

        Returns
        -------
        torch.Tensor
            Model values on active cells only
        """
        # Convert to torch tensor if needed
        if not isinstance(full_model, torch.Tensor):
            full_model = torch.tensor(full_model, dtype=torch.float64)

        # Validate input size
        if full_model.numel() != self.mesh.nC:
            raise ValueError(
                f"Full model size ({full_model.numel()}) must match "
                f"total number of cells ({self.mesh.nC})"
            )

        # Extract active values
        active_model = full_model.flatten()[self.active_cells]

        return active_model


class IdentityMap:
    """
    Identity mapping that returns the input unchanged.

    Useful as a default mapping when no transformation is needed.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object (kept for compatibility)

    Examples
    --------
    >>> mapping = IdentityMap(mesh)
    >>> model_out = mapping * model_in  # model_out == model_in
    """

    def __init__(self, mesh):
        self.mesh = mesh

    def __mul__(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the identity mapping."""
        return self.forward(model)

    def forward(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the identity mapping."""
        if not isinstance(model, torch.Tensor):
            model = torch.tensor(model, dtype=torch.float64)
        return model

    def inverse(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the inverse identity mapping (same as forward)."""
        return self.forward(model)


class ExpMap:
    """
    Exponential mapping: m_transformed = exp(m_input).

    Note: This is provided for compatibility, but torch.exp() can be used directly
    with automatic differentiation.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object (kept for compatibility)

    Examples
    --------
    >>> mapping = ExpMap(mesh)
    >>> log_model = torch.ones(mesh.nC) * torch.log(torch.tensor(100.0))  # log(100)
    >>> linear_model = mapping * log_model  # Should give ~100
    """

    def __init__(self, mesh):
        self.mesh = mesh
        print(
            "Note: ExpMap is provided for compatibility. Consider using torch.exp() directly."
        )

    def __mul__(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the exponential mapping."""
        return self.forward(model)

    def forward(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the exponential mapping."""
        if not isinstance(model, torch.Tensor):
            model = torch.tensor(model, dtype=torch.float64)
        return torch.exp(model)

    def inverse(self, model: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply the inverse exponential mapping (logarithm)."""
        if not isinstance(model, torch.Tensor):
            model = torch.tensor(model, dtype=torch.float64)
        return torch.log(model)
