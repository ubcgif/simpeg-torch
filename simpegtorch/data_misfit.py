"""
PyTorch-based data misfit framework for SimPEG-Torch

This module provides PyTorch implementations of data misfit functions using
automatic differentiation for geophysical inversions.
"""

from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class BaseDataMisfit(nn.Module):
    """
    Base data misfit class using PyTorch autograd.

    Computes the data misfit φ_d = ||W_d(F(m) - d_obs)||² where:
    - F(m) is the forward simulation
    - d_obs are observed data
    - W_d is data weighting matrix

    Parameters
    ----------
    solver : Solver
        Forward simulation object
    data : torch.Tensor
        Observed data vector
    weights : torch.Tensor, optional
        Data weights (default: uniform weighting)
    device : str, optional
        PyTorch device ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for computations
    """

    def __init__(
        self,
        solver,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.solver = solver
        self.device = device
        self.dtype = dtype

        # Register observed data
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=dtype, device=device)
        self.register_buffer("data_obs", data.to(dtype=dtype, device=device))

        # Set up data weights
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.tensor(weights, dtype=dtype, device=device)
            self.register_buffer("weights", weights.to(dtype=dtype, device=device))
        else:
            # Default to uniform weighting
            self.register_buffer("weights", torch.ones_like(self.data_obs))

        self.n_data = len(self.data_obs)

    def forward(self) -> torch.Tensor:
        """
        Compute data misfit: φ_d = ||W_d(F(m) - d_obs)||²

        Parameters
        ----------

        Returns
        -------
        torch.Tensor
            Data misfit value
        """

        # Forward simulation
        data_pred = self.solver.forward()

        # Residual
        residual = data_pred - self.data_obs

        # Weighted misfit
        weighted_residual = self.weights * residual
        return torch.sum(weighted_residual**2)


class L2DataMisfit(BaseDataMisfit):
    """
    L2 (least-squares) data misfit.

    Standard least-squares data fitting: φ_d = ||W_d(F(m) - d_obs)||²

    Inherits all functionality from BaseDataMisfit.
    """

    pass


class L1DataMisfit(BaseDataMisfit):
    """
    L1 (robust) data misfit.

    Robust data fitting using L1 norm: φ_d = ||W_d(F(m) - d_obs)||₁

    More robust to outliers than L2 misfit.
    """

    def forward(self) -> torch.Tensor:
        """
        Compute L1 data misfit: φ_d = ||W_d(F(m) - d_obs)||₁

        Returns
        -------
        torch.Tensor
            L1 data misfit value
        """
        # Forward simulation
        data_pred = self.solver.forward()

        # Residual
        residual = data_pred - self.data_obs

        # Weighted L1 misfit
        weighted_residual = self.weights * residual
        return torch.sum(torch.abs(weighted_residual))


class HuberDataMisfit(BaseDataMisfit):
    """
    Huber data misfit for robust fitting.

    Combines L2 and L1 behavior:
    - L2 for small residuals
    - L1 for large residuals

    Parameters
    ----------
    delta : float, optional
        Threshold for switching between L2 and L1 (default: 1.0)
    **kwargs
        Additional arguments passed to BaseDataMisfit
    """

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def forward(self) -> torch.Tensor:
        """
        Compute Huber data misfit.

        Returns
        -------
        torch.Tensor
            Huber data misfit value
        """
        # Forward simulation
        data_pred = self.solver.forward()

        # Residual
        residual = data_pred - self.data_obs
        weighted_residual = self.weights * residual

        # Huber loss
        abs_residual = torch.abs(weighted_residual)
        huber_loss = torch.where(
            abs_residual <= self.delta,
            0.5 * weighted_residual**2,
            self.delta * (abs_residual - 0.5 * self.delta),
        )

        return torch.sum(huber_loss)
