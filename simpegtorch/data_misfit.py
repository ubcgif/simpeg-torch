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
    simulation : Simulation
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
        simulation,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mapping=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.simulation = simulation
        self.mapping = mapping  # Optional mapping (e.g., log mapping)
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

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute data misfit: φ_d = ||W_d(F(m) - d_obs)||²

        Parameters
        ----------
        model : torch.Tensor
            Model parameters (may be on active cells if mapping is provided)

        Returns
        -------
        torch.Tensor
            Data misfit value
        """
        # Apply mapping if provided (e.g., active cell mapping or log mapping)
        if self.mapping is not None:
            if hasattr(self.mapping, "forward") and callable(self.mapping.forward):
                # PyTorch module or object with forward method
                full_model = self.mapping.forward(model)
            elif callable(self.mapping):
                # Simple callable function
                full_model = self.mapping(model)
            else:
                raise TypeError(
                    f"Mapping must be callable or have forward method, got {type(self.mapping)}"
                )
        else:
            full_model = model

        # Forward simulation - use dpred method like original SimPEG
        if hasattr(self.simulation, "dpred"):
            data_pred = self.simulation.dpred(full_model)
        else:
            # Fallback to calling simulation directly
            data_pred = self.simulation(full_model)

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

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 data misfit: φ_d = ||W_d(F(m) - d_obs)||₁

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            L1 data misfit value
        """
        # Forward simulation
        data_pred = self.simulation(model)

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

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber data misfit.

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            Huber data misfit value
        """
        # Forward simulation
        data_pred = self.simulation(model)

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


class ComboDataMisfit(nn.Module):
    """
    Combination data misfit for joint inversions.

    Combines multiple data misfit terms:
    φ_d = Σᵢ wᵢ φ_dᵢ

    Parameters
    ----------
    misfits : list of BaseDataMisfit
        List of individual data misfit terms
    weights : list of float, optional
        Weights for each misfit term (default: equal weights)
    """

    def __init__(self, misfits: list, weights: Optional[list] = None):
        super().__init__()
        self.misfits = nn.ModuleList(misfits)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(misfits)

        # Total number of data points
        self.n_data = sum(misfit.n_data for misfit in self.misfits)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute combined data misfit.

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            Combined data misfit value
        """
        total_misfit = torch.tensor(0.0, dtype=model.dtype, device=model.device)

        for weight, misfit in zip(self.weights, self.misfits):
            total_misfit = total_misfit + weight * misfit(model)

        return total_misfit


class PseudoHuberDataMisfit(BaseDataMisfit):
    """
    Pseudo-Huber data misfit for smooth robust fitting.

    Smooth approximation to Huber loss that is differentiable everywhere:
    φ(r) = δ²(√(1 + (r/δ)²) - 1)

    Parameters
    ----------
    delta : float, optional
        Smoothing parameter (default: 1.0)
    **kwargs
        Additional arguments passed to BaseDataMisfit
    """

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo-Huber data misfit.

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            Pseudo-Huber data misfit value
        """
        # Forward simulation
        data_pred = self.simulation(model)

        # Residual
        residual = data_pred - self.data_obs
        weighted_residual = self.weights * residual

        # Pseudo-Huber loss
        pseudo_huber = self.delta**2 * (
            torch.sqrt(1 + (weighted_residual / self.delta) ** 2) - 1
        )

        return torch.sum(pseudo_huber)


class WeightedDataMisfit(BaseDataMisfit):
    """
    Data misfit with uncertainty-based weighting.

    Uses data uncertainties to set weights: W_d = 1/σ where σ are standard deviations.

    Parameters
    ----------
    simulation : Simulation
        Forward simulation object
    data : torch.Tensor
        Observed data vector
    uncertainties : torch.Tensor
        Data uncertainties (standard deviations)
    **kwargs
        Additional arguments passed to BaseDataMisfit
    """

    def __init__(
        self, simulation, data: torch.Tensor, uncertainties: torch.Tensor, **kwargs
    ):
        # Convert uncertainties to weights
        if isinstance(uncertainties, np.ndarray):
            uncertainties = torch.tensor(
                uncertainties, dtype=kwargs.get("dtype", torch.float64)
            )

        weights = 1.0 / torch.clamp(uncertainties, min=1e-12)  # Avoid division by zero

        super().__init__(simulation, data, weights=weights, **kwargs)

        # Store uncertainties for reference
        self.register_buffer(
            "uncertainties", uncertainties.to(dtype=self.dtype, device=self.device)
        )


class LogDataMisfit(BaseDataMisfit):
    """
    Logarithmic data misfit for positive data.

    Useful for data that spans many orders of magnitude (e.g., resistivity).
    φ_d = ||W_d(log(F(m)) - log(d_obs))||²

    Parameters
    ----------
    epsilon : float, optional
        Small value added to prevent log(0) (default: 1e-12)
    **kwargs
        Additional arguments passed to BaseDataMisfit
    """

    def __init__(self, epsilon: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        # Convert observed data to log space
        self.data_obs = torch.log(torch.clamp(self.data_obs, min=self.epsilon))

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        """
        Compute logarithmic data misfit.

        Parameters
        ----------
        model : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            Log data misfit value
        """
        # Forward simulation
        data_pred = self.simulation(model)

        # Convert to log space
        log_data_pred = torch.log(torch.clamp(data_pred, min=self.epsilon))

        # Residual in log space
        residual = log_data_pred - self.data_obs

        # Weighted misfit
        weighted_residual = self.weights * residual
        return torch.sum(weighted_residual**2)
