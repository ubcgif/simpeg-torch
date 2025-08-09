"""
Model mapping classes for SimPEG-Torch

This module provides various model parameter transformations using PyTorch modules
with automatic differentiation support.
"""

import torch


class LogMapping(torch.nn.Module):
    """
    Log mapping using PyTorch autograd.

    This creates a PyTorch module that maps from log-space parameters to linear space
    using automatic differentiation.


    Examples
    --------
    >>> # Simple log mapping
    >>> log_map = LogMapping()
    >>> log_params = torch.tensor([2.0, 3.0, 4.0])  # ln(values)
    >>> linear_params = log_map(log_params)  # exp(log_params)
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_params):
        """
        Transform log-parameters to linear space

        Parameters
        ----------
        log_params : torch.Tensor
            Log-space parameter values

        Returns
        -------
        torch.Tensor
            Transformed parameter values (linear space), optionally mapped
        """
        # Transform from log to linear space (differentiable)
        return torch.exp(log_params)

    def inverse(self, linear_params):
        """
        Inverse of the transform taking parameters from log space to linear space
        Parameters
        ----------
        linear_params : torch.Tensor
            Linear-space parameter values
        Returns
        -------
        torch.Tensor
            Transformed parameter values (log space)
        """
        return torch.log(linear_params)


class LinearMapping(torch.nn.Module):
    """
    Linear (identity) mapping using PyTorch autograd.

    This is an identity mapping that passes parameters through unchanged.
    Can be used with or without an additional mapping for consistency with
    other mapping classes.

    Parameters
    ----------
    mapping : torch.nn.Module, optional
        Additional mapping to apply after the identity transformation.
        If None, parameters pass through unchanged.

    Examples
    --------
    >>> # Simple identity mapping
    >>> linear_map = LinearMapping()
    >>> params = torch.tensor([10.0, 100.0, 1000.0])
    >>> output = linear_map(params)  # Same as params

    >>> # Linear mapping with active cell injection
    >>> from simpegtorch.utils import InjectActiveCells
    >>> active_mapping = InjectActiveCells(mesh, active_cells, valInactive=1e8)
    >>> linear_resist_map = LinearMapping(active_mapping)
    >>> resistivity = torch.tensor([10.0, 100.0, 1000.0])  # resistivity on active cells
    >>> full_resistivity = linear_resist_map(resistivity)  # resistivity on full mesh
    """

    def __init__(self):
        super().__init__()

    def forward(self, params):
        """
        Pass parameters through and apply additional mapping if provided.

        Parameters
        ----------
        params : torch.Tensor
            Parameter values

        Returns
        -------
        torch.Tensor
            Parameter values, optionally mapped
        """
        return params.clone()  # Return a copy to avoid in-place operations


# Legacy aliases for backward compatibility
LogResistivityMapping = LogMapping
