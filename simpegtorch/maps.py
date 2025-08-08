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
    using automatic differentiation. Can be used with or without an additional
    mapping to inject values into a mesh.

    Parameters
    ----------
    mapping : torch.nn.Module, optional
        Additional mapping to apply after the log transformation.
        If None, only the log transformation is applied.

    Examples
    --------
    >>> # Simple log mapping
    >>> log_map = LogMapping()
    >>> log_params = torch.tensor([2.0, 3.0, 4.0])  # ln(values)
    >>> linear_params = log_map(log_params)  # exp(log_params)

    >>> # Log mapping with active cell injection
    >>> from simpegtorch.utils import InjectActiveCells
    >>> active_mapping = InjectActiveCells(mesh, active_cells, valInactive=1e8)
    >>> log_resist_map = LogMapping(active_mapping)
    >>> log_resistivity = torch.tensor([4.0, 5.0, 6.0])  # log-resistivity on active cells
    >>> full_resistivity = log_resist_map(log_resistivity)  # resistivity on full mesh
    """

    def __init__(self, mapping=None):
        super().__init__()
        self.mapping = mapping

    def forward(self, log_params):
        """
        Transform log-parameters to linear space and apply additional mapping.

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
        linear_params = torch.exp(log_params)

        # Apply additional mapping if provided
        if self.mapping is not None:
            return self.mapping.forward(linear_params)
        else:
            return linear_params


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

    def __init__(self, mapping=None):
        super().__init__()
        self.mapping = mapping

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
        # Apply additional mapping if provided
        if self.mapping is not None:
            return self.mapping.forward(params)
        else:
            return params.clone()  # Return a copy to avoid in-place operations


class SqrtMapping(torch.nn.Module):
    """
    Square root mapping using PyTorch autograd.

    This creates a PyTorch module that maps from squared parameters to linear space
    using automatic differentiation. Useful for ensuring positivity without
    the dynamic range issues of log mapping.

    Parameters
    ----------
    mapping : torch.nn.Module, optional
        Additional mapping to apply after the square root transformation.
        If None, only the square root transformation is applied.

    Examples
    --------
    >>> # Simple sqrt mapping
    >>> sqrt_map = SqrtMapping()
    >>> squared_params = torch.tensor([4.0, 9.0, 16.0])  # squared values
    >>> linear_params = sqrt_map(squared_params)  # sqrt(squared_params)

    >>> # Square root mapping with active cell injection
    >>> from simpegtorch.utils import InjectActiveCells
    >>> active_mapping = InjectActiveCells(mesh, active_cells, valInactive=1e8)
    >>> sqrt_resist_map = SqrtMapping(active_mapping)
    >>> squared_resistivity = torch.tensor([100.0, 10000.0, 1000000.0])  # squared resistivity
    >>> full_resistivity = sqrt_resist_map(squared_resistivity)  # resistivity on full mesh
    """

    def __init__(self, mapping=None):
        super().__init__()
        self.mapping = mapping

    def forward(self, squared_params):
        """
        Transform squared parameters to linear space and apply additional mapping.

        Parameters
        ----------
        squared_params : torch.Tensor
            Squared parameter values

        Returns
        -------
        torch.Tensor
            Square root of parameters, optionally mapped
        """
        # Transform from squared to linear space (differentiable)
        linear_params = torch.sqrt(torch.abs(squared_params))

        # Apply additional mapping if provided
        if self.mapping is not None:
            return self.mapping.forward(linear_params)
        else:
            return linear_params


# Legacy aliases for backward compatibility
LogResistivityMapping = LogMapping
