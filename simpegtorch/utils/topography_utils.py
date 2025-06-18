"""
Simple topography utilities for simpeg-torch.

This module provides basic topography handling functions for flat/simple geometries.
For more complex topography, refer to the full SimPEG implementation.
"""

import torch
import numpy as np
from typing import Union


def active_from_xyz(
    mesh, surface_xyz: Union[torch.Tensor, np.ndarray], method="linear"
):
    """
    Get active cells from surface topography for flat surfaces.

    This is a simplified implementation that works with flat topography.
    For complex topography, use the full SimPEG implementation.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object
    surface_xyz : torch.Tensor or numpy.ndarray
        Surface points with shape (n_points, 3) defining the topography.
        For flat topography, all z-values should be approximately the same.
    method : str, default="linear"
        Interpolation method (only "linear" supported in this simplified version)

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_cells,) indicating which cells are active (below surface)

    Examples
    --------
    >>> import torch
    >>> from simpegtorch.discretize import TensorMesh
    >>> from simpegtorch.utils import active_from_xyz
    >>>
    >>> # Create a simple mesh
    >>> h = torch.ones(10) * 20.0
    >>> mesh = TensorMesh([h, h, h], x0=torch.tensor([-100, -100, -100]))
    >>>
    >>> # Define flat topography at z=0
    >>> x = torch.linspace(-100, 100, 11)
    >>> y = torch.linspace(-100, 100, 11)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> Z = torch.zeros_like(X)  # Flat at z=0
    >>> surface_xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    >>>
    >>> # Get active cells (below surface)
    >>> active = active_from_xyz(mesh, surface_xyz)
    >>> print(f"Active cells: {torch.sum(active)} out of {mesh.nC}")
    """

    # Convert inputs to torch tensors
    if not isinstance(surface_xyz, torch.Tensor):
        surface_xyz = torch.tensor(surface_xyz, dtype=torch.float64)

    # Get mesh cell centers
    cell_centers = mesh.cell_centers

    # For flat topography, find the surface elevation
    # This simplified approach assumes approximately flat surface
    surface_z = torch.mean(surface_xyz[:, 2])

    print(f"Surface elevation: {surface_z:.2f} m")

    # Cells are active if their centers are below the surface
    active_cells = cell_centers[:, 2] < surface_z

    n_active = torch.sum(active_cells).item()
    print(f"Found {n_active} active cells out of {mesh.nC} total cells")

    return active_cells


def drape_electrodes_on_topography(
    survey, surface_xyz: Union[torch.Tensor, np.ndarray], method="linear"
):
    """
    Drape electrode locations onto flat topography.

    This is a simplified implementation for flat surfaces.

    Parameters
    ----------
    survey : Survey
        The survey object containing electrode locations
    surface_xyz : torch.Tensor or numpy.ndarray
        Surface points defining the topography
    method : str, default="linear"
        Interpolation method (only "linear" supported)

    Returns
    -------
    Survey
        Survey object with updated electrode locations

    Examples
    --------
    >>> # For flat topography, this function updates electrode z-coordinates
    >>> # to match the surface elevation
    >>> draped_survey = drape_electrodes_on_topography(survey, surface_xyz)
    """

    # Convert inputs to torch tensors
    if not isinstance(surface_xyz, torch.Tensor):
        surface_xyz = torch.tensor(surface_xyz, dtype=torch.float64)

    # For flat topography, use mean surface elevation
    surface_z = torch.mean(surface_xyz[:, 2])

    print(f"Draping electrodes to surface elevation: {surface_z:.2f} m")

    # Update electrode locations for all sources
    for src in survey.source_list:
        # Update source location
        if hasattr(src, "location") and src.location is not None:
            # Convert to tensor if needed and extract scalar values
            if isinstance(src.location, torch.Tensor):
                # Handle different tensor shapes
                if src.location.numel() == 1:
                    # Single value tensor
                    x_val = src.location.item()
                    y_val = 0.0  # Default
                elif src.location.numel() >= 2:
                    # Multi-element tensor
                    x_val = (
                        src.location[0].item()
                        if src.location[0].numel() == 1
                        else src.location[0]
                    )
                    y_val = (
                        src.location[1].item()
                        if src.location[1].numel() == 1
                        else src.location[1]
                    )
                else:
                    continue  # Skip if invalid
            else:
                x_val = src.location[0]
                y_val = src.location[1]

            # Ensure we have scalar values
            if isinstance(x_val, torch.Tensor):
                x_val = x_val.item()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.item()

            src.location = torch.tensor(
                [x_val, y_val, surface_z.item()], dtype=torch.float64
            )

        # Update receiver locations
        for rx in src.receiver_list:
            if hasattr(rx, "locations") and rx.locations is not None:
                # Ensure rx.locations is a tensor
                if not isinstance(rx.locations, torch.Tensor):
                    rx.locations = torch.tensor(rx.locations, dtype=torch.float64)

                rx.locations = torch.column_stack(
                    [
                        rx.locations[:, 0],  # x
                        rx.locations[:, 1],  # y
                        torch.full(
                            (rx.locations.shape[0],),
                            surface_z.item(),
                            dtype=torch.float64,
                        ),  # z
                    ]
                )

    return survey


def create_flat_topography(
    x_extent, y_extent, elevation=0.0, n_points_x=51, n_points_y=51
):
    """
    Create a flat topography surface.

    Parameters
    ----------
    x_extent : tuple
        (x_min, x_max) extent in x direction
    y_extent : tuple
        (y_min, y_max) extent in y direction
    elevation : float, default=0.0
        Constant elevation of the flat surface
    n_points_x : int, default=51
        Number of points in x direction
    n_points_y : int, default=51
        Number of points in y direction

    Returns
    -------
    torch.Tensor
        Surface points with shape (n_points_x * n_points_y, 3)

    Examples
    --------
    >>> topo = create_flat_topography((-2000, 2000), (-2000, 2000), elevation=0.0)
    >>> print(f"Created flat topography with {topo.shape[0]} points")
    """

    x = torch.linspace(x_extent[0], x_extent[1], n_points_x)
    y = torch.linspace(y_extent[0], y_extent[1], n_points_y)

    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = torch.full_like(X, elevation)

    # Stack into (N, 3) array
    surface_xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    return surface_xyz
