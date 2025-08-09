"""
Topography utilities for simpeg-torch.

This module provides topography handling functions including proper interpolation
for complex 3D surfaces based on the original SimPEG implementation.
"""

import torch
import numpy as np
from typing import Union
from scipy.spatial import Delaunay, cKDTree
from scipy import interpolate


def active_from_xyz(
    mesh,
    surface_xyz: Union[torch.Tensor, np.ndarray],
    grid_reference="CC",
    method="linear",
):
    """
    Return boolean array indicating which cells are below surface.

    For a set of locations defining a surface, active_from_xyz outputs a
    boolean array indicating which mesh cells lie below the surface points.
    This method uses SciPy's interpolation routine to interpolate between
    location points defining the surface. Nearest neighbour interpolation
    is used for cells outside the convex hull of the surface points.

    Parameters
    ----------
    mesh : BaseMesh
        The mesh object
    surface_xyz : torch.Tensor or numpy.ndarray
        Points defining the surface topography with shape (n_points, 3)
    grid_reference : {'CC', 'N'}, default='CC'
        Define where the cell is defined relative to surface.

        - If 'CC' is used, cells are active if their centers are below the surface.
        - If 'N' is used, cells are active if they lie entirely below the surface.
    method : {'linear', 'nearest'}, default='linear'
        Interpolation method for locations between the xyz points.

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
    >>> # Define complex topography
    >>> x = torch.linspace(-100, 100, 21)
    >>> y = torch.linspace(-100, 100, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> Z = 50 * torch.sin(X/100) * torch.cos(Y/100)  # Complex surface
    >>> surface_xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    >>>
    >>> # Get active cells (below surface)
    >>> active = active_from_xyz(mesh, surface_xyz)
    >>> print(f"Active cells: {torch.sum(active)} out of {mesh.nC}")
    """

    # Validate inputs
    if grid_reference not in ["N", "CC"]:
        raise ValueError(
            "Value of grid_reference must be 'N' (nodal) or 'CC' (cell center)"
        )

    # Convert surface_xyz to numpy for scipy operations
    if isinstance(surface_xyz, torch.Tensor):
        xyz = surface_xyz.detach().cpu().numpy()
    else:
        xyz = np.array(surface_xyz)

    dim = mesh.dim - 1

    # Setup interpolation based on mesh dimensions
    if mesh.dim == 3:
        if xyz.shape[1] != 3:
            raise ValueError("xyz locations of shape (*, 3) required for 3D mesh")
        if method == "linear":
            tri2D = Delaunay(xyz[:, :2])
            z_interpolate = interpolate.LinearNDInterpolator(tri2D, xyz[:, 2])
        else:
            z_interpolate = interpolate.NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
    elif mesh.dim == 2:
        if xyz.shape[1] != 2:
            raise ValueError("xyz locations of shape (*, 2) required for 2D mesh")
        z_interpolate = interpolate.interp1d(
            xyz[:, 0], xyz[:, 1], bounds_error=False, fill_value=np.nan, kind=method
        )
    else:
        if xyz.ndim != 1:
            raise ValueError("xyz locations of shape (*, ) required for 1D mesh")

    # Get evaluation locations based on grid reference
    if grid_reference == "CC":
        # Use cell centers
        locations = mesh.cell_centers.detach().cpu().numpy()

        if mesh.dim == 1:
            active = np.zeros(mesh.nC, dtype=bool)
            cell_centers_x = mesh.cell_centers[:, 0].detach().cpu().numpy()
            active[np.searchsorted(cell_centers_x, xyz).max() :] = True
            return torch.tensor(
                active, dtype=torch.bool, device=mesh.cell_centers.device
            )

    elif grid_reference == "N":
        # Use corner nodes - cells active if ALL corners are below surface
        if mesh.dim == 3:
            # Get the 4 upper corners of each cell
            cell_centers = mesh.cell_centers.detach().cpu().numpy()
            h_gridded = mesh.h_gridded.detach().cpu().numpy()
            locations = np.vstack(
                [
                    cell_centers
                    + (np.c_[-1, 1, 1][:, None] * h_gridded / 2.0).squeeze(),
                    cell_centers
                    + (np.c_[-1, -1, 1][:, None] * h_gridded / 2.0).squeeze(),
                    cell_centers
                    + (np.c_[1, 1, 1][:, None] * h_gridded / 2.0).squeeze(),
                    cell_centers
                    + (np.c_[1, -1, 1][:, None] * h_gridded / 2.0).squeeze(),
                ]
            )
        elif mesh.dim == 2:
            # Get the 2 upper corners of each cell
            cell_centers = mesh.cell_centers.detach().cpu().numpy()
            h_gridded = mesh.h_gridded.detach().cpu().numpy()
            locations = np.vstack(
                [
                    cell_centers + (np.c_[-1, 1][:, None] * h_gridded / 2.0).squeeze(),
                    cell_centers + (np.c_[1, 1][:, None] * h_gridded / 2.0).squeeze(),
                ]
            )
        else:
            active = np.zeros(mesh.nC, dtype=bool)
            nodes_x = mesh.nodes_x.detach().cpu().numpy()
            active[np.searchsorted(nodes_x, xyz).max() :] = True
            return torch.tensor(
                active, dtype=torch.bool, device=mesh.cell_centers.device
            )

    # Interpolate z values at evaluation locations
    z_xyz = z_interpolate(locations[:, :-1]).squeeze()

    # Handle extrapolation using nearest neighbor
    ind_nan = np.isnan(z_xyz)
    if np.any(ind_nan):
        tree = cKDTree(xyz)
        _, ind = tree.query(locations[ind_nan, :])
        z_xyz[ind_nan] = xyz[ind, dim]

    # Determine active cells
    if grid_reference == "CC":
        # Cell centers below surface
        active = locations[:, dim] < z_xyz
    else:
        # All corner points below surface (for 'N' reference)
        active = np.all(
            (locations[:, dim] < z_xyz).reshape((mesh.nC, -1), order="F"), axis=1
        )

    # Print summary
    n_active = np.sum(active)
    surface_min = xyz[:, dim].min()
    surface_max = xyz[:, dim].max()
    surface_mean = xyz[:, dim].mean()
    print(
        f"Surface elevation range: {surface_min:.2f} to {surface_max:.2f} m (mean: {surface_mean:.2f} m)"
    )
    print(f"Found {n_active} active cells out of {mesh.nC} total cells")

    # Convert back to torch tensor with appropriate device
    return torch.tensor(
        active.ravel(), dtype=torch.bool, device=mesh.cell_centers.device
    )


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
