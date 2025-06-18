"""
Model building utilities for simpeg-torch.

This module provides utility functions for creating synthetic models
and extracting indices for geometric shapes.
"""

import torch
import numpy as np
from typing import Union


def get_indices_sphere(
    center: Union[torch.Tensor, np.ndarray, list],
    radius: float,
    cell_centers: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """
    Get boolean indices for cells whose centers lie inside a sphere.

    Parameters
    ----------
    center : torch.Tensor, numpy.ndarray, or list
        Location of the center of the sphere. Should be (dim,) shaped.
    radius : float
        Radius of the sphere
    cell_centers : torch.Tensor or numpy.ndarray
        Cell center locations with shape (n_cells, dim). Can also be a mesh object
        with a cell_centers attribute.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_cells,) indicating which cells are inside the sphere

    Examples
    --------
    Create a simple 2D example:

    >>> import torch
    >>> from simpegtorch.utils import get_indices_sphere
    >>>
    >>> # Define cell centers in a 3x3 grid
    >>> x = torch.linspace(-1, 1, 3)
    >>> y = torch.linspace(-1, 1, 3)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> cell_centers = torch.stack([X.flatten(), Y.flatten()], dim=1)
    >>>
    >>> # Find cells within radius 0.8 of origin
    >>> center = torch.tensor([0.0, 0.0])
    >>> radius = 0.8
    >>> mask = get_indices_sphere(center, radius, cell_centers)
    >>> print(f"Cells inside sphere: {torch.sum(mask)} out of {len(mask)}")

    For 3D case:

    >>> # 3D cell centers
    >>> cell_centers_3d = torch.randn(100, 3)  # 100 random cell centers
    >>> center_3d = torch.tensor([0.0, 0.0, 0.0])
    >>> radius_3d = 1.0
    >>> mask_3d = get_indices_sphere(center_3d, radius_3d, cell_centers_3d)
    """

    # Handle mesh objects that have cell_centers attribute
    if hasattr(cell_centers, "cell_centers"):
        cell_centers = cell_centers.cell_centers

    # Convert inputs to torch tensors
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float64)
    if not isinstance(cell_centers, torch.Tensor):
        cell_centers = torch.tensor(cell_centers, dtype=torch.float64)

    # Ensure center is the right shape
    if center.dim() == 0:
        center = center.unsqueeze(0)

    # Validation: mesh and center live in the same dimensional space
    dim_mesh = cell_centers.shape[1]
    dim_center = center.shape[0]

    if dim_center != dim_mesh:
        raise ValueError(
            f"Dimension mismatch: center has dimension {dim_center} "
            f"but cell_centers has dimension {dim_mesh}"
        )

    # Compute distances from center to all cell centers
    # Broadcasting: (n_cells, dim) - (dim,) -> (n_cells, dim)
    distances = torch.norm(cell_centers - center, dim=1)

    # Return boolean mask for cells within the sphere
    return distances < radius


def get_indices_box(
    center: Union[torch.Tensor, np.ndarray, list],
    dimensions: Union[torch.Tensor, np.ndarray, list],
    cell_centers: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """
    Get boolean indices for cells whose centers lie inside a box.

    Parameters
    ----------
    center : torch.Tensor, numpy.ndarray, or list
        Location of the center of the box. Should be (dim,) shaped.
    dimensions : torch.Tensor, numpy.ndarray, or list
        Half-widths of the box in each dimension. Should be (dim,) shaped.
    cell_centers : torch.Tensor or numpy.ndarray
        Cell center locations with shape (n_cells, dim).

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_cells,) indicating which cells are inside the box

    Examples
    --------
    >>> import torch
    >>> from simpegtorch.utils.model_builder import get_indices_box
    >>>
    >>> # 2D cell centers
    >>> cell_centers = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> center = torch.tensor([1.0, 1.0])
    >>> dimensions = torch.tensor([0.8, 0.8])  # ±0.8 in each direction
    >>> mask = get_indices_box(center, dimensions, cell_centers)
    """

    # Handle mesh objects that have cell_centers attribute
    if hasattr(cell_centers, "cell_centers"):
        cell_centers = cell_centers.cell_centers

    # Convert inputs to torch tensors
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float64)
    if not isinstance(dimensions, torch.Tensor):
        dimensions = torch.tensor(dimensions, dtype=torch.float64)
    if not isinstance(cell_centers, torch.Tensor):
        cell_centers = torch.tensor(cell_centers, dtype=torch.float64)

    # Ensure proper shapes
    if center.dim() == 0:
        center = center.unsqueeze(0)
    if dimensions.dim() == 0:
        dimensions = dimensions.unsqueeze(0)

    # Validation
    dim_mesh = cell_centers.shape[1]
    if center.shape[0] != dim_mesh or dimensions.shape[0] != dim_mesh:
        raise ValueError(
            "Dimension mismatch between center, dimensions, and cell_centers"
        )

    # Check if each cell is within the box bounds
    lower_bounds = center - dimensions
    upper_bounds = center + dimensions

    # Check each dimension
    within_bounds = torch.all(
        (cell_centers >= lower_bounds) & (cell_centers <= upper_bounds), dim=1
    )

    return within_bounds


def create_block_in_wholespace(
    cell_centers: Union[torch.Tensor, np.ndarray],
    block_center: Union[torch.Tensor, np.ndarray, list],
    block_dimensions: Union[torch.Tensor, np.ndarray, list],
    background_value: float = 1.0,
    block_value: float = 10.0,
) -> torch.Tensor:
    """
    Create a simple block model in a wholespace.

    Parameters
    ----------
    cell_centers : torch.Tensor or numpy.ndarray
        Cell center locations with shape (n_cells, dim)
    block_center : torch.Tensor, numpy.ndarray, or list
        Center location of the block
    block_dimensions : torch.Tensor, numpy.ndarray, or list
        Half-widths of the block in each dimension
    background_value : float, default=1.0
        Physical property value for the background
    block_value : float, default=10.0
        Physical property value for the block

    Returns
    -------
    torch.Tensor
        Physical property model with shape (n_cells,)

    Examples
    --------
    >>> import torch
    >>> from simpegtorch.utils.model_builder import create_block_in_wholespace
    >>>
    >>> # Create a simple 2D grid
    >>> x = torch.linspace(-2, 2, 21)
    >>> y = torch.linspace(-2, 2, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> cell_centers = torch.stack([X.flatten(), Y.flatten()], dim=1)
    >>>
    >>> # Create block model
    >>> model = create_block_in_wholespace(
    ...     cell_centers,
    ...     block_center=[0.0, 0.0],
    ...     block_dimensions=[0.5, 0.5],
    ...     background_value=1.0,
    ...     block_value=10.0
    ... )
    """

    # Handle mesh objects
    if hasattr(cell_centers, "cell_centers"):
        cell_centers = cell_centers.cell_centers

    if not isinstance(cell_centers, torch.Tensor):
        cell_centers = torch.tensor(cell_centers, dtype=torch.float64)

    n_cells = cell_centers.shape[0]

    # Create background model
    model = torch.full((n_cells,), background_value, dtype=torch.float64)

    # Get indices for the block
    block_mask = get_indices_box(block_center, block_dimensions, cell_centers)

    # Set block values
    model[block_mask] = block_value

    return model


def create_sphere_in_wholespace(
    cell_centers: Union[torch.Tensor, np.ndarray],
    sphere_center: Union[torch.Tensor, np.ndarray, list],
    sphere_radius: float,
    background_value: float = 1.0,
    sphere_value: float = 10.0,
) -> torch.Tensor:
    """
    Create a simple spherical anomaly model in a wholespace.

    Parameters
    ----------
    cell_centers : torch.Tensor or numpy.ndarray
        Cell center locations with shape (n_cells, dim)
    sphere_center : torch.Tensor, numpy.ndarray, or list
        Center location of the sphere
    sphere_radius : float
        Radius of the sphere
    background_value : float, default=1.0
        Physical property value for the background
    sphere_value : float, default=10.0
        Physical property value for the sphere

    Returns
    -------
    torch.Tensor
        Physical property model with shape (n_cells,)
    """

    # Handle mesh objects
    if hasattr(cell_centers, "cell_centers"):
        cell_centers = cell_centers.cell_centers

    if not isinstance(cell_centers, torch.Tensor):
        cell_centers = torch.tensor(cell_centers, dtype=torch.float64)

    n_cells = cell_centers.shape[0]

    # Create background model
    model = torch.full((n_cells,), background_value, dtype=torch.float64)

    # Get indices for the sphere
    sphere_mask = get_indices_sphere(sphere_center, sphere_radius, cell_centers)

    # Set sphere values
    model[sphere_mask] = sphere_value

    return model
