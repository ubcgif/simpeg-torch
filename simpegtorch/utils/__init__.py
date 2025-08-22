"""
Utility functions for simpeg-torch.

This module provides general utility functions for model building,
mesh operations, and other common tasks in geophysical modeling.
"""

from .model_builder import (
    get_indices_sphere,
    get_indices_box,
    create_sphere_in_wholespace,
    create_block_in_wholespace,
)
from .topography_utils import (
    active_from_xyz,
    create_flat_topography,
)


__all__ = [
    "get_indices_sphere",
    "get_indices_box",
    "create_sphere_in_wholespace",
    "create_block_in_wholespace",
    "active_from_xyz",
    "create_flat_topography",
    "InjectActiveCells",
    "IdentityMap",
    "ExpMap",
]
