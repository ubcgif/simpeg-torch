"""
Electromagnetic utilities for simpeg-torch.

This module provides utility functions for electromagnetic simulations,
including DC/IP survey generation, data processing, and visualization.
"""

from .static_utils import (
    generate_dcip_sources_line,
    pseudo_locations,
    apparent_resistivity_from_voltage,
    geometric_factor,
    convert_survey_3d_to_2d_lines,
    plot_pseudosection,
    electrode_separations,
)

__all__ = [
    "generate_dcip_sources_line",
    "pseudo_locations",
    "apparent_resistivity_from_voltage",
    "geometric_factor",
    "convert_survey_3d_to_2d_lines",
    "plot_pseudosection",
    "electrode_separations",
]
