"""
DC resistivity forward modeling module for simpeg-torch.

This module provides classes for DC resistivity forward modeling using PyTorch tensors
for automatic differentiation and GPU acceleration.

Classes
-------
PDE Formulations (New Framework)
    DC3DCellCentered - 3D cell-centered finite volume DC resistivity PDE
    DC3DNodal - 3D nodal finite difference DC resistivity PDE

Sources
    BaseSrc - Base source class
    Pole - Pole source (single electrode)
    Dipole - Dipole source (A-B electrode pair)
    Multipole - Generic multipole source

Receivers
    BaseRx - Base receiver class
    Pole - Pole receiver (single electrode)
    Dipole - Dipole receiver (M-N electrode pair)

Survey
    Survey - Container for sources and receivers with geometric factors

Usage
-----
Use DC3DCellCentered or DC3DNodal with DirectSolver:

    >>> from simpegtorch.simulation.resistivity import DC3DCellCentered, Survey
    >>> from simpegtorch.simulation.base import DirectSolver, mappings
    >>> pde = DC3DCellCentered(mesh, survey, mapping)
    >>> solver = DirectSolver(pde)
    >>> data = solver.forward()
"""

from .dc_pde import DC3DCellCentered, DC3DNodal
from .dc_2D_pde import DC2DCellCentered
from .sources import BaseSrc, Pole as SrcPole, Dipole as SrcDipole, Multipole
from .receivers import BaseRx, Pole as RxPole, Dipole as RxDipole
from .survey import Survey

# Aliases for common usage patterns
Src = BaseSrc
Rx = BaseRx

__all__ = [
    # PDE formulations
    "DC3DCellCentered",
    "DC3DNodal",
    "DC2DCellCentered",
    # Sources
    "BaseSrc",
    "Src",
    "SrcPole",
    "SrcDipole",
    "Multipole",
    # Receivers
    "BaseRx",
    "Rx",
    "RxPole",
    "RxDipole",
    # Survey
    "Survey",
]
