"""
DC resistivity forward modeling module for simpeg-torch.

This module provides classes for DC resistivity forward modeling using PyTorch tensors
for automatic differentiation and GPU acceleration.

Classes
-------
Simulation
    DCStaticSimulationCellCentered - Cell-centered finite volume DC resistivity simulation

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
"""

from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .sources import BaseSrc, Pole as SrcPole, Dipole as SrcDipole, Multipole
from .receivers import BaseRx, Pole as RxPole, Dipole as RxDipole
from .survey import Survey

# Main simulation class
Simulation = Simulation3DCellCentered

# Aliases for common usage patterns and backward compatibility
Simulation3DCellCentered = Simulation3DCellCentered
Simulation3DNodal = Simulation3DNodal
Src = BaseSrc
Rx = BaseRx

__all__ = [
    # Main simulation
    "Simulation",
    "Simulation3DCellCentered",
    "Simulation3DNodal",
    "Simulation3DCellCentered",
    "Simulation3DNodal",
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
