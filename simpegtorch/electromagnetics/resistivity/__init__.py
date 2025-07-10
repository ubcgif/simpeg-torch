"""
DC resistivity forward modeling module for simpeg-torch.

This module provides classes for DC resistivity forward modeling using PyTorch tensors
for automatic differentiation and GPU acceleration.

Classes
-------
3D Simulations
    Simulation3DCellCentered - 3D cell-centered finite volume DC resistivity simulation
    Simulation3DNodal - 3D nodal finite difference DC resistivity simulation

2D Simulations (2.5D using Fourier transform)
    BaseDCSimulation2D - Base class for 2.5D DC simulations
    Simulation2DCellCentered - 2.5D cell-centered finite volume DC resistivity simulation
    Simulation2DNodal - 2.5D nodal finite difference DC resistivity simulation

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
from .simulation_2d import (
    BaseDCSimulation2D, 
    Simulation2DCellCentered, 
    Simulation2DNodal,
    Simulation2DCellCentred  # UK spelling alias
)
from .sources import BaseSrc, Pole as SrcPole, Dipole as SrcDipole, Multipole
from .receivers import BaseRx, Pole as RxPole, Dipole as RxDipole
from .survey import Survey

# Main simulation class
Simulation = Simulation3DCellCentered

# Aliases for common usage patterns and backward compatibility
Simulation3DCellCentered = Simulation3DCellCentered
Simulation3DNodal = Simulation3DNodal
Simulation2DCellCentered = Simulation2DCellCentered  
Simulation2DNodal = Simulation2DNodal
Src = BaseSrc
Rx = BaseRx

__all__ = [
    # Main simulation
    "Simulation",
    # 3D simulations
    "Simulation3DCellCentered",
    "Simulation3DNodal", 
    # 2D simulations
    "BaseDCSimulation2D",
    "Simulation2DCellCentered",
    "Simulation2DNodal", 
    "Simulation2DCellCentred",  # UK spelling
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
