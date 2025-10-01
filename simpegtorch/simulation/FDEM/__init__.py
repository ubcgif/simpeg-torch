"""
FDEM (Frequency Domain Electromagnetics) forward modeling module for simpeg-torch.

This module provides classes for FDEM forward modeling using PyTorch tensors
for automatic differentiation and GPU acceleration.

Classes
-------
PDE Formulations (New Framework)
    FDEM3DMagneticFluxDensity - 3D FDEM PDE formulation for magnetic flux density

Sources
    BaseFDEMSrc - Base source class
    MagneticDipole - Magnetic dipole source
    ElectricDipole - Electric dipole source
    LoopSource - Loop source (wire loop)

Receivers
    BaseFDEMReceiver - Base receiver class
    PointMagneticFluxDensity - Point receiver for magnetic flux density
    PointElectricField - Point receiver for electric field

Survey
    Survey - Container for sources and receivers with frequency management

Usage
-----
Use FDEM3DMagneticFluxDensity with DirectSolver:

    >>> from simpegtorch.simulation.FDEM import FDEM3DMagneticFluxDensity, Survey
    >>> from simpegtorch.simulation.base import DirectSolver, mappings
    >>> pde = FDEM3DMagneticFluxDensity(mesh, survey, mapping)
    >>> solver = DirectSolver(pde)
    >>> data = solver.forward()
"""

from .fdem_pde import FDEM3DMagneticFluxDensity
from .sources import BaseFDEMSrc, MagneticDipole, ElectricDipole, LoopSource
from .receivers import (
    BaseFDEMReceiver,
    PointMagneticFluxDensity as RxPointMagneticFluxDensity,
    PointElectricField as RxPointElectricField,
)
from .survey import Survey

# Aliases for common usage patterns
Src = BaseFDEMSrc
Rx = BaseFDEMReceiver

__all__ = [
    # PDE formulation
    "FDEM3DMagneticFluxDensity",
    # Sources
    "BaseFDEMSrc",
    "Src",
    "MagneticDipole",
    "ElectricDipole",
    "LoopSource",
    # Receivers
    "BaseFDEMReceiver",
    "Rx",
    "RxPointMagneticFluxDensity",
    "RxPointElectricField",
    # Survey
    "Survey",
]
