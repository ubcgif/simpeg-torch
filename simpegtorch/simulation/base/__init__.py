from .universal_simulation import UniversalSimulation
from .dc_pde import DCResistivityPDE
from .fdem_pde import FDEMPDE
from ..basePDE import BasePDE, BaseMapping

__all__ = [
    "UniversalSimulation",
    "DCResistivityPDE",
    "FDEMPDE",
    "BasePDE",
    "BaseMapping",
]
