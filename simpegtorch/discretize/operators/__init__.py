"""
================================================
Discrete Operators (:mod:`discretize.operators`)
================================================
.. currentmodule:: discretize.operators

The ``operators`` package contains the classes discretize meshes with regular structure
use to construct discrete versions of the differential operators.

Operator Classes
----------------
.. autosummary::
  :toctree: generated/

  DiffOperators
  InnerProducts
"""

from .inner_products import InnerProducts
from .differential_operators import DiffOperators
