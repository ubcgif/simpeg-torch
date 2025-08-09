"""
==================================
Discretize
==================================

The ``base`` sub-package houses the fundamental classes for all meshes in ``discretize``.

Base Mesh Class
---------------
.. autosummary::
  :toctree: generated/

  TensorMesh

Mixins
------
.. autosummary::
  :toctree: generated/

  InterfaceMPL
"""

from .tensor_mesh import TensorMesh
from .tensor_cell import TensorCell
from .mixins import InterfaceMPL
