"""
==================================
Base Mesh
==================================

The ``base`` sub-package houses the fundamental classes for all meshes in ``discretize``.

Base Mesh Class
---------------
.. autosummary::
  :toctree: generated/

  BaseRegularMesh
  BaseRectangularMesh
  BaseTensorMesh
"""

from .base_regular_mesh import BaseRegularMesh, BaseRectangularMesh
from .base_tensor_mesh import BaseTensorMesh
