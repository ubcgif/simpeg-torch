"""
========================================================
Utility Classes and Functions
========================================================

The ``utils`` package contains utilities for helping with common operations involving
discrete meshes

Utility Classes
===============
.. autosummary::
  :toctree: generated/

  TensorType
  Zero
  Identity

Utility Functions
=================

Code Utilities
--------------
.. autosummary::
  :toctree: generated/

  is_scalar
  as_array_n_by_dim
  requires

Coordinate Transform Utilities
------------------------------
.. autosummary::
  :toctree: generated/

  rotate_points_from_normals
  rotation_matrix_from_normals
  cylindrical_to_cartesian
  cartesian_to_cylindrical

Interpolation Utilities
-----------------------
.. autosummary::
  :toctree: generated/

  interpolation_matrix
  volume_average

IO utilities
------------
.. autosummary::
  :toctree: generated/

  load_mesh
  download

Matrix Utilities
----------------
.. autosummary::
  :toctree: generated/

  mkvc
  sdiag
  sdinv
  speye
  kron3
  spzeros
  ddx
  av
  av_extrap
  ndgrid
  ind2sub
  sub2ind
  get_subarray
  inverse_3x3_block_diagonal
  inverse_2x2_block_diagonal
  invert_blocks
  make_property_tensor
  inverse_property_tensor
  cross2d

Mesh Utilities
--------------
.. autosummary::
  :toctree: generated/

  unpack_widths
  closest_points_index
  extract_core_mesh
  random_model
  refine_tree_xyz
  active_from_xyz
  mesh_builder_xyz

Utilities for Curvilinear Meshes
--------------------------------
.. autosummary::
  :toctree: generated/

  example_curvilinear_grid
  volume_tetrahedron
  face_info
  index_cube
"""

from .code_utils import (
    is_scalar,
    atleast_1d,
    as_array_n_by_dim,
    fortran_ravel_index,
)  # , requires
from .matrix_utils import (
    mkvc,
    sdiag,
    sdinv,
    speye,
    spzeros,
    kron,
    reshape_fortran,
    kron3,
    ddx,
    av,
    av_extrap,
    ndgrid,
    make_boundary_bool,
    ind2sub,
    sub2ind,
    get_diag,
    get_subarray,
    inverse_3x3_block_diagonal,
    inverse_2x2_block_diagonal,
    TensorType,
    make_property_tensor,
    inverse_property_tensor,
    Zero,
    Identity,
    torch_blockdiag,
    cross2d,
)
from .mesh_utils import (
    unpack_widths,
    #     closest_points_index,
    #     extract_core_mesh,
    #     random_model,
    #     refine_tree_xyz,
    #     active_from_xyz,
    #     mesh_builder_xyz,
    #     example_simplex_mesh,
)

# from .curvilinear_utils import (
#     example_curvilinear_grid,
#     volume_tetrahedron,
#     face_info,
#     index_cube,
# )
from .interpolation_utils import interpolation_matrix, volume_average

# from .coordinate_utils import (
#     rotate_points_from_normals,
#     rotation_matrix_from_normals,
#     cyl2cart,
#     cart2cyl,
#     cylindrical_to_cartesian,
#     cartesian_to_cylindrical,
#     # rotate_vec_cyl2cart
# )

# from .io_utils import download, load_mesh
