import torch

from .matrix_utils import sub2ind, mkvc


def interpolation_matrix(locs, x, y=None, z=None, dtype=torch.float64, device=None):
    """
    PyTorch version of interpolation_matrix using sparse COO tensor output.

    Parameters
    ----------
    locs : (n, dim) torch.Tensor
        Query locations.
    x, y, z : torch.Tensor
        Coordinate vectors along each axis.

    Returns
    -------
    torch.sparse_coo_tensor
        Interpolation matrix of shape (n, nx * ny * nz).
    """
    locs = locs.to(dtype=dtype, device=device)
    x = x.to(dtype=dtype, device=device)
    # dim = locs.shape[1]
    npts = locs.shape[0]

    if y is None and z is None:
        shape = [x.size(0)]
        inds, vals = _interpmat1D(mkvc(locs), x)
    elif z is None:
        y = y.to(dtype=dtype, device=device)
        shape = [x.size(0), y.size(0)]
        inds, vals = _interpmat2D(locs, x, y)
    else:
        y = y.to(dtype=dtype, device=device)
        z = z.to(dtype=dtype, device=device)
        shape = [x.size(0), y.size(0), z.size(0)]
        inds, vals = _interpmat3D(locs, x, y, z)

    I = torch.repeat_interleave(torch.arange(npts, device=device), 2 ** len(shape))
    if isinstance(inds, tuple):
        # Tuple of index tensors → stack and transpose
        subs = torch.stack(inds, dim=0).T  # (N, ndim)
    elif inds.ndim == 2:
        # Already 2D tensor → should already be (N, ndim)
        subs = inds
    elif inds.ndim == 1:
        # Single 1D tensor → unsqueeze to get (N, 1)
        subs = inds.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected inds shape: {inds.shape}")
    J = sub2ind(shape, subs)
    Q = torch.sparse_coo_tensor(
        indices=torch.stack([I, J]),
        values=vals,
        size=(npts, int(torch.prod(torch.tensor(shape)))),
    )
    return Q.coalesce()


def _interp_point_1D(x, xp):
    """Find bounding indices and weights for 1D linear interpolation."""
    idx = torch.searchsorted(x, xp, right=True)
    nx = x.numel()
    i2 = int(torch.clamp(idx, 0, nx - 1))
    i1 = int(torch.clamp(idx - 1, 0, nx - 1))

    if i1 == i2:
        w1 = w2 = 0.5
    else:
        denom = x[i2] - x[i1]
        w1 = (x[i2] - xp) / denom
        w2 = 1.0 - w1

    return i1, i2, w1, w2


def _get_inds_ws(x, xp):
    """
    Get indices and interpolation weights for value `xp` within a sorted 1D tensor `x`.

    Returns
    -------
    i1, i2 : int
        Indices of neighboring values in `x`.
    w1, w2 : float
        Linear interpolation weights.
    """
    nx = x.numel()

    # Binary search to find index where xp would go
    i2 = torch.searchsorted(x, torch.tensor([xp]), right=True).item()
    i1 = i2 - 1

    # Clamp to bounds
    i2 = max(min(i2, nx - 1), 0)
    i1 = max(min(i1, nx - 1), 0)

    if i1 == i2:
        w1 = 0.5
    else:
        denom = x[i2] - x[i1]
        w1 = (x[i2] - xp) / denom
    w2 = 1.0 - w1

    # Convert w1 to scalar if it's a tensor
    if isinstance(w1, torch.Tensor):
        w1 = w1.item()
    return i1, i2, w1, w2


def _interpmat1D(locs, x):
    """Generate indices and weights for 1D interpolation."""
    npts = locs.numel()
    inds = torch.empty(npts * 2, dtype=torch.long, device=locs.device)
    vals = torch.empty(npts * 2, dtype=locs.dtype, device=locs.device)

    for i in range(npts):
        i1, i2, w1, w2 = _get_inds_ws(x, locs[i].item())
        inds[2 * i] = i1
        inds[2 * i + 1] = i2
        vals[2 * i] = w1
        vals[2 * i + 1] = w2

    return inds, vals


def _interpmat2D(locs, x, y):
    """Generate interpolation indices and weights for 2D linear interpolation."""
    npts = locs.shape[0]
    inds = torch.empty((npts * 4, 2), dtype=torch.long, device=locs.device)
    vals = torch.empty(npts * 4, dtype=locs.dtype, device=locs.device)

    for i in range(npts):
        xi1, xi2, wx1, wx2 = _get_inds_ws(x, locs[i, 0].item())
        yi1, yi2, wy1, wy2 = _get_inds_ws(y, locs[i, 1].item())

        inds[4 * i + 0] = torch.tensor([xi1, yi1], device=locs.device)
        inds[4 * i + 1] = torch.tensor([xi1, yi2], device=locs.device)
        inds[4 * i + 2] = torch.tensor([xi2, yi1], device=locs.device)
        inds[4 * i + 3] = torch.tensor([xi2, yi2], device=locs.device)

        vals[4 * i + 0] = wx1 * wy1
        vals[4 * i + 1] = wx1 * wy2
        vals[4 * i + 2] = wx2 * wy1
        vals[4 * i + 3] = wx2 * wy2

    return inds, vals


def _interpmat3D(locs, x, y, z):
    """Generate interpolation indices and weights for 3D linear interpolation."""
    npts = locs.shape[0]
    inds = torch.empty((npts * 8, 3), dtype=torch.long, device=locs.device)
    vals = torch.empty(npts * 8, dtype=locs.dtype, device=locs.device)

    for i in range(npts):
        xi1, xi2, wx1, wx2 = _get_inds_ws(x, locs[i, 0].item())
        yi1, yi2, wy1, wy2 = _get_inds_ws(y, locs[i, 1].item())
        zi1, zi2, wz1, wz2 = _get_inds_ws(z, locs[i, 2].item())

        inds[8 * i + 0] = torch.tensor([xi1, yi1, zi1], device=locs.device)
        inds[8 * i + 1] = torch.tensor([xi1, yi2, zi1], device=locs.device)
        inds[8 * i + 2] = torch.tensor([xi2, yi1, zi1], device=locs.device)
        inds[8 * i + 3] = torch.tensor([xi2, yi2, zi1], device=locs.device)
        inds[8 * i + 4] = torch.tensor([xi1, yi1, zi2], device=locs.device)
        inds[8 * i + 5] = torch.tensor([xi1, yi2, zi2], device=locs.device)
        inds[8 * i + 6] = torch.tensor([xi2, yi1, zi2], device=locs.device)
        inds[8 * i + 7] = torch.tensor([xi2, yi2, zi2], device=locs.device)

        vals[8 * i + 0] = wx1 * wy1 * wz1
        vals[8 * i + 1] = wx1 * wy2 * wz1
        vals[8 * i + 2] = wx2 * wy1 * wz1
        vals[8 * i + 3] = wx2 * wy2 * wz1
        vals[8 * i + 4] = wx1 * wy1 * wz2
        vals[8 * i + 5] = wx1 * wy2 * wz2
        vals[8 * i + 6] = wx2 * wy1 * wz2
        vals[8 * i + 7] = wx2 * wy2 * wz2

    return inds, vals


def _tensor_volume_averaging(mesh_in, mesh_out, values=None, output=None):
    """
    Either apply volume-averaging from mesh_in to mesh_out,
    or return a sparse averaging matrix (PyTorch version).
    """
    dim = mesh_in.dim
    device = mesh_in.device
    dtype = mesh_in.dtype

    # Get averaging weights and mappings along each axis
    w1, i1_in, i1_out = _volume_avg_weights(mesh_in.nodes_x, mesh_out.nodes_x)
    w2, i2_in, i2_out = (
        torch.tensor([1.0], device=device, dtype=dtype),
        torch.tensor([0], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )
    w3, i3_in, i3_out = w2, i2_in, i2_out

    if dim > 1:
        w2, i2_in, i2_out = _volume_avg_weights(mesh_in.nodes_y, mesh_out.nodes_y)
    if dim > 2:
        w3, i3_in, i3_out = _volume_avg_weights(mesh_in.nodes_z, mesh_out.nodes_z)

    vol_out = mesh_out.cell_volumes.to(dtype=dtype).reshape(*mesh_out.shape_cells[::-1])

    if values is not None:
        val_in = values.to(dtype=dtype).reshape(*mesh_in.shape_cells[::-1])
        val_out = torch.zeros_like(vol_out)

        for z_in, z_out, wz in zip(i3_in, i3_out, w3):
            for y_in, y_out, wy in zip(i2_in, i2_out, w2):
                for x_in, x_out, wx in zip(i1_in, i1_out, w1):
                    weight = wx * wy * wz
                    val_out[x_out, y_out, z_out] += (
                        weight * val_in[x_in, y_in, z_in] / vol_out[x_out, y_out, z_out]
                    )
        return val_out.reshape(-1)

    # Else construct sparse averaging matrix
    n = len(w1) * len(w2) * len(w3)
    W = torch.empty(n, dtype=dtype, device=device)
    row = torch.empty(n, dtype=torch.long, device=device)
    col = torch.empty(n, dtype=torch.long, device=device)

    idx = 0
    for z_in, z_out, wz in zip(i3_in, i3_out, w3):
        for y_in, y_out, wy in zip(i2_in, i2_out, w2):
            for x_in, x_out, wx in zip(i1_in, i1_out, w1):
                flat_in = (z_in * mesh_in.shape_cells[1] + y_in) * mesh_in.shape_cells[
                    0
                ] + x_in
                flat_out = (
                    z_out * mesh_out.shape_cells[1] + y_out
                ) * mesh_out.shape_cells[0] + x_out
                row[idx] = flat_out
                col[idx] = flat_in
                W[idx] = (wx * wy * wz) / vol_out[x_out, y_out, z_out]
                idx += 1

    shape = (mesh_out.n_cells, mesh_in.n_cells)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), W, size=shape)
    return A.coalesce()


def _volume_avg_weights(x1, x2):
    """Compute volume averaging weights between 1D grids x1 (input) and x2 (output)."""
    dtype = x1.dtype
    device = x1.device
    xs = torch.cat([x1, x2]).unique(sorted=True)

    hs = []
    i1_list = []
    i2_list = []

    for i in range(xs.numel() - 1):
        center = 0.5 * (xs[i] + xs[i + 1])
        if x2[0] <= center <= x2[-1]:
            hs.append(xs[i + 1] - xs[i])
            i1 = torch.searchsorted(x1, center) - 1
            i2 = torch.searchsorted(x2, center) - 1
            i1_list.append(min(max(i1.item(), 0), x1.numel() - 1))
            i2_list.append(min(max(i2.item(), 0), x2.numel() - 1))

    return (
        torch.tensor(hs, dtype=dtype, device=device),
        torch.tensor(i1_list, dtype=torch.long, device=device),
        torch.tensor(i2_list, dtype=torch.long, device=device),
    )


def volume_average(
    mesh_in, mesh_out, values=None, output=None, dtype=torch.float64, device=None
):
    """Volume averaging interpolation between meshes.

    This volume averaging function looks for overlapping cells in each mesh,
    and weights the output values by the partial volume ratio of the overlapping
    input cells. The volume average operation should result in an output such that
    ``torch.sum(mesh_in.cell_volumes*values)`` = ``torch.sum(mesh_out.cell_volumes*output)``,
    when the input and output meshes have the exact same extent. When the output mesh
    extent goes beyond the input mesh, it is assumed to have constant values in that
    direction. When the output mesh extent is smaller than the input mesh, only the
    overlapping extent of the input mesh contributes to the output.

    This function operates in three different modes. If only *mesh_in* and
    *mesh_out* are given, the returned value is a ``scipy.sparse.csr_matrix``
    that represents this operation (so it could potentially be applied repeatedly).
    If *values* is given, the volume averaging is performed right away (without
    internally forming the matrix) and the returned value is the result of this.
    If *output* is given as well, it will be filled with the values of the
    operation and then returned (assuming it has the correct ``dtype``).

    Parameters
    ----------
    mesh_in : ~discretize.TensorMesh or ~discretize.TreeMesh
        Input mesh (the mesh you are interpolating from)
    mesh_out : ~discretize.TensorMesh or ~discretize.TreeMesh
        Output mesh (the mesh you are interpolating to)
    values : (mesh_in.n_cells) numpy.ndarray, optional
        Array with values defined at the cells of ``mesh_in``
    output : (mesh_out.n_cells) numpy.ndarray of float, optional
        Output array to be overwritten

    Returns
    -------
    (mesh_out.n_cells, mesh_in.n_cells) scipy.sparse.csr_matrix or (mesh_out.n_cells) numpy.ndarray
        If *values* = *None* , the returned value is a matrix representing this
        operation, otherwise it is a :class:`numpy.ndarray` of the result of the
        operation.

    Examples
    --------
    Create two meshes with the same extent, but different divisions (the meshes
    do not have to be the same extent).

    >>> import numpy as np
    >>> from discretize import TensorMesh
    >>> rng = np.random.default_rng(853)
    >>> h1 = np.ones(32)
    >>> h2 = np.ones(16)*2
    >>> mesh_in = TensorMesh([h1, h1])
    >>> mesh_out = TensorMesh([h2, h2])

    Create a random model defined on the input mesh, and use volume averaging to
    interpolate it to the output mesh.

    >>> from discretize.utils import volume_average
    >>> model1 = rng.random(mesh_in.nC)
    >>> model2 = volume_average(mesh_in, mesh_out, model1)

    Because these two meshes' cells are perfectly aligned, but the output mesh
    has 1 cell for each 4 of the input cells, this operation should effectively
    look like averaging each of those cells values

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 3))
    >>> ax1 = plt.subplot(121)
    >>> mesh_in.plot_image(model1, ax=ax1)
    >>> ax2 = plt.subplot(122)
    >>> mesh_out.plot_image(model2, ax=ax2)
    >>> plt.show()

    """
    try:
        in_type = mesh_in._meshType
        out_type = mesh_out._meshType
    except AttributeError:
        raise TypeError("Both input and output mesh must be valid discetize meshes")

    valid_meshs = ["TENSOR", "TREE"]
    if in_type not in valid_meshs or out_type not in valid_meshs:
        raise NotImplementedError(
            f"Volume averaging is only implemented for TensorMesh and TreeMesh, "
            f"not {type(mesh_in).__name__} and/or {type(mesh_out).__name__}"
        )

    if mesh_in.dim != mesh_out.dim:
        raise ValueError("Both meshes must have the same dimension")

    if values is not None and len(values) != mesh_in.nC:
        raise ValueError(
            "Input array does not have the same length as the number of cells in input mesh"
        )
    if output is not None and len(output) != mesh_out.nC:
        raise ValueError(
            "Output array does not have the same length as the number of cells in output mesh"
        )

    if values is not None:
        values = torch.tensor(values, dtype=dtype, device=device)
    if output is not None:
        output = torch.tensor(output, dtype=dtype, device=device)

    if in_type == "TENSOR":
        if out_type == "TENSOR":
            return _tensor_volume_averaging(mesh_in, mesh_out, values, output)
        elif out_type == "TREE":
            return mesh_out._vol_avg_from_tens(mesh_in, values, output)
    elif in_type == "TREE":
        if out_type == "TENSOR":
            return mesh_in._vol_avg_to_tens(mesh_out, values, output)
        elif out_type == "TREE":
            return mesh_out._vol_avg_from_tree(mesh_in, values, output)
    else:
        raise TypeError("Unsupported mesh types")
