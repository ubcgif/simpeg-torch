import torch

from .matrix_utils import sub2ind


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
    dim = locs.shape[1]
    npts = locs.shape[0]

    if dim == 1:
        shape = [x.size(0)]
        inds, vals = _interpmat1D(locs.view(-1), x)
    elif dim == 2:
        y = y.to(dtype=dtype, device=device)
        shape = [x.size(0), y.size(0)]
        inds, vals = _interpmat2D(locs, x, y)
    elif dim == 3:
        y = y.to(dtype=dtype, device=device)
        z = z.to(dtype=dtype, device=device)
        shape = [x.size(0), y.size(0), z.size(0)]
        inds, vals = _interpmat3D(locs, x, y, z)
    else:
        raise ValueError("locs must be 1D, 2D or 3D")

    I = torch.repeat_interleave(torch.arange(npts, device=device), 2**dim)
    J = sub2ind(shape, inds)
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


def _interpmat1D(locs, x):
    """Generate indices and weights for 1D interpolation."""
    npts = locs.numel()
    inds = torch.empty(npts * 2, dtype=torch.long, device=locs.device)
    vals = torch.empty(npts * 2, dtype=locs.dtype, device=locs.device)

    for i in range(npts):
        i1, i2, w1, w2 = _interp_point_1D(x, locs[i].item())
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
        xi1, xi2, wx1, wx2 = _interp_point_1D(x, locs[i, 0].item())
        yi1, yi2, wy1, wy2 = _interp_point_1D(y, locs[i, 1].item())

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
        xi1, xi2, wx1, wx2 = _interp_point_1D(x, locs[i, 0].item())
        yi1, yi2, wy1, wy2 = _interp_point_1D(y, locs[i, 1].item())
        zi1, zi2, wz1, wz2 = _interp_point_1D(z, locs[i, 2].item())

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
