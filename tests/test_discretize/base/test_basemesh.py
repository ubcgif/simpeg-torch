from simpegtorch.discretize.base import BaseRectangularMesh
from simpegtorch.discretize import TensorMesh
import torch
import pytest

torch.set_default_dtype(torch.float64)


@pytest.fixture()
def mesh():
    return BaseRectangularMesh([6, 2, 3])


def test_meshDimensions(mesh):
    assert mesh.dim == 3


def test_mesh_nc(mesh):
    assert mesh.n_cells == 36
    assert mesh.shape_cells == (6, 2, 3)


def test_mesh_nc_xyz(mesh):
    assert mesh.shape_cells[0] == 6
    assert mesh.shape_cells[1] == 2
    assert mesh.shape_cells[2] == 3


def test_mesh_nf(mesh):
    assert mesh.shape_faces_x == (7, 2, 3)
    assert mesh.shape_faces_y == (6, 3, 3)
    assert mesh.shape_faces_z == (6, 2, 4)


def test_mesh_ne(mesh):
    assert mesh.shape_edges_x == (6, 3, 4)
    assert mesh.shape_edges_y == (7, 2, 4)
    assert mesh.shape_edges_z == (7, 3, 3)


def test_mesh_numbers(mesh):
    assert mesh.n_cells == 36
    assert mesh.n_faces_per_direction == (42, 54, 48)
    assert mesh.n_edges_per_direction == (72, 56, 63)
    assert mesh.n_faces == torch.sum(torch.tensor([42, 54, 48]))
    assert mesh.n_edges == torch.sum(torch.tensor([72, 56, 63]))


def test_mesh_origin():
    h = [torch.ones(6), torch.ones(2), torch.ones(3)]
    origin = torch.tensor([1.0, 2.0, 3.0])

    mesh1 = TensorMesh(h, origin=origin)
    mesh2 = TensorMesh(h)
    mesh2.origin = origin
    assert torch.allclose(mesh1.origin, mesh2.origin)


def test_n_cells_correctness():
    # Test 1D mesh
    h1d = [torch.ones(5)]
    mesh1d = TensorMesh(h1d)
    assert mesh1d.n_cells == 5

    # Test 2D mesh
    h2d = [torch.ones(3), torch.ones(4)]
    mesh2d = TensorMesh(h2d)
    assert mesh2d.n_cells == 3 * 4

    # Test 3D mesh
    h3d = [torch.ones(2), torch.ones(3), torch.ones(4)]
    mesh3d = TensorMesh(h3d)
    assert mesh3d.n_cells == 2 * 3 * 4


def test_n_nodes_correctness():
    # Test 1D mesh: nodes = (n_cells_x + 1)
    h1d = [torch.ones(5)]
    mesh1d = TensorMesh(h1d)
    assert mesh1d.n_nodes == 6

    # Test 2D mesh: nodes = (n_cells_x + 1) * (n_cells_y + 1)
    h2d = [torch.ones(3), torch.ones(4)]
    mesh2d = TensorMesh(h2d)
    assert mesh2d.n_nodes == 4 * 5

    # Test 3D mesh: nodes = (n_cells_x + 1) * (n_cells_y + 1) * (n_cells_z + 1)
    h3d = [torch.ones(2), torch.ones(3), torch.ones(4)]
    mesh3d = TensorMesh(h3d)
    assert mesh3d.n_nodes == 3 * 4 * 5


def test_n_edges_x_correctness():
    # Test 1D mesh: x-edges = n_cells_x
    h1d = [torch.ones(5)]
    mesh1d = TensorMesh(h1d)
    assert mesh1d.n_edges_x == 5

    # Test 2D mesh: x-edges = n_cells_x * (n_cells_y + 1)
    h2d = [torch.ones(3), torch.ones(4)]
    mesh2d = TensorMesh(h2d)
    assert mesh2d.n_edges_x == 3 * 5

    # Test 3D mesh: x-edges = n_cells_x * (n_cells_y + 1) * (n_cells_z + 1)
    h3d = [torch.ones(2), torch.ones(3), torch.ones(4)]
    mesh3d = TensorMesh(h3d)
    assert mesh3d.n_edges_x == 2 * 4 * 5


def test_n_edges_y_correctness():
    # Test 1D mesh: no y-edges
    h1d = [torch.ones(5)]
    mesh1d = TensorMesh(h1d)
    assert mesh1d.n_edges_y is None

    # Test 2D mesh: y-edges = (n_cells_x + 1) * n_cells_y
    h2d = [torch.ones(3), torch.ones(4)]
    mesh2d = TensorMesh(h2d)
    assert mesh2d.n_edges_y == 4 * 4

    # Test 3D mesh: y-edges = (n_cells_x + 1) * n_cells_y * (n_cells_z + 1)
    h3d = [torch.ones(2), torch.ones(3), torch.ones(4)]
    mesh3d = TensorMesh(h3d)
    assert mesh3d.n_edges_y == 3 * 3 * 5


def test_n_edges_z_correctness():
    # Test 1D mesh: no z-edges
    h1d = [torch.ones(5)]
    mesh1d = TensorMesh(h1d)
    assert mesh1d.n_edges_z is None

    # Test 2D mesh: no z-edges
    h2d = [torch.ones(3), torch.ones(4)]
    mesh2d = TensorMesh(h2d)
    assert mesh2d.n_edges_z is None

    # Test 3D mesh: z-edges = (n_cells_x + 1) * (n_cells_y + 1) * n_cells_z
    h3d = [torch.ones(2), torch.ones(3), torch.ones(4)]
    mesh3d = TensorMesh(h3d)
    assert mesh3d.n_edges_z == 3 * 4 * 4
