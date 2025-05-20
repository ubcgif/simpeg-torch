from simpegtorch.discretize.base import BaseRectangularMesh
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
