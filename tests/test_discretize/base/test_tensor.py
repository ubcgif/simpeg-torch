import pytest
import torch

from simpegtorch import discretize

TOL = 1e-10

gen = torch.Generator().manual_seed(123)
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def setup_meshes():
    a = torch.tensor([1.0, 1.0, 1.0])
    b = torch.tensor([1.0, 2.0])
    c = torch.tensor([1.0, 4.0])
    mesh2 = discretize.TensorMesh([a, b], [3, 5])
    mesh3 = discretize.TensorMesh([a, b, c])
    return mesh2, mesh3


def test_gridded_2D(setup_meshes):
    mesh2, _ = setup_meshes
    H = mesh2.h_gridded
    test_hx = torch.all(H[:, 0] == torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    test_hy = torch.all(H[:, 1] == torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
    assert test_hx and test_hy


def test_gridded_3D(setup_meshes):
    _, mesh3 = setup_meshes
    H = mesh3.h_gridded
    test_hx = torch.all(
        H[:, 0]
        == torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    test_hy = torch.all(
        H[:, 1]
        == torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    )
    test_hz = torch.all(
        H[:, 2]
        == torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    )
    assert test_hx and test_hy and test_hz


def test_vectorN_2D(setup_meshes):
    mesh2, _ = setup_meshes
    testNx = torch.tensor([3, 4, 5, 6])
    testNy = torch.tensor([5, 6, 8])
    xtest = torch.all(mesh2.nodes_x == testNx)
    ytest = torch.all(mesh2.nodes_y == testNy)
    assert xtest and ytest


def test_vectorCC_2D(setup_meshes):
    mesh2, _ = setup_meshes
    testNx = torch.tensor([3.5, 4.5, 5.5])
    testNy = torch.tensor([5.5, 7.0])
    xtest = torch.all(mesh2.cell_centers_x == testNx)
    ytest = torch.all(mesh2.cell_centers_y == testNy)
    assert xtest and ytest


def test_area_2D(setup_meshes):
    mesh2, _ = setup_meshes
    test_area = torch.tensor(
        [
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    assert torch.all(mesh2.face_areas == test_area)


def test_area_3D(setup_meshes):
    _, mesh3 = setup_meshes
    test_area = torch.tensor(
        [
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
        ],
    )
    assert torch.all(mesh3.face_areas == test_area)


def test_vol_3D(setup_meshes):
    _, mesh3 = setup_meshes
    test_vol = torch.tensor([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
    assert torch.all(mesh3.cell_volumes == test_vol)


def test_vol_2D(setup_meshes):
    mesh2, _ = setup_meshes
    test_vol = torch.tensor([1, 1, 1, 2, 2, 2])
    assert torch.all(mesh2.cell_volumes == test_vol)


def test_edge_3D(setup_meshes):
    _, mesh3 = setup_meshes
    test_edge = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ],
    )
    assert torch.all(mesh3.edge_lengths == test_edge)


def test_edge_2D(setup_meshes):
    mesh2, _ = setup_meshes
    test_edge = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    assert torch.all(mesh2.edge_lengths == test_edge)


def test_oneCell():
    hx = torch.tensor([1e-5])
    M = discretize.TensorMesh([hx])
    assert M.n_cells == 1


def test_printing():
    print(discretize.TensorMesh([10]))
    print(discretize.TensorMesh([10, 10]))
    print(discretize.TensorMesh([10, 10, 10]))


def test_centering():
    M1d = discretize.TensorMesh([10], origin="C")
    M2d = discretize.TensorMesh([10, 10], origin="CC")
    M3d = discretize.TensorMesh([10, 10, 10], origin="CCC")
    assert torch.abs(M1d.origin + 0.5).sum() < TOL
    assert torch.abs(M2d.origin + 0.5).sum() < TOL
    assert torch.abs(M3d.origin + 0.5).sum() < TOL


def test_negative():
    M1d = discretize.TensorMesh([10], origin="N")
    with pytest.raises(Exception):
        discretize.TensorMesh([10], "F")
    M2d = discretize.TensorMesh([10, 10], origin="NN")
    M3d = discretize.TensorMesh([10, 10, 10], origin="NNN")
    assert torch.abs(M1d.origin + 1.0).sum() < TOL
    assert torch.abs(M2d.origin + 1.0).sum() < TOL
    assert torch.abs(M3d.origin + 1.0).sum() < TOL


def test_cent_neg():
    M3d = discretize.TensorMesh([10, 10, 10], origin="C0N")
    assert torch.abs(M3d.origin + torch.tensor([0.5, 0.0, 1.0])).sum() < TOL


def test_tensor():
    M = discretize.TensorMesh([[(10.0, 2)]])
    assert torch.abs(M.h[0] - torch.tensor([10.0, 10.0])).sum() < TOL


def test_serialization(setup_meshes):
    mesh2, _ = setup_meshes
    mesh = discretize.TensorMesh.deserialize(mesh2.serialize())
    assert torch.all(mesh2.origin == mesh.origin)
    assert mesh2.shape_cells == mesh.shape_cells
    assert torch.all(mesh2.h[0] == mesh.h[0])
    assert torch.all(mesh2.h[1] == mesh.h[1])
    assert torch.all(mesh2.cell_centers == mesh.cell_centers)
