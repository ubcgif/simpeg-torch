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


# Test fixtures for reshape tests
@pytest.fixture
def mesh3d():
    """Create a 3D mesh for reshape testing."""
    hx = torch.ones(3)
    hy = 2 * torch.ones(2)
    hz = 3 * torch.ones(2)
    return TensorMesh([hx, hy, hz])


@pytest.fixture
def mesh2d():
    """Create a 2D mesh for reshape testing."""
    hx = torch.ones(3)
    hy = 2 * torch.ones(2)
    return TensorMesh([hx, hy])


class TestReshape3D:
    """Test reshape method for 3D meshes."""

    def test_mesh_r_E_V(self, mesh3d):
        """Test reshape of edge vectors to components in vector format."""
        ex = torch.ones(mesh3d.n_edges_x)
        ey = torch.ones(mesh3d.n_edges_y) * 2
        ez = torch.ones(mesh3d.n_edges_z) * 3
        e = torch.cat([ex, ey, ez])

        tex = mesh3d.reshape(e, "E", "Ex", "V")
        tey = mesh3d.reshape(e, "E", "Ey", "V")
        tez = mesh3d.reshape(e, "E", "Ez", "V")

        assert torch.all(tex == ex)
        assert torch.all(tey == ey)
        assert torch.all(tez == ez)

        # Test returning all components at once
        tex, tey, tez = mesh3d.reshape(e, "E", "E", "V")
        assert torch.all(tex == ex)
        assert torch.all(tey == ey)
        assert torch.all(tez == ez)

    def test_mesh_r_F_V(self, mesh3d):
        """Test reshape of face vectors to components in vector format."""
        fx = torch.ones(mesh3d.n_faces_x)
        fy = torch.ones(mesh3d.n_faces_y) * 2
        fz = torch.ones(mesh3d.n_faces_z) * 3
        f = torch.cat([fx, fy, fz])

        tfx = mesh3d.reshape(f, "F", "Fx", "V")
        tfy = mesh3d.reshape(f, "F", "Fy", "V")
        tfz = mesh3d.reshape(f, "F", "Fz", "V")

        assert torch.all(tfx == fx)
        assert torch.all(tfy == fy)
        assert torch.all(tfz == fz)

        # Test returning all components at once
        tfx, tfy, tfz = mesh3d.reshape(f, "F", "F", "V")
        assert torch.all(tfx == fx)
        assert torch.all(tfy == fy)
        assert torch.all(tfz == fz)

    def test_mesh_r_E_M(self, mesh3d):
        """Test reshape of edge vectors to matrix format."""
        g = torch.ones((torch.prod(torch.tensor(mesh3d.shape_edges_x)).item(), 3))
        g[:, 1] = 2
        g[:, 2] = 3

        Xex, Yex, Zex = mesh3d.reshape(g, "Ex", "Ex", "M")

        assert Xex.shape == mesh3d.shape_edges_x
        assert Yex.shape == mesh3d.shape_edges_x
        assert Zex.shape == mesh3d.shape_edges_x
        assert torch.all(Xex == 1)
        assert torch.all(Yex == 2)
        assert torch.all(Zex == 3)

    def test_mesh_r_F_M(self, mesh3d):
        """Test reshape of face vectors to matrix format."""
        g = torch.ones((torch.prod(torch.tensor(mesh3d.shape_faces_x)).item(), 3))
        g[:, 1] = 2
        g[:, 2] = 3

        Xfx, Yfx, Zfx = mesh3d.reshape(g, "Fx", "Fx", "M")

        assert Xfx.shape == mesh3d.shape_faces_x
        assert Yfx.shape == mesh3d.shape_faces_x
        assert Zfx.shape == mesh3d.shape_faces_x
        assert torch.all(Xfx == 1)
        assert torch.all(Yfx == 2)
        assert torch.all(Zfx == 3)

    def test_mesh_r_CC_M(self, mesh3d):
        """Test reshape of cell center vectors to matrix format."""
        g = torch.ones((mesh3d.n_cells, 3))
        g[:, 1] = 2
        g[:, 2] = 3

        Xc, Yc, Zc = mesh3d.reshape(g, "CC", "CC", "M")

        assert Xc.shape == mesh3d.shape_cells
        assert Yc.shape == mesh3d.shape_cells
        assert Zc.shape == mesh3d.shape_cells
        assert torch.all(Xc == 1)
        assert torch.all(Yc == 2)
        assert torch.all(Zc == 3)


class TestReshape2D:
    """Test reshape method for 2D meshes."""

    def test_mesh_r_E_V(self, mesh2d):
        """Test reshape of edge vectors to components in vector format for 2D."""
        ex = torch.ones(mesh2d.n_edges_x)
        ey = torch.ones(mesh2d.n_edges_y) * 2
        e = torch.cat([ex, ey])

        tex = mesh2d.reshape(e, "E", "Ex", "V")
        tey = mesh2d.reshape(e, "E", "Ey", "V")

        assert torch.all(tex == ex)
        assert torch.all(tey == ey)

        tex, tey = mesh2d.reshape(e, "E", "E", "V")
        assert torch.all(tex == ex)
        assert torch.all(tey == ey)

        # Test that asking for z-component raises error in 2D
        with pytest.raises(ValueError):
            mesh2d.reshape(e, "E", "Ez", "V")

    def test_mesh_r_F_V(self, mesh2d):
        """Test reshape of face vectors to components in vector format for 2D."""
        fx = torch.ones(mesh2d.n_faces_x)
        fy = torch.ones(mesh2d.n_faces_y) * 2
        f = torch.cat([fx, fy])

        tfx = mesh2d.reshape(f, "F", "Fx", "V")
        tfy = mesh2d.reshape(f, "F", "Fy", "V")

        assert torch.all(tfx == fx)
        assert torch.all(tfy == fy)

        tfx, tfy = mesh2d.reshape(f, "F", "F", "V")
        assert torch.all(tfx == fx)
        assert torch.all(tfy == fy)

        # Test that asking for z-component raises error in 2D
        with pytest.raises(ValueError):
            mesh2d.reshape(f, "F", "Fz", "V")

    def test_mesh_r_E_M(self, mesh2d):
        """Test reshape of edge vectors to matrix format for 2D."""
        g = torch.ones((torch.prod(torch.tensor(mesh2d.shape_edges_x)).item(), 2))
        g[:, 1] = 2

        Xex, Yex = mesh2d.reshape(g, "Ex", "Ex", "M")

        assert Xex.shape == mesh2d.shape_edges_x
        assert Yex.shape == mesh2d.shape_edges_x
        assert torch.all(Xex == 1)
        assert torch.all(Yex == 2)

    def test_mesh_r_F_M(self, mesh2d):
        """Test reshape of face vectors to matrix format for 2D."""
        g = torch.ones((torch.prod(torch.tensor(mesh2d.shape_faces_x)).item(), 2))
        g[:, 1] = 2

        Xfx, Yfx = mesh2d.reshape(g, "Fx", "Fx", "M")

        assert Xfx.shape == mesh2d.shape_faces_x
        assert Yfx.shape == mesh2d.shape_faces_x
        assert torch.all(Xfx == 1)
        assert torch.all(Yfx == 2)

    def test_mesh_r_CC_M(self, mesh2d):
        """Test reshape of cell center vectors to matrix format for 2D."""
        g = torch.ones((mesh2d.n_cells, 2))
        g[:, 1] = 2

        Xc, Yc = mesh2d.reshape(g, "CC", "CC", "M")

        assert Xc.shape == mesh2d.shape_cells
        assert Yc.shape == mesh2d.shape_cells
        assert torch.all(Xc == 1)
        assert torch.all(Yc == 2)
