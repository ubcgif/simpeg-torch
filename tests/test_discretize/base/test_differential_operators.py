"""
Tests for differential operators in PyTorch implementation.

Migrated from simpeg_old/discretize/tests/base/test_operators.py
"""

# flake8: noqa: E731
import torch
import numpy as np
import pytest
from simpegtorch.discretize.tensor_mesh import TensorMesh
from simpegtorch.discretize.tests import OrderTest
from simpegtorch.discretize.tests import setup_mesh

MESHTYPES = ["uniformTensorMesh"]


# Tolerance for numerical tests
TOL = 1e-12

# Test mesh sizes for order tests
MESH_SIZES_2D = [8, 16, 32]
MESH_SIZES_3D = [8, 16, 24]


# Helper functions for analytical solutions
def call2(fun, xyz):
    """Call 2D function on grid points."""
    return fun(xyz[:, 0], xyz[:, 1])


def call3(fun, xyz):
    """Call 3D function on grid points."""
    return fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])


def cart_row2(g, xfun, yfun):
    """Create 2D vector row."""
    return torch.stack([call2(xfun, g), call2(yfun, g)], dim=-1)


def cartF2(M, fx, fy):
    """Create 2D face vector field."""
    # Original: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
    fx_faces = cart_row2(M.faces_x, fx, fy)
    fy_faces = cart_row2(M.faces_y, fx, fy)
    return torch.cat([fx_faces, fy_faces], dim=0)


def cartE2(M, ex, ey):
    """Create 2D edge vector field."""
    # Original: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
    ex_edges = cart_row2(M.edges_x, ex, ey)
    ey_edges = cart_row2(M.edges_y, ex, ey)
    return torch.cat([ex_edges, ey_edges], dim=0)


def cart_row3(g, xfun, yfun, zfun):
    """Create 3D vector row."""
    return torch.stack([call3(xfun, g), call3(yfun, g), call3(zfun, g)], dim=-1)


def cartF3(M, fx, fy, fz):
    """Create 3D face vector field."""
    # Original: np.vstack((
    #     cart_row3(M.gridFx, fx, fy, fz),
    #     cart_row3(M.gridFy, fx, fy, fz),
    #     cart_row3(M.gridFz, fx, fy, fz),
    # ))
    fx_faces = cart_row3(M.faces_x, fx, fy, fz)
    fy_faces = cart_row3(M.faces_y, fx, fy, fz)
    fz_faces = cart_row3(M.faces_z, fx, fy, fz)
    return torch.cat([fx_faces, fy_faces, fz_faces], dim=0)


def cartE3(M, ex, ey, ez):
    """Create 3D edge vector field."""
    # Original: np.vstack((
    #     cart_row3(M.gridEx, ex, ey, ez),
    #     cart_row3(M.gridEy, ex, ey, ez),
    #     cart_row3(M.gridEz, ex, ey, ez),
    # ))
    ex_edges = cart_row3(M.edges_x, ex, ey, ez)
    ey_edges = cart_row3(M.edges_y, ex, ey, ez)
    ez_edges = cart_row3(M.edges_z, ex, ey, ez)
    return torch.cat([ex_edges, ey_edges, ez_edges], dim=0)


def create_uniform_tensor_mesh(n, dim=2, device="cpu"):
    """Create uniform tensor mesh for testing."""
    if dim == 1:
        h = [torch.ones(n, device=device, dtype=torch.float64)]
    elif dim == 2:
        h = [
            torch.ones(n, device=device, dtype=torch.float64),
            torch.ones(n, device=device, dtype=torch.float64),
        ]
    elif dim == 3:
        h = [
            torch.ones(n, device=device, dtype=torch.float64),
            torch.ones(n, device=device, dtype=torch.float64),
            torch.ones(n, device=device, dtype=torch.float64),
        ]
    else:
        raise ValueError("dim must be 1, 2, or 3")

    return TensorMesh(h, device=device)


class TestCurl(OrderTest):
    name = "Curl"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # fun: i (cos(y)) + j (cos(z)) + k (cos(x))
        # sol: i (sin(z)) + j (sin(x)) + k (sin(y))

        funX = lambda x, y, z: torch.cos(2 * torch.pi * y)
        funY = lambda x, y, z: torch.cos(2 * torch.pi * z)
        funZ = lambda x, y, z: torch.cos(2 * torch.pi * x)

        solX = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * z)
        solY = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * x)
        solZ = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * y)

        # Create edge field - convert to 1D format
        Ec = cartE3(self.M, funX, funY, funZ)
        # Extract tangential components and concatenate
        E_x = Ec[: self.M.n_edges_x, 0]  # x-component at x-edges
        E_y = Ec[
            self.M.n_edges_x : self.M.n_edges_x + self.M.n_edges_y, 1
        ]  # y-component at y-edges
        E_z = Ec[self.M.n_edges_x + self.M.n_edges_y :, 2]  # z-component at z-edges
        E = torch.cat([E_x, E_y, E_z])

        # Analytical curl - convert to same format
        Fc = cartF3(self.M, solX, solY, solZ)
        # Project to face normals like face divergence
        curlE_ana = torch.sum(Fc * self.M.face_normals, dim=1)

        curlE = self.M.edge_curl @ E

        err = torch.norm(curlE - curlE_ana, float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestCurl2D(OrderTest):
    name = "Cell Grad 2D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        ex = lambda x, y: torch.cos(y)
        ey = lambda x, y: torch.cos(x)
        sol = lambda x, y: -torch.sin(x) + torch.sin(y)

        sol_curl2d = call2(sol, self.M.cell_centers)

        # Create edge field - convert to 1D format
        Ec = cartE2(self.M, ex, ey)
        # Extract tangential components and concatenate
        E_x = Ec[: self.M.n_edges_x, 0]  # x-component at x-edges
        E_y = Ec[self.M.n_edges_x :, 1]  # y-component at y-edges
        E = torch.cat([E_x, E_y])

        sol_ana = self.M.edge_curl @ E
        err = torch.norm(sol_curl2d - sol_ana, float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad2D_Dirichlet(OrderTest):
    name = "Cell Grad 2D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = (
            lambda x, y: 2 * np.pi * torch.cos(2 * np.pi * x) * torch.sin(2 * np.pi * y)
        )
        fy = (
            lambda x, y: 2 * np.pi * torch.cos(2 * np.pi * y) * torch.sin(2 * np.pi * x)
        )
        sol = lambda x, y: torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

        xc = call2(sol, self.M.gridCC)

        Fc = cartF2(self.M, fx, fy)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("dirichlet")
        gradX = self.M.cell_gradient @ xc

        err = torch.norm((gradX - gradX_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad3D_Dirichlet(OrderTest):
    name = "Cell Grad 3D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # Test function
        fx = (
            lambda x, y, z: 2
            * np.pi
            * torch.cos(2 * np.pi * x)
            * torch.sin(2 * np.pi * y)
            * torch.sin(2 * np.pi * z)
        )
        fy = (
            lambda x, y, z: 2
            * np.pi
            * torch.sin(2 * np.pi * x)
            * torch.cos(2 * np.pi * y)
            * torch.sin(2 * np.pi * z)
        )
        fz = (
            lambda x, y, z: 2
            * np.pi
            * torch.sin(2 * np.pi * x)
            * torch.sin(2 * np.pi * y)
            * torch.cos(2 * np.pi * z)
        )
        sol = (
            lambda x, y, z: torch.sin(2 * np.pi * x)
            * torch.sin(2 * np.pi * y)
            * torch.sin(2 * np.pi * z)
        )

        xc = call3(sol, self.M.gridCC)

        Fc = cartF3(self.M, fx, fy, fz)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("dirichlet")
        gradX = self.M.cell_gradient @ xc

        err = torch.norm((gradX - gradX_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad2D_Neumann(OrderTest):
    name = "Cell Grad 2D - Neumann"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = (
            lambda x, y: -2
            * np.pi
            * torch.sin(2 * np.pi * x)
            * torch.cos(2 * np.pi * y)
        )
        fy = (
            lambda x, y: -2
            * np.pi
            * torch.sin(2 * np.pi * y)
            * torch.cos(2 * np.pi * x)
        )
        sol = lambda x, y: torch.cos(2 * np.pi * x) * torch.cos(2 * np.pi * y)

        xc = call2(sol, self.M.gridCC)

        Fc = cartF2(self.M, fx, fy)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("neumann")
        gradX = self.M.cell_gradient @ xc

        err = torch.norm((gradX - gradX_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad3D_Neumann(OrderTest):
    name = "Cell Grad 3D - Neumann"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # Test function
        fx = (
            lambda x, y, z: -2
            * np.pi
            * torch.sin(2 * np.pi * x)
            * torch.cos(2 * np.pi * y)
            * torch.cos(2 * np.pi * z)
        )
        fy = (
            lambda x, y, z: -2
            * np.pi
            * torch.cos(2 * np.pi * x)
            * torch.sin(2 * np.pi * y)
            * torch.cos(2 * np.pi * z)
        )
        fz = (
            lambda x, y, z: -2
            * np.pi
            * torch.cos(2 * np.pi * x)
            * torch.cos(2 * np.pi * y)
            * torch.sin(2 * np.pi * z)
        )
        sol = (
            lambda x, y, z: torch.cos(2 * np.pi * x)
            * torch.cos(2 * np.pi * y)
            * torch.cos(2 * np.pi * z)
        )

        xc = call3(sol, self.M.gridCC)

        Fc = cartF3(self.M, fx, fy, fz)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("neumann")
        gradX = self.M.cell_gradient @ (xc)

        err = torch.norm((gradX - gradX_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv3D(OrderTest):
    name = "Face Divergence 3D"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y, z: torch.sin(2 * np.pi * x)
        fy = lambda x, y, z: torch.sin(2 * np.pi * y)
        fz = lambda x, y, z: torch.sin(2 * np.pi * z)
        sol = lambda x, y, z: (
            2 * np.pi * torch.cos(2 * np.pi * x)
            + 2 * np.pi * torch.cos(2 * np.pi * y)
            + 2 * np.pi * torch.cos(2 * np.pi * z)
        )

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.project_face_vector(Fc)

        divF = self.M.face_divergence @ F
        divF_ana = call3(sol, self.M.gridCC)

        if self._meshType == "rotateCurv":
            # Really it is the integration we should be caring about:
            # So, let us look at the l2 norm.
            err = torch.norm(self.M.cell_volumes * (divF - divF_ana), 2).item()
        else:
            err = torch.norm((divF - divF_ana), float("inf")).item()
        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv2D(OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y: torch.sin(2 * np.pi * x)
        fy = lambda x, y: torch.sin(2 * np.pi * y)
        sol = (
            lambda x, y: 2
            * np.pi
            * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))
        )

        Fc = cartF2(self.M, fx, fy)
        F = self.M.project_face_vector(Fc)

        divF = self.M.face_divergence @ F
        divF_ana = call2(sol, self.M.gridCC)

        err = torch.norm((divF - divF_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(OrderTest):
    name = "Nodal Gradient"
    meshTypes = MESHTYPES

    def getError(self):
        # Test function
        fun = lambda x, y, z: (torch.cos(x) + torch.cos(y) + torch.cos(z))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y, z: -torch.sin(x)
        solY = lambda x, y, z: -torch.sin(y)
        solZ = lambda x, y, z: -torch.sin(z)

        phi = call3(fun, self.M.gridN)
        gradE = self.M.nodal_gradient @ phi

        Ec = cartE3(self.M, solX, solY, solZ)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = torch.norm((gradE - gradE_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        # Test function
        fun = lambda x, y: (torch.cos(x) + torch.cos(y))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y: -torch.sin(x)
        solY = lambda x, y: -torch.sin(y)

        phi = call2(fun, self.M.gridN)
        gradE = self.M.nodal_gradient @ phi

        Ec = cartE2(self.M, solX, solY)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = torch.norm((gradE - gradE_ana), float("inf")).item()

        return err

    def test_order(self):
        self.orderTest()


@pytest.mark.parametrize("device", ["cpu"])
class TestFaceDivergence:
    """Test face divergence operators."""

    def test_face_divergence_2d(self, device):
        """Test 2D face divergence operator with analytical solution."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(10), 0.1 * torch.ones(10)]
        mesh = TensorMesh(h, device=device)

        # Test function
        fx = lambda x, y: torch.sin(2 * torch.pi * x)
        fy = lambda x, y: torch.sin(2 * torch.pi * y)
        sol = (
            lambda x, y: 2
            * torch.pi
            * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))
        )

        # Create face vector field
        Fc = cartF2(mesh, fx, fy)

        # Project to face normals (equivalent to mesh.project_face_vector(Fc))
        # For tensor mesh: dot product with face normals
        F = torch.sum(Fc * mesh.face_normals, dim=1)

        # Compute divergence
        divF = mesh.face_divergence @ F
        divF_ana = call2(sol, mesh.cell_centers)

        # Check error
        err = torch.norm(divF - divF_ana, float("inf")).item()
        rel_err = err / torch.norm(divF_ana, float("inf")).item()
        print(f"2D Face divergence error: {err}, relative: {rel_err:.4f}")

        # Use relative error tolerance (should be <5% for proper implementation)
        assert rel_err < 0.05  # 5% relative error tolerance

    def test_face_divergence_3d(self, device):
        """Test 3D face divergence operator with analytical solution."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(8), 0.1 * torch.ones(8), 0.1 * torch.ones(8)]
        mesh = TensorMesh(h, device=device)

        # Test function
        fx = lambda x, y, z: torch.sin(2 * torch.pi * x)
        fy = lambda x, y, z: torch.sin(2 * torch.pi * y)
        fz = lambda x, y, z: torch.sin(2 * torch.pi * z)
        sol = lambda x, y, z: (
            2 * torch.pi * torch.cos(2 * torch.pi * x)
            + 2 * torch.pi * torch.cos(2 * torch.pi * y)
            + 2 * torch.pi * torch.cos(2 * torch.pi * z)
        )

        # Create face vector field
        Fc = cartF3(mesh, fx, fy, fz)

        # Project to face normals (equivalent to mesh.project_face_vector(Fc))
        # For tensor mesh: dot product with face normals
        F = torch.sum(Fc * mesh.face_normals, dim=1)

        # Compute divergence
        divF = mesh.face_divergence @ F
        divF_ana = call3(sol, mesh.cell_centers)

        # Check error
        err = torch.norm(divF - divF_ana, float("inf")).item()
        rel_err = err / torch.norm(divF_ana, float("inf")).item()
        print(f"3D Face divergence error: {err}, relative: {rel_err:.4f}")

        # Use relative error tolerance (should be <5% for proper implementation)
        assert rel_err < 0.05  # 5% relative error tolerance


@pytest.mark.parametrize("device", ["cpu"])
class TestNodalGradient:
    """Test nodal gradient operators."""

    def test_nodal_gradient_2d(self, device):
        """Test 2D nodal gradient operator."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(10), 0.1 * torch.ones(10)]
        mesh = TensorMesh(h, device=device)

        # Test function
        fun = lambda x, y: torch.cos(x) + torch.cos(y)
        solX = lambda x, y: -torch.sin(x)
        solY = lambda x, y: -torch.sin(y)

        # Nodal values
        phi = call2(fun, mesh.nodes)

        # Compute gradient
        gradE = mesh.nodal_gradient @ phi

        # Analytical gradient - convert to same format as computed gradient
        Ec = cartE2(mesh, solX, solY)
        # Extract tangential components and concatenate
        gradE_ana_x = Ec[: mesh.n_edges_x, 0]  # x-component at x-edges
        gradE_ana_y = Ec[mesh.n_edges_x :, 1]  # y-component at y-edges
        gradE_ana = torch.cat([gradE_ana_x, gradE_ana_y])

        # Check error (use L2 norm as edges might have different ordering)
        err = torch.norm(gradE - gradE_ana).item() / torch.norm(gradE_ana).item()
        print(f"2D Nodal gradient relative error: {err}")
        assert err < 0.01  # Should have <1% relative error for proper implementation

    def test_nodal_gradient_3d(self, device):
        """Test 3D nodal gradient operator."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(8), 0.1 * torch.ones(8), 0.1 * torch.ones(8)]
        mesh = TensorMesh(h, device=device)

        # Test function
        fun = lambda x, y, z: torch.cos(x) + torch.cos(y) + torch.cos(z)
        solX = lambda x, y, z: -torch.sin(x)
        solY = lambda x, y, z: -torch.sin(y)
        solZ = lambda x, y, z: -torch.sin(z)

        # Nodal values
        phi = call3(fun, mesh.nodes)

        # Compute gradient
        gradE = mesh.nodal_gradient @ phi

        # Analytical gradient - convert to same format as computed gradient
        Ec = cartE3(mesh, solX, solY, solZ)
        # Extract tangential components and concatenate
        gradE_ana_x = Ec[: mesh.n_edges_x, 0]  # x-component at x-edges
        gradE_ana_y = Ec[
            mesh.n_edges_x : mesh.n_edges_x + mesh.n_edges_y, 1
        ]  # y-component at y-edges
        gradE_ana_z = Ec[mesh.n_edges_x + mesh.n_edges_y :, 2]  # z-component at z-edges
        gradE_ana = torch.cat([gradE_ana_x, gradE_ana_y, gradE_ana_z])

        # Check error (use L2 norm as edges might have different ordering)
        err = torch.norm(gradE - gradE_ana).item() / torch.norm(gradE_ana).item()
        print(f"3D Nodal gradient relative error: {err}")
        assert err < 0.01  # Should have <1% relative error for proper implementation


@pytest.mark.parametrize("device", ["cpu"])
class TestEdgeCurl:
    """Test edge curl operators."""

    def test_edge_curl_2d(self, device):
        """Test 2D edge curl operator."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(10), 0.1 * torch.ones(10)]
        mesh = TensorMesh(h, device=device)

        # Test function
        ex = lambda x, y: torch.cos(y)
        ey = lambda x, y: torch.cos(x)
        sol = lambda x, y: -torch.sin(x) + torch.sin(y)

        # Analytical solution
        sol_curl2d = call2(sol, mesh.cell_centers)

        # Create edge field - convert to 1D format
        Ec = cartE2(mesh, ex, ey)
        # Extract tangential components and concatenate
        E_x = Ec[: mesh.n_edges_x, 0]  # x-component at x-edges
        E_y = Ec[mesh.n_edges_x :, 1]  # y-component at y-edges
        E = torch.cat([E_x, E_y])

        # Compute curl
        curlE = mesh.edge_curl @ E

        # Check error
        err = torch.norm(curlE - sol_curl2d, float("inf")).item()
        print(f"2D Edge curl error: {err}")
        # Use proper tolerance based on mesh spacing (h=0.1, expect ~h^2 error)
        assert err < 0.02  # Should be O(h^2) ≈ 0.01 for h=0.1

    def test_edge_curl_3d(self, device):
        """Test 3D edge curl operator."""
        # Use a mesh with spacing that doesn't align with the zero points of sin
        h = [0.1 * torch.ones(6), 0.1 * torch.ones(6), 0.1 * torch.ones(6)]
        mesh = TensorMesh(h, device=device)

        # Test function: i(cos(y)) + j(cos(z)) + k(cos(x))
        # Solution: i(sin(z)) + j(sin(x)) + k(sin(y))
        funX = lambda x, y, z: torch.cos(2 * torch.pi * y)
        funY = lambda x, y, z: torch.cos(2 * torch.pi * z)
        funZ = lambda x, y, z: torch.cos(2 * torch.pi * x)

        solX = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * z)
        solY = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * x)
        solZ = lambda x, y, z: 2 * torch.pi * torch.sin(2 * torch.pi * y)

        # Create edge field - convert to 1D format
        Ec = cartE3(mesh, funX, funY, funZ)
        # Extract tangential components and concatenate
        E_x = Ec[: mesh.n_edges_x, 0]  # x-component at x-edges
        E_y = Ec[
            mesh.n_edges_x : mesh.n_edges_x + mesh.n_edges_y, 1
        ]  # y-component at y-edges
        E_z = Ec[mesh.n_edges_x + mesh.n_edges_y :, 2]  # z-component at z-edges
        E = torch.cat([E_x, E_y, E_z])

        # Compute curl
        curlE = mesh.edge_curl @ E

        # Analytical curl - convert to same format
        Fc = cartF3(mesh, solX, solY, solZ)
        # Project to face normals like face divergence
        curlE_ana = torch.sum(Fc * mesh.face_normals, dim=1)

        # Check error (use L2 norm as faces might have different ordering)
        err = torch.norm(curlE - curlE_ana).item() / torch.norm(curlE_ana).item()
        print(f"3D Edge curl relative error: {err}")
        assert err < 0.02  # Should have <2% relative error for proper implementation


@pytest.mark.parametrize("device", ["cpu"])
class TestAveragingOperators:
    """Test averaging operators."""

    def test_averaging_face_to_cell_2d(self, device):
        """Test 2D face-to-cell averaging operators."""
        mesh = create_uniform_tensor_mesh(16, dim=2, device=device)

        # Test with simple linear function
        fun = lambda x, y: x + y

        # Face values
        fx_vals = call2(fun, mesh.faces_x)
        fy_vals = call2(fun, mesh.faces_y)

        # Expected cell values
        cc_vals_expected = call2(fun, mesh.cell_centers)

        # Compute averaged values
        cc_from_fx = mesh.average_face_x_to_cell @ fx_vals
        cc_from_fy = mesh.average_face_y_to_cell @ fy_vals

        # Check that averaging preserves linear functions reasonably well
        err_x = (
            torch.norm(cc_from_fx - cc_vals_expected).item()
            / torch.norm(cc_vals_expected).item()
        )
        err_y = (
            torch.norm(cc_from_fy - cc_vals_expected).item()
            / torch.norm(cc_vals_expected).item()
        )

        print(f"2D Face-x to cell averaging relative error: {err_x}")
        print(f"2D Face-y to cell averaging relative error: {err_y}")

        # These should be small for linear functions (averaging should preserve them well)
        assert err_x < 0.05  # Should have <5% error for linear functions
        assert err_y < 0.05

    def test_averaging_edge_to_cell_2d(self, device):
        """Test 2D edge-to-cell averaging operators."""
        mesh = create_uniform_tensor_mesh(16, dim=2, device=device)

        # Test with simple linear function
        fun = lambda x, y: x + y

        # Edge values
        ex_vals = call2(fun, mesh.edges_x)
        ey_vals = call2(fun, mesh.edges_y)

        # Expected cell values
        cc_vals_expected = call2(fun, mesh.cell_centers)

        # Compute averaged values
        cc_from_ex = mesh.average_edge_x_to_cell @ ex_vals
        cc_from_ey = mesh.average_edge_y_to_cell @ ey_vals

        # Check that averaging preserves linear functions reasonably well
        err_x = (
            torch.norm(cc_from_ex - cc_vals_expected).item()
            / torch.norm(cc_vals_expected).item()
        )
        err_y = (
            torch.norm(cc_from_ey - cc_vals_expected).item()
            / torch.norm(cc_vals_expected).item()
        )

        print(f"2D Edge-x to cell averaging relative error: {err_x}")
        print(f"2D Edge-y to cell averaging relative error: {err_y}")

        # These should be small for linear functions (averaging should preserve them well)
        assert err_x < 0.05  # Should have <5% error for linear functions
        assert err_y < 0.05

    def test_average_face_to_cell_vector_2d(self, device):
        """Test 2D face-to-cell vector averaging operator."""
        mesh = create_uniform_tensor_mesh(8, dim=2, device=device)

        # Create test face data
        torch.manual_seed(42)
        face_data = torch.randn(mesh.n_faces, device=device, dtype=mesh.dtype)

        # Apply vector averaging
        ave_op = mesh.average_face_to_cell_vector
        result = ave_op @ face_data

        # Check dimensions
        assert result.shape == (
            mesh.n_cells * 2,
        ), f"Expected shape {(mesh.n_cells * 2,)}, got {result.shape}"

        # Test block diagonal structure
        # X-faces should only affect x-components
        x_face_data = torch.zeros_like(face_data)
        x_face_data[: mesh.n_faces_x] = 1.0
        x_result = ave_op @ x_face_data

        x_components = x_result[: mesh.n_cells]
        y_components = x_result[mesh.n_cells :]

        assert torch.all(y_components == 0), "X-faces should not affect y-components"
        assert torch.any(x_components != 0), "X-faces should affect x-components"

        # Y-faces should only affect y-components
        y_face_data = torch.zeros_like(face_data)
        y_face_data[mesh.n_faces_x :] = 1.0
        y_result = ave_op @ y_face_data

        x_components_y = y_result[: mesh.n_cells]
        y_components_y = y_result[mesh.n_cells :]

        assert torch.all(x_components_y == 0), "Y-faces should not affect x-components"
        assert torch.any(y_components_y != 0), "Y-faces should affect y-components"

    def test_average_face_to_cell_vector_3d(self, device):
        """Test 3D face-to-cell vector averaging operator."""
        mesh = create_uniform_tensor_mesh(4, dim=3, device=device)

        # Create test face data
        torch.manual_seed(42)
        face_data = torch.randn(mesh.n_faces, device=device, dtype=mesh.dtype)

        # Apply vector averaging
        ave_op = mesh.average_face_to_cell_vector
        result = ave_op @ face_data

        # Check dimensions
        assert result.shape == (
            mesh.n_cells * 3,
        ), f"Expected shape {(mesh.n_cells * 3,)}, got {result.shape}"

        # Test block diagonal structure
        # X-faces should only affect x-components
        x_face_data = torch.zeros_like(face_data)
        x_face_data[: mesh.n_faces_x] = 1.0
        x_result = ave_op @ x_face_data

        x_components = x_result[: mesh.n_cells]
        y_components = x_result[mesh.n_cells : 2 * mesh.n_cells]
        z_components = x_result[2 * mesh.n_cells :]

        assert torch.all(y_components == 0), "X-faces should not affect y-components"
        assert torch.all(z_components == 0), "X-faces should not affect z-components"
        assert torch.any(x_components != 0), "X-faces should affect x-components"

    def test_average_cell_vector_to_face_2d(self, device):
        """Test 2D cell-vector-to-face averaging operator."""
        mesh = create_uniform_tensor_mesh(8, dim=2, device=device)

        # Create test cell vector data (x and y components)
        torch.manual_seed(42)
        cell_vector_data = torch.randn(
            mesh.n_cells * 2, device=device, dtype=mesh.dtype
        )

        # Apply vector averaging
        ave_op = mesh.average_cell_vector_to_face
        result = ave_op @ cell_vector_data

        # Check dimensions
        assert result.shape == (
            mesh.n_faces,
        ), f"Expected shape {(mesh.n_faces,)}, got {result.shape}"

        # Test block diagonal structure
        # X-components should only affect x-faces
        x_cell_data = torch.zeros_like(cell_vector_data)
        x_cell_data[: mesh.n_cells] = 1.0  # Only x-components
        x_result = ave_op @ x_cell_data

        x_face_result = x_result[: mesh.n_faces_x]
        y_face_result = x_result[mesh.n_faces_x :]

        assert torch.all(y_face_result == 0), "X-components should not affect y-faces"
        assert torch.any(x_face_result != 0), "X-components should affect x-faces"

        # Y-components should only affect y-faces
        y_cell_data = torch.zeros_like(cell_vector_data)
        y_cell_data[mesh.n_cells :] = 1.0  # Only y-components
        y_result = ave_op @ y_cell_data

        x_face_result_y = y_result[: mesh.n_faces_x]
        y_face_result_y = y_result[mesh.n_faces_x :]

        assert torch.all(x_face_result_y == 0), "Y-components should not affect x-faces"
        assert torch.any(y_face_result_y != 0), "Y-components should affect y-faces"

    def test_average_cell_vector_to_face_3d(self, device):
        """Test 3D cell-vector-to-face averaging operator."""
        mesh = create_uniform_tensor_mesh(4, dim=3, device=device)

        # Create test cell vector data (x, y, z components)
        torch.manual_seed(42)
        cell_vector_data = torch.randn(
            mesh.n_cells * 3, device=device, dtype=mesh.dtype
        )

        # Apply vector averaging
        ave_op = mesh.average_cell_vector_to_face
        result = ave_op @ cell_vector_data

        # Check dimensions
        assert result.shape == (
            mesh.n_faces,
        ), f"Expected shape {(mesh.n_faces,)}, got {result.shape}"

        # Test block diagonal structure
        # X-components should only affect x-faces
        x_cell_data = torch.zeros_like(cell_vector_data)
        x_cell_data[: mesh.n_cells] = 1.0  # Only x-components
        x_result = ave_op @ x_cell_data

        x_face_result = x_result[: mesh.n_faces_x]
        y_face_result = x_result[mesh.n_faces_x : mesh.n_faces_x + mesh.n_faces_y]
        z_face_result = x_result[mesh.n_faces_x + mesh.n_faces_y :]

        assert torch.all(y_face_result == 0), "X-components should not affect y-faces"
        assert torch.all(z_face_result == 0), "X-components should not affect z-faces"
        assert torch.any(x_face_result != 0), "X-components should affect x-faces"

    def test_average_edge_to_cell_vector_2d(self, device):
        """Test 2D edge-to-cell vector averaging operator."""
        mesh = create_uniform_tensor_mesh(8, dim=2, device=device)

        # Create test edge data
        torch.manual_seed(42)
        edge_data = torch.randn(mesh.n_edges, device=device, dtype=mesh.dtype)

        # Apply vector averaging
        ave_op = mesh.average_edge_to_cell_vector
        result = ave_op @ edge_data

        # Check dimensions
        assert result.shape == (
            mesh.n_cells * 2,
        ), f"Expected shape {(mesh.n_cells * 2,)}, got {result.shape}"

        # Test block diagonal structure
        # X-edges should only affect x-components
        x_edge_data = torch.zeros_like(edge_data)
        x_edge_data[: mesh.n_edges_x] = 1.0
        x_result = ave_op @ x_edge_data

        x_components = x_result[: mesh.n_cells]
        y_components = x_result[mesh.n_cells :]

        assert torch.all(y_components == 0), "X-edges should not affect y-components"
        assert torch.any(x_components != 0), "X-edges should affect x-components"

        # Y-edges should only affect y-components
        y_edge_data = torch.zeros_like(edge_data)
        y_edge_data[mesh.n_edges_x :] = 1.0
        y_result = ave_op @ y_edge_data

        x_components_y = y_result[: mesh.n_cells]
        y_components_y = y_result[mesh.n_cells :]

        assert torch.all(x_components_y == 0), "Y-edges should not affect x-components"
        assert torch.any(y_components_y != 0), "Y-edges should affect y-components"

    def test_average_edge_to_cell_vector_3d(self, device):
        """Test 3D edge-to-cell vector averaging operator."""
        mesh = create_uniform_tensor_mesh(4, dim=3, device=device)

        # Create test edge data
        torch.manual_seed(42)
        edge_data = torch.randn(mesh.n_edges, device=device, dtype=mesh.dtype)

        # Apply vector averaging
        ave_op = mesh.average_edge_to_cell_vector
        result = ave_op @ edge_data

        # Check dimensions
        assert result.shape == (
            mesh.n_cells * 3,
        ), f"Expected shape {(mesh.n_cells * 3,)}, got {result.shape}"

        # Test block diagonal structure
        # X-edges should only affect x-components
        x_edge_data = torch.zeros_like(edge_data)
        x_edge_data[: mesh.n_edges_x] = 1.0
        x_result = ave_op @ x_edge_data

        x_components = x_result[: mesh.n_cells]
        y_components = x_result[mesh.n_cells : 2 * mesh.n_cells]
        z_components = x_result[2 * mesh.n_cells :]

        assert torch.all(y_components == 0), "X-edges should not affect y-components"
        assert torch.all(z_components == 0), "X-edges should not affect z-components"
        assert torch.any(x_components != 0), "X-edges should affect x-components"

    def test_vector_averaging_consistency(self, device):
        """Test consistency between scalar and vector averaging operators."""
        mesh = create_uniform_tensor_mesh(8, dim=2, device=device)

        # Create test data
        torch.manual_seed(42)
        face_data = torch.randn(mesh.n_faces, device=device, dtype=mesh.dtype)

        # Test that applying scalar operators separately is equivalent to vector operator
        # For face to cell vector
        fx_result = mesh.average_face_x_to_cell @ face_data[: mesh.n_faces_x]
        fy_result = mesh.average_face_y_to_cell @ face_data[mesh.n_faces_x :]

        expected_vector_result = torch.cat([fx_result, fy_result])
        actual_vector_result = mesh.average_face_to_cell_vector @ face_data

        # They should be identical
        assert torch.allclose(
            expected_vector_result, actual_vector_result, rtol=1e-10
        ), "Vector averaging should be equivalent to concatenated scalar averaging"

    def test_vector_averaging_sparsity(self, device):
        """Test sparsity properties of vector averaging operators."""
        mesh = create_uniform_tensor_mesh(6, dim=2, device=device)

        # Get operators
        face_to_cell_vec = mesh.average_face_to_cell_vector
        cell_vec_to_face = mesh.average_cell_vector_to_face
        edge_to_cell_vec = mesh.average_edge_to_cell_vector

        # Check sparsity (should be reasonable for averaging operators)
        f2c_sparsity = face_to_cell_vec._nnz() / (
            face_to_cell_vec.shape[0] * face_to_cell_vec.shape[1]
        )
        c2f_sparsity = cell_vec_to_face._nnz() / (
            cell_vec_to_face.shape[0] * cell_vec_to_face.shape[1]
        )
        e2c_sparsity = edge_to_cell_vec._nnz() / (
            edge_to_cell_vec.shape[0] * edge_to_cell_vec.shape[1]
        )

        print(f"Face-to-cell vector sparsity: {f2c_sparsity:.4f}")
        print(f"Cell-vector-to-face sparsity: {c2f_sparsity:.4f}")
        print(f"Edge-to-cell vector sparsity: {e2c_sparsity:.4f}")

        # Sparsity should be reasonable (not too dense, not empty)
        assert (
            0.01 < f2c_sparsity < 0.5
        ), f"Face-to-cell vector sparsity {f2c_sparsity} seems unreasonable"
        assert (
            0.01 < c2f_sparsity < 0.5
        ), f"Cell-vector-to-face sparsity {c2f_sparsity} seems unreasonable"
        assert (
            0.01 < e2c_sparsity < 0.5
        ), f"Edge-to-cell vector sparsity {e2c_sparsity} seems unreasonable"


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
class TestMimeticProperties:
    """Test mimetic properties of differential operators."""

    def test_div_curl_property_3d(self, device):
        """Test that div(curl(v)) = 0 for any vector field v."""
        mesh = create_uniform_tensor_mesh(12, dim=3, device=device)

        # Random edge field
        torch.manual_seed(42)  # For reproducibility
        v = torch.randn(mesh.n_edges, device=device, dtype=mesh.dtype)

        # Compute div(curl(v))
        curl_v = mesh.edge_curl @ v
        div_curl_v = mesh.face_divergence @ curl_v

        # Should be approximately zero
        rel_err = torch.norm(div_curl_v).item() / torch.norm(v).item()
        print(f"3D Div(Curl) relative error: {rel_err}")

        assert rel_err < 1e-11  # Should be very small (mimetic property)

    def test_curl_grad_property_3d(self, device):
        """Test that curl(grad(phi)) = 0 for any scalar field phi."""
        mesh = create_uniform_tensor_mesh(12, dim=3, device=device)

        # Random nodal field
        torch.manual_seed(42)  # For reproducibility
        phi = torch.randn(mesh.n_nodes, device=device, dtype=mesh.dtype)

        # Compute curl(grad(phi))
        grad_phi = mesh.nodal_gradient @ phi
        curl_grad_phi = mesh.edge_curl @ grad_phi

        # Should be approximately zero
        rel_err = torch.norm(curl_grad_phi).item() / torch.norm(phi).item()
        print(f"3D Curl(Grad) relative error: {rel_err}")

        assert rel_err < 1e-11  # Should be very small (mimetic property)

    def test_div_curl_property_2d(self, device):
        """Test mimetic properties in 2D."""
        mesh = create_uniform_tensor_mesh(16, dim=2, device=device)

        # Random edge field
        torch.manual_seed(42)  # For reproducibility
        v = torch.randn(mesh.n_edges, device=device, dtype=mesh.dtype)

        # In 2D, curl produces a scalar field (faces are cells)
        curl_v = mesh.edge_curl @ v

        # For 2D, the curl is to cell centers, so we can't directly apply face divergence
        # This is more of a dimensional consistency check
        print(f"2D Edge curl output shape: {curl_v.shape}")
        print(f"2D Expected cell count: {mesh.n_cells}")

        assert curl_v.shape[0] == mesh.n_cells


class TestAveraging1D(OrderTest):
    name = "Averaging 1D"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) @ self.getHere(self.M)
        if hasattr(num, "is_sparse") and num.is_sparse:
            num = num.to_dense()
        err = torch.linalg.norm((self.getThere(self.M) - num), ord=float("inf"))
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 1D: N2CC"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.nodes)
        self.getThere = lambda M: fun(M.cell_centers)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_exactN2F(self):
        self.name = "Averaging 1D: N2F"
        fun = lambda x: torch.cos(x)
        M, _ = setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveN2F @ fun(M.nodes)
        if v1.is_sparse:
            v1 = v1.to_dense()
        v2 = fun(M.faces)
        assert torch.allclose(v1, v2)

    def test_orderN2E(self):
        self.name = "Averaging 1D: N2E"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.nodes)
        self.getThere = lambda M: fun(M.edges)
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 1D: F2CC"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.faces)
        self.getThere = lambda M: fun(M.cell_centers)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 1D: F2CCV"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.faces)
        self.getThere = lambda M: fun(M.cell_centers)
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 1D: CC2F"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.cell_centers)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_exactE2CC(self):
        self.name = "Averaging 1D: E2CC"
        fun = lambda x: torch.cos(x)
        M, _ = setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveE2CC @ fun(M.edges)
        if v1.is_sparse:
            v1 = v1.to_dense()
        v2 = fun(M.cell_centers)
        assert torch.allclose(v1, v2)

    def test_exactE2CCV(self):
        self.name = "Averaging 1D: E2CCV"
        fun = lambda x: torch.cos(x)
        M, _ = setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveE2CCV @ fun(M.edges)
        if v1.is_sparse:
            v1 = v1.to_dense()
        v2 = fun(M.cell_centers)
        assert torch.allclose(v1, v2)

    def test_exactCC2E(self):
        self.name = "Averaging 1D: cell_centers_to_edges"
        fun = lambda x: torch.cos(x)
        M, _ = setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.average_cell_to_edge @ fun(M.cell_centers)
        if v1.is_sparse:
            v1 = v1.to_dense()
        v2 = fun(M.edges)
        assert torch.allclose(v1, v2)

    def test_orderCC2FV(self):
        self.name = "Averaging 1D: CC2FV"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.cell_centers)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderE2FV(self):
        self.name = "Averaging 1D: E2FV"
        fun = lambda x: torch.cos(x)
        self.getHere = lambda M: fun(M.edges)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.average_edge_to_face
        self.expectedOrders = 1  # Extrapolation operator has order 1
        self.orderTest()
        self.expectedOrders = 2  # Reset for other tests


class TestAveraging2D(OrderTest):
    name = "Averaging 2D"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) @ self.getHere(self.M)
        if hasattr(num, "is_sparse") and num.is_sparse:
            num = num.to_dense()
        err = torch.linalg.norm((self.getThere(self.M) - num), ord=float("inf"))
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 2D: N2CC"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: call2(fun, M.nodes)
        self.getThere = lambda M: call2(fun, M.cell_centers)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2F(self):
        self.name = "Averaging 2D: N2F"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: call2(fun, M.nodes)
        self.getThere = lambda M: torch.cat(
            [call2(fun, M.faces_x), call2(fun, M.faces_y)]
        )
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 2D: N2E"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: call2(fun, M.nodes)
        self.getThere = lambda M: torch.cat(
            [call2(fun, M.edges_x), call2(fun, M.edges_y)]
        )
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 2D: F2CC"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: torch.cat(
            [call2(fun, M.faces_x), call2(fun, M.faces_y)]
        )
        self.getThere = lambda M: call2(fun, M.cell_centers)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 2D: F2CCV"
        funX = lambda x, y: (torch.cos(x) + torch.sin(y))
        funY = lambda x, y: (torch.cos(y) * torch.sin(x))
        self.getHere = lambda M: torch.cat(
            [call2(funX, M.faces_x), call2(funY, M.faces_y)]
        )
        self.getThere = lambda M: torch.cat(
            [call2(funX, M.cell_centers), call2(funY, M.cell_centers)]
        )
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 2D: CC2F"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: call2(fun, M.cell_centers)
        self.getThere = lambda M: torch.cat(
            [call2(fun, M.faces_x), call2(fun, M.faces_y)]
        )
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])

    def test_orderE2CC(self):
        self.name = "Averaging 2D: E2CC"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: torch.cat(
            [call2(fun, M.edges_x), call2(fun, M.edges_y)]
        )
        self.getThere = lambda M: call2(fun, M.cell_centers)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 2D: E2CCV"
        funX = lambda x, y: (torch.cos(x) + torch.sin(y))
        funY = lambda x, y: (torch.cos(y) * torch.sin(x))
        self.getHere = lambda M: torch.cat(
            [call2(funX, M.edges_x), call2(funY, M.edges_y)]
        )
        self.getThere = lambda M: torch.cat(
            [call2(funX, M.cell_centers), call2(funY, M.cell_centers)]
        )
        self.getAve = lambda M: M.aveE2CCV
        self.orderTest()

    def test_orderCC2E(self):
        self.name = "Averaging 2D: cell_centers_to_edges"
        fun = lambda x, y: (torch.cos(x) + torch.sin(y))
        self.getHere = lambda M: call2(fun, M.cell_centers)
        self.getThere = lambda M: call2(fun, M.edges)
        self.getAve = lambda M: M.average_cell_to_edge
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])

    def test_orderCC2FV(self):
        self.name = "Averaging 2D: CC2FV"
        funX = lambda x, y: (torch.cos(x) + torch.sin(y))
        funY = lambda x, y: (torch.cos(y) * torch.sin(x))
        self.getHere = lambda M: torch.cat(
            [call2(funX, M.cell_centers), call2(funY, M.cell_centers)]
        )
        self.getThere = lambda M: torch.cat(
            [call2(funX, M.faces_x), call2(funY, M.faces_y)]
        )
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])


class TestAveraging3D(OrderTest):
    name = "Averaging 3D"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) @ self.getHere(self.M)
        if hasattr(num, "is_sparse") and num.is_sparse:
            num = num.to_dense()
        err = torch.linalg.norm((self.getThere(self.M) - num), ord=float("inf"))
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 3D: N2CC"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: call3(fun, M.nodes)
        self.getThere = lambda M: call3(fun, M.cell_centers)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2F(self):
        self.name = "Averaging 3D: N2F"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: call3(fun, M.nodes)
        self.getThere = lambda M: torch.cat(
            [call3(fun, M.faces_x), call3(fun, M.faces_y), call3(fun, M.faces_z)]
        )
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 3D: N2E"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: call3(fun, M.nodes)
        self.getThere = lambda M: torch.cat(
            [call3(fun, M.edges_x), call3(fun, M.edges_y), call3(fun, M.edges_z)]
        )
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 3D: F2CC"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: torch.cat(
            [call3(fun, M.faces_x), call3(fun, M.faces_y), call3(fun, M.faces_z)]
        )
        self.getThere = lambda M: call3(fun, M.cell_centers)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 3D: F2CCV"
        funX = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        funY = lambda x, y, z: (torch.cos(x) + torch.sin(y) * torch.exp(z))
        funZ = lambda x, y, z: (torch.cos(x) * torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: torch.cat(
            [call3(funX, M.faces_x), call3(funY, M.faces_y), call3(funZ, M.faces_z)]
        )
        self.getThere = lambda M: torch.cat(
            [
                call3(funX, M.cell_centers),
                call3(funY, M.cell_centers),
                call3(funZ, M.cell_centers),
            ]
        )
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderE2CC(self):
        self.name = "Averaging 3D: E2CC"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: torch.cat(
            [call3(fun, M.edges_x), call3(fun, M.edges_y), call3(fun, M.edges_z)]
        )
        self.getThere = lambda M: call3(fun, M.cell_centers)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 3D: E2CCV"
        funX = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        funY = lambda x, y, z: (torch.cos(x) + torch.sin(y) * torch.exp(z))
        funZ = lambda x, y, z: (torch.cos(x) * torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: torch.cat(
            [call3(funX, M.edges_x), call3(funY, M.edges_y), call3(funZ, M.edges_z)]
        )
        self.getThere = lambda M: torch.cat(
            [
                call3(funX, M.cell_centers),
                call3(funY, M.cell_centers),
                call3(funZ, M.cell_centers),
            ]
        )
        self.getAve = lambda M: M.aveE2CCV
        self.expectedOrders = torch.tensor([2.0])
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 3D: CC2F"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: call3(fun, M.cell_centers)
        self.getThere = lambda M: torch.cat(
            [call3(fun, M.faces_x), call3(fun, M.faces_y), call3(fun, M.faces_z)]
        )
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])

    def test_orderCC2E(self):
        self.name = "Averaging 3D: CC2E"
        fun = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: call3(fun, M.cell_centers)
        self.getThere = lambda M: call3(fun, M.edges)
        self.getAve = lambda M: M.average_cell_to_edge
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])

    def test_orderCCV2F(self):
        self.name = "Averaging 3D: CC2FV"
        funX = lambda x, y, z: (torch.cos(x) + torch.sin(y) + torch.exp(z))
        funY = lambda x, y, z: (torch.cos(x) + torch.sin(y) * torch.exp(z))
        funZ = lambda x, y, z: (torch.cos(x) * torch.sin(y) + torch.exp(z))
        self.getHere = lambda M: torch.cat(
            [
                call3(funX, M.cell_centers),
                call3(funY, M.cell_centers),
                call3(funZ, M.cell_centers),
            ]
        )
        self.getThere = lambda M: torch.cat(
            [call3(funX, M.faces_x), call3(funY, M.faces_y), call3(funZ, M.faces_z)]
        )
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = torch.tensor([2.0]) / 2.0
        self.orderTest()
        self.expectedOrders = torch.tensor([2.0])


if __name__ == "__main__":
    pytest.main([__file__])
