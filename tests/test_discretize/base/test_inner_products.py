import torch
import pytest
from simpegtorch.discretize import TensorMesh


# Device fixtures for testing across different devices
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Pytest fixture to test on CPU and CUDA (if available)."""
    return torch.device(request.param)


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float64


class TestBasicInnerProducts:
    """Basic tests for inner product functionality."""

    def test_face_inner_product_1d(self, device, dtype):
        """Test 1D face inner product with isotropic model."""
        mesh = TensorMesh([10], device=device, dtype=dtype)

        # Test with default (ones) model
        A_default = mesh.get_face_inner_product()
        assert A_default.shape == (mesh.n_faces, mesh.n_faces)
        assert A_default.is_sparse
        assert A_default.device == device

        # Test with scalar model
        sigma = 2.0
        A_scalar = mesh.get_face_inner_product(sigma)
        assert A_scalar.shape == (mesh.n_faces, mesh.n_faces)
        assert A_scalar.device == device

        # Test with vector model
        sigma_vec = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 3.0
        A_vector = mesh.get_face_inner_product(sigma_vec)
        assert A_vector.shape == (mesh.n_faces, mesh.n_faces)
        assert A_vector.device == device

    def test_face_inner_product_2d(self, device, dtype):
        """Test 2D face inner product with different tensor types."""
        mesh = TensorMesh([8, 8], device=device, dtype=dtype)

        # Isotropic
        sigma_iso = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0
        A_iso = mesh.get_face_inner_product(sigma_iso)
        assert A_iso.shape == (mesh.n_faces, mesh.n_faces)
        assert A_iso.device == device

        # Anisotropic (diagonal)
        sigma_aniso = (
            torch.ones(mesh.n_cells * 2, device=device, dtype=dtype) * 2.0
        )  # [sigma_x, sigma_y]
        A_aniso = mesh.get_face_inner_product(sigma_aniso)
        assert A_aniso.shape == (mesh.n_faces, mesh.n_faces)
        assert A_aniso.device == device

        # Full tensor
        sigma_tensor = torch.ones(
            mesh.n_cells * 3, device=device, dtype=dtype
        )  # [sigma_xx, sigma_yy, sigma_xy]
        A_tensor = mesh.get_face_inner_product(sigma_tensor)
        assert A_tensor.shape == (mesh.n_faces, mesh.n_faces)
        assert A_tensor.device == device

    def test_face_inner_product_3d(self, device, dtype):
        """Test 3D face inner product with different tensor types."""
        mesh = TensorMesh([4, 4, 4], device=device, dtype=dtype)

        # Isotropic
        sigma_iso = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0
        A_iso = mesh.get_face_inner_product(sigma_iso)
        assert A_iso.shape == (mesh.n_faces, mesh.n_faces)
        assert A_iso.device == device

        # Anisotropic (diagonal)
        sigma_aniso = (
            torch.ones(mesh.n_cells * 3, device=device, dtype=dtype) * 2.0
        )  # [sigma_x, sigma_y, sigma_z]
        A_aniso = mesh.get_face_inner_product(sigma_aniso)
        assert A_aniso.shape == (mesh.n_faces, mesh.n_faces)
        assert A_aniso.device == device

        # Full tensor
        sigma_tensor = torch.ones(
            mesh.n_cells * 6, device=device, dtype=dtype
        )  # [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz]
        A_tensor = mesh.get_face_inner_product(sigma_tensor)
        assert A_tensor.shape == (mesh.n_faces, mesh.n_faces)
        assert A_tensor.device == device

    def test_edge_inner_product_2d(self, device, dtype):
        """Test 2D edge inner product."""
        mesh = TensorMesh([8, 8], device=device, dtype=dtype)

        # Isotropic
        sigma_iso = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0
        A_iso = mesh.get_edge_inner_product(sigma_iso)
        assert A_iso.shape == (mesh.n_edges, mesh.n_edges)
        assert A_iso.device == device

        # Anisotropic
        sigma_aniso = torch.ones(mesh.n_cells * 2, device=device, dtype=dtype) * 2.0
        A_aniso = mesh.get_edge_inner_product(sigma_aniso)
        assert A_aniso.shape == (mesh.n_edges, mesh.n_edges)
        assert A_aniso.device == device

    def test_edge_inner_product_3d(self, device, dtype):
        """Test 3D edge inner product."""
        mesh = TensorMesh([4, 4, 4], device=device, dtype=dtype)

        # Isotropic
        sigma_iso = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0
        A_iso = mesh.get_edge_inner_product(sigma_iso)
        assert A_iso.shape == (mesh.n_edges, mesh.n_edges)
        assert A_iso.device == device

        # Anisotropic
        sigma_aniso = torch.ones(mesh.n_cells * 3, device=device, dtype=dtype) * 2.0
        A_aniso = mesh.get_edge_inner_product(sigma_aniso)
        assert A_aniso.shape == (mesh.n_edges, mesh.n_edges)
        assert A_aniso.device == device

    def test_invert_model_flag(self, device, dtype):
        """Test the invert_model flag."""
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)
        sigma = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0

        # Normal
        # A1 = mesh.get_face_inner_product(sigma)

        # With invert_model=True (should be equivalent to passing 1/sigma)
        A2 = mesh.get_face_inner_product(sigma, invert_model=True)
        A3 = mesh.get_face_inner_product(1.0 / sigma)

        # A2 and A3 should be approximately equal
        diff = (A2 - A3).to_dense().abs().max()
        assert diff < 1e-10
        assert A2.device == device
        assert A3.device == device

    def test_invert_matrix_flag(self, device, dtype):
        """Test the invert_matrix flag."""
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)
        sigma = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0

        # Normal
        A = mesh.get_face_inner_product(sigma)

        # With invert_matrix=True
        A_inv = mesh.get_face_inner_product(sigma, invert_matrix=True)

        # Should be approximately inverse (for isotropic case)
        identity_approx = torch.sparse.mm(A.to_sparse_coo(), A_inv.to_sparse_coo())
        diag_vals = identity_approx.to_dense().diag()

        # Diagonal should be close to 1
        assert torch.allclose(diag_vals, torch.ones_like(diag_vals), atol=1e-10)
        assert A.device == device
        assert A_inv.device == device

    def test_edge_inner_product_surface_1d(self, device, dtype):
        """Test 1D edge inner product surface with isotropic model."""
        mesh = TensorMesh([10], device=device, dtype=dtype)

        # Test with default (ones) model
        A_default = mesh.get_edge_inner_product_surface()
        assert A_default.shape == (mesh.n_edges, mesh.n_edges)
        assert A_default.is_sparse
        assert A_default.device == device

        # Test with scalar model
        tau = 2.0
        A_scalar = mesh.get_edge_inner_product_surface(tau)
        assert A_scalar.shape == (mesh.n_edges, mesh.n_edges)
        assert A_scalar.device == device

        # Test with vector model
        tau_vec = torch.ones(mesh.n_edges, device=device, dtype=dtype) * 3.0
        A_vector = mesh.get_edge_inner_product_surface(tau_vec)
        assert A_vector.shape == (mesh.n_edges, mesh.n_edges)
        assert A_vector.device == device

    def test_edge_inner_product_surface_2d(self, device, dtype):
        """Test 2D edge inner product surface."""
        mesh = TensorMesh([8, 8], device=device, dtype=dtype)

        # Test with default model
        A_default = mesh.get_edge_inner_product_surface()
        assert A_default.shape == (mesh.n_edges, mesh.n_edges)
        assert A_default.device == device

        # Test with edge-based model
        tau = torch.ones(mesh.n_edges, device=device, dtype=dtype) * 2.0
        A = mesh.get_edge_inner_product_surface(tau)
        assert A.shape == (mesh.n_edges, mesh.n_edges)
        assert A.device == device

    def test_edge_inner_product_surface_3d(self, device, dtype):
        """Test 3D edge inner product surface."""
        mesh = TensorMesh([4, 4, 4], device=device, dtype=dtype)

        # Test with default model
        A_default = mesh.get_edge_inner_product_surface()
        assert A_default.shape == (mesh.n_edges, mesh.n_edges)
        assert A_default.device == device

        # Test with edge-based model
        tau = torch.ones(mesh.n_edges, device=device, dtype=dtype) * 2.0
        A = mesh.get_edge_inner_product_surface(tau)
        assert A.shape == (mesh.n_edges, mesh.n_edges)
        assert A.device == device

    def test_edge_inner_product_surface_flags(self, device, dtype):
        """Test edge inner product surface with invert flags."""
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)
        tau = torch.ones(mesh.n_edges, device=device, dtype=dtype) * 2.0

        # Normal
        A = mesh.get_edge_inner_product_surface(tau)

        # With invert_model=True (should be equivalent to passing 1/tau)
        A2 = mesh.get_edge_inner_product_surface(tau, invert_model=True)
        A3 = mesh.get_edge_inner_product_surface(1.0 / tau)

        # A2 and A3 should be approximately equal
        diff = (A2 - A3).to_dense().abs().max()
        assert diff < 1e-10
        assert A2.device == device
        assert A3.device == device

        # With invert_matrix=True
        A_inv = mesh.get_edge_inner_product_surface(tau, invert_matrix=True)

        # Should be approximately inverse
        identity_approx = torch.sparse.mm(A.to_sparse_coo(), A_inv.to_sparse_coo())
        diag_vals = identity_approx.to_dense().diag()

        # Diagonal should be close to 1
        assert torch.allclose(diag_vals, torch.ones_like(diag_vals), atol=1e-10)
        assert A.device == device
        assert A_inv.device == device


class TestInnerProductIntegration:
    """Integration tests comparing analytical and numerical results."""

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_1d_face_integration(self, device):
        """Test 1D face inner product integration."""
        device = torch.device(device)
        dtype = torch.float64
        mesh = TensorMesh([32], device=device, dtype=dtype)  # Fine mesh for accuracy

        # Define function: f(x) = x^2 + 60 (y=12, z=5 from original)
        def ex_func(x):
            return x**2 + 60

        # Define conductivity: sigma(x) = x*12 + 1
        def sigma_func(x):
            return x * 12 + 1

        # Evaluate functions at grid points
        sigma = torch.tensor(
            [sigma_func(x) for x in mesh.cell_centers_x], dtype=dtype, device=device
        )
        F = torch.tensor([ex_func(x) for x in mesh.faces_x], dtype=dtype, device=device)

        # Get inner product matrix
        A = mesh.get_face_inner_product(sigma)

        # Numerical integration
        numeric = torch.dot(F, A.to_dense() @ F)

        # Analytical result (computed with sympy): 128011/5 = 25602.2
        analytic = 128011.0 / 5

        # Check relative error
        rel_error = abs(numeric.item() - analytic) / analytic
        assert rel_error < 0.01  # 1% tolerance

    def test_2d_face_integration_isotropic(self):
        """Test 2D face inner product with isotropic conductivity."""
        mesh = TensorMesh([16, 16])  # Moderate resolution

        # Define functions
        def ex_func(x, y):
            return x**2 + y * 5  # z=5

        def ey_func(x, y):
            return 25 * x + y * 5  # z^2=25, z=5

        def sigma_func(x, y):
            return x * y + 1

        # Evaluate at grid centers and faces
        Gc = mesh.cell_centers
        sigma = torch.tensor([sigma_func(x, y) for x, y in Gc], dtype=torch.float64)

        # Face vectors
        Fx = torch.tensor([ex_func(x, y) for x, y in mesh.faces_x], dtype=torch.float64)
        Fy = torch.tensor([ey_func(x, y) for x, y in mesh.faces_y], dtype=torch.float64)
        F = torch.cat([Fx, Fy])

        # Inner product
        A = mesh.get_face_inner_product(sigma)
        numeric = torch.dot(F, A.to_dense() @ F)

        # Analytical result (from sympy): 144877/360
        analytic = 144877.0 / 360

        rel_error = abs(numeric.item() - analytic) / analytic
        assert rel_error < 0.05  # 5% tolerance for 2D

    def test_3d_face_integration_isotropic(self):
        """Test 3D face inner product with isotropic conductivity."""
        mesh = TensorMesh([8, 8, 8])  # Coarser for 3D

        # Define functions
        def ex_func(x, y, z):
            return x**2 + y * z

        def ey_func(x, y, z):
            return z**2 * x + y * z

        def ez_func(x, y, z):
            return y**2 + x * z

        def sigma_func(x, y, z):
            return x * y + 1

        # Evaluate functions
        Gc = mesh.cell_centers
        sigma = torch.tensor(
            [sigma_func(x, y, z) for x, y, z in Gc], dtype=torch.float64
        )

        # Face vectors
        Fx = torch.tensor(
            [ex_func(x, y, z) for x, y, z in mesh.faces_x], dtype=torch.float64
        )
        Fy = torch.tensor(
            [ey_func(x, y, z) for x, y, z in mesh.faces_y], dtype=torch.float64
        )
        Fz = torch.tensor(
            [ez_func(x, y, z) for x, y, z in mesh.faces_z], dtype=torch.float64
        )
        F = torch.cat([Fx, Fy, Fz])

        # Inner product
        A = mesh.get_face_inner_product(sigma)
        numeric = torch.dot(F, A.to_dense() @ F)

        # Analytical result (from sympy): 647/360
        analytic = 647.0 / 360

        rel_error = abs(numeric.item() - analytic) / analytic
        assert rel_error < 0.1  # 10% tolerance for coarse 3D


class TestInnerProductProperties:
    """Test mathematical properties of inner products."""

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_symmetry(self, device):
        """Test that inner product matrices are symmetric."""
        device = torch.device(device)
        dtype = torch.float64
        mesh = TensorMesh([8, 8], device=device, dtype=dtype)
        sigma = torch.ones(mesh.n_cells, device=device, dtype=dtype) * 2.0

        A = mesh.get_face_inner_product(sigma)
        A_dense = A.to_dense()

        # Check symmetry
        assert torch.allclose(A_dense, A_dense.T, atol=1e-12)
        assert A.device == device

    def test_positive_definiteness(self):
        """Test that inner product matrices are positive definite."""
        mesh = TensorMesh([6, 6])
        sigma = torch.ones(mesh.n_cells) * 2.0

        A = mesh.get_face_inner_product(sigma)
        A_dense = A.to_dense()

        # Check that all eigenvalues are positive
        eigenvals = torch.linalg.eigvals(A_dense).real
        assert torch.all(eigenvals > 0)

    def test_scaling_property(self):
        """Test that scaling sigma scales the inner product matrix."""
        mesh = TensorMesh([6, 6])
        sigma = torch.ones(mesh.n_cells) * 2.0
        scale_factor = 3.0

        A1 = mesh.get_face_inner_product(sigma)
        A2 = mesh.get_face_inner_product(sigma * scale_factor)

        # A2 should be approximately scale_factor * A1
        diff = (A2 - scale_factor * A1).to_dense().abs().max()
        assert diff < 1e-10


class TestInnerProductProjectionMatrices:
    """Test the projection matrix functions."""

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_face_projection_1d(self, device):
        """Test 1D face projection matrices."""
        device = torch.device(device)
        dtype = torch.float64
        mesh = TensorMesh([4], device=device, dtype=dtype)

        # Test _getFacePx
        Px = mesh._getFacePx()

        # Test both face orientations
        P_fXm = Px("fXm")
        P_fXp = Px("fXp")

        assert P_fXm.shape == (mesh.n_cells, mesh.n_faces)
        assert P_fXp.shape == (mesh.n_cells, mesh.n_faces)
        assert P_fXm.is_sparse
        assert P_fXp.is_sparse
        assert P_fXm.device == device
        assert P_fXp.device == device

    def test_face_projection_2d(self):
        """Test 2D face projection matrices."""
        mesh = TensorMesh([4, 4])

        # Test _getFacePxx
        Pxx = mesh._getFacePxx()

        # Test different face combinations
        P = Pxx("fXm", "fYm")
        assert P.shape == (2 * mesh.n_cells, mesh.n_faces)
        assert P.is_sparse

    def test_face_projection_3d(self):
        """Test 3D face projection matrices."""
        mesh = TensorMesh([4, 4, 4])

        # Test _getFacePxxx
        Pxxx = mesh._getFacePxxx()

        # Test different face combinations
        P = Pxxx("fXm", "fYm", "fZm")
        assert P.shape == (3 * mesh.n_cells, mesh.n_faces)
        assert P.is_sparse

    def test_edge_projection_1d(self):
        """Test 1D edge projection matrices."""
        mesh = TensorMesh([4])

        # Test _getEdgePx
        Px = mesh._getEdgePx()
        P = Px("eX0")

        assert P.shape == (mesh.n_cells, mesh.n_cells)  # Identity for 1D edges
        assert P.is_sparse

    def test_edge_projection_2d(self):
        """Test 2D edge projection matrices."""
        mesh = TensorMesh([4, 4])

        # Test _getEdgePxx
        Pxx = mesh._getEdgePxx()
        P = Pxx("eX0", "eY0")

        assert P.shape == (2 * mesh.n_cells, mesh.n_edges)
        assert P.is_sparse

    def test_edge_projection_3d(self):
        """Test 3D edge projection matrices."""
        mesh = TensorMesh([3, 3, 3])

        # Test _getEdgePxxx
        Pxxx = mesh._getEdgePxxx()
        P = Pxxx("eX0", "eY0", "eZ0")

        assert P.shape == (3 * mesh.n_cells, mesh.n_edges)
        assert P.is_sparse


class TestInnerProductDerivatives:
    """Test derivatives of inner products using PyTorch automatic differentiation."""

    def test_face_inner_product_derivative_1d(self, device, dtype):
        """Test 1D face inner product derivative with respect to model parameters."""
        device = torch.device(device)
        mesh = TensorMesh([8], device=device, dtype=dtype)

        # Create model parameter with gradient tracking
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 2.0
        )

        # Create test vector
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute inner product: u^T @ A(sigma) @ u
        A = mesh.get_face_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient using autograd
        grad_auto = torch.autograd.grad(quadratic_form, sigma, create_graph=True)[0]

        # Verify gradient shape and properties
        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()
        assert not torch.isinf(grad_auto).any()

    def test_face_inner_product_derivative_2d_isotropic(self, device, dtype):
        """Test 2D face inner product derivative for isotropic model."""
        device = torch.device(device)
        mesh = TensorMesh([6, 6], device=device, dtype=dtype)

        # Isotropic model with gradient tracking
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 3.0
        )

        # Test vectors
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)
        v = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute bilinear form: u^T @ A(sigma) @ v
        A = mesh.get_face_inner_product(sigma)
        bilinear_form = torch.dot(u, A.to_dense() @ v)

        # Compute gradient
        grad_auto = torch.autograd.grad(bilinear_form, sigma, create_graph=True)[0]

        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_face_inner_product_derivative_2d_anisotropic(self, device, dtype):
        """Test 2D face inner product derivative for anisotropic model."""
        device = torch.device(device)
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Anisotropic model: [sigma_x, sigma_y] for each cell
        sigma = (
            torch.ones(mesh.n_cells * 2, device=device, dtype=dtype, requires_grad=True)
            * 2.0
        )

        # Test vector
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_face_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient
        grad_auto = torch.autograd.grad(quadratic_form, sigma, create_graph=True)[0]

        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_face_inner_product_derivative_3d_isotropic(self, device, dtype):
        """Test 3D face inner product derivative for isotropic model."""
        device = torch.device(device)
        mesh = TensorMesh([3, 3, 3], device=device, dtype=dtype)

        # Isotropic model
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 1.5
        )

        # Test vector
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_face_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient
        grad_auto = torch.autograd.grad(quadratic_form, sigma, create_graph=True)[0]

        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_edge_inner_product_derivative_2d(self, device, dtype):
        """Test 2D edge inner product derivative."""
        device = torch.device(device)
        mesh = TensorMesh([6, 6], device=device, dtype=dtype)

        # Isotropic model for edges
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 2.5
        )

        # Test vector
        u = torch.randn(mesh.n_edges, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_edge_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient
        grad_auto = torch.autograd.grad(quadratic_form, sigma, create_graph=True)[0]

        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_edge_inner_product_derivative_3d(self, device, dtype):
        """Test 3D edge inner product derivative."""
        device = torch.device(device)
        mesh = TensorMesh([3, 3, 3], device=device, dtype=dtype)

        # Anisotropic model for edges: [sigma_x, sigma_y, sigma_z]
        sigma = (
            torch.ones(mesh.n_cells * 3, device=device, dtype=dtype, requires_grad=True)
            * 1.8
        )

        # Test vector
        u = torch.randn(mesh.n_edges, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_edge_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient
        grad_auto = torch.autograd.grad(quadratic_form, sigma, create_graph=True)[0]

        assert grad_auto.shape == sigma.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_edge_inner_product_surface_derivative(self, device, dtype):
        """Test edge inner product surface derivative."""
        device = torch.device(device)
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Edge-based model parameter
        tau = (
            torch.ones(mesh.n_edges, device=device, dtype=dtype, requires_grad=True)
            * 1.2
        )

        # Test vector
        u = torch.randn(mesh.n_edges, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_edge_inner_product_surface(tau)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Compute gradient
        grad_auto = torch.autograd.grad(quadratic_form, tau, create_graph=True)[0]

        assert grad_auto.shape == tau.shape
        assert grad_auto.device == device
        assert not torch.isnan(grad_auto).any()

    def test_gradient_flow_preservation(self, device, dtype):
        """Test that gradients flow properly through inner product operations."""
        device = torch.device(device)
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Model parameter
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 2.0
        )

        # Test vector
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute quadratic form
        A = mesh.get_face_inner_product(sigma)
        quadratic_form = torch.dot(u, A.to_dense() @ u)

        # Verify that the gradient can be computed multiple times
        grad1 = torch.autograd.grad(quadratic_form, sigma, retain_graph=True)[0]
        grad2 = torch.autograd.grad(quadratic_form, sigma, retain_graph=True)[0]

        # Gradients should be identical when computed multiple times
        assert torch.allclose(grad1, grad2, atol=1e-12)
        assert grad1.shape == sigma.shape
        assert grad1.device == device
        assert not torch.isnan(grad1).any()

    def test_derivative_linearity(self, device, dtype):
        """Test linearity property of derivatives."""
        device = torch.device(device)
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Two different model parameters
        sigma1 = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 1.0
        )
        sigma2 = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 2.0
        )

        # Scalars for linear combination
        a, b = 2.0, 3.0

        # Test vector
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Compute derivatives for linear combination: a*sigma1 + b*sigma2
        sigma_combined = a * sigma1 + b * sigma2
        A_combined = mesh.get_face_inner_product(sigma_combined)
        quad_combined = torch.dot(u, A_combined.to_dense() @ u)

        # This tests that the derivative computation works for linear combinations
        grad_combined = torch.autograd.grad(
            quad_combined, [sigma1, sigma2], create_graph=True
        )

        assert len(grad_combined) == 2
        assert grad_combined[0].shape == sigma1.shape
        assert grad_combined[1].shape == sigma2.shape
        assert grad_combined[0].device == device
        assert grad_combined[1].device == device

    def test_derivative_with_different_vector_operations(self, device, dtype):
        """Test derivatives with different vector operations."""
        device = torch.device(device)
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Model parameter
        sigma = (
            torch.ones(mesh.n_cells, device=device, dtype=dtype, requires_grad=True)
            * 1.5
        )

        # Test vectors
        u = torch.randn(mesh.n_faces, device=device, dtype=dtype)
        v = torch.randn(mesh.n_faces, device=device, dtype=dtype)

        # Get inner product matrix
        A = mesh.get_face_inner_product(sigma)
        A_dense = A.to_dense()

        # Test different operations
        operations = [
            torch.dot(u, A_dense @ u),  # Quadratic form
            torch.dot(u, A_dense @ v),  # Bilinear form
            torch.trace(A_dense),  # Trace
            torch.sum(torch.diagonal(A_dense)),  # Sum of diagonal
        ]

        for i, op in enumerate(operations):
            grad_auto = torch.autograd.grad(op, sigma, retain_graph=True)[0]
            assert grad_auto.shape == sigma.shape, f"Failed for operation {i}"
            assert grad_auto.device == device, f"Failed for operation {i}"
            assert not torch.isnan(grad_auto).any(), f"Failed for operation {i}"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_invalid_tensor_size(self, device):
        """Test that invalid tensor sizes raise appropriate errors."""
        device = torch.device(device)
        dtype = torch.float64
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Wrong size for 2D mesh
        with pytest.raises(Exception):
            wrong_size_tensor = torch.ones(
                mesh.n_cells * 5, device=device, dtype=dtype
            )  # Invalid size
            mesh.get_face_inner_product(wrong_size_tensor)

    def test_invalid_projection_type(self):
        """Test invalid projection types."""
        mesh = TensorMesh([4, 4])

        with pytest.raises(TypeError):
            mesh._getInnerProductProjectionMatrices("X", 1)  # Invalid projection type

    def test_edge_projection_invalid_edge(self):
        """Test invalid edge names."""
        mesh = TensorMesh([4])
        Px = mesh._getEdgePx()

        with pytest.raises(TypeError):
            Px("eX1")  # Invalid edge name for 1D

    def test_edge_inner_product_surface_invalid_size(self, device):
        """Test that invalid edge model sizes raise appropriate errors."""
        device = torch.device(device)
        dtype = torch.float64
        mesh = TensorMesh([4, 4], device=device, dtype=dtype)

        # Wrong size for edge model
        with pytest.raises(ValueError):
            wrong_size_tensor = torch.ones(
                mesh.n_edges + 5, device=device, dtype=dtype
            )  # Invalid size
            mesh.get_edge_inner_product_surface(wrong_size_tensor)
