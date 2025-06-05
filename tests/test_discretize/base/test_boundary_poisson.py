# flake8: noqa: E731
import torch
import unittest
from simpegtorch.discretize import utils
from simpegtorch.discretize.tests import OrderTest


MESHTYPES = ["uniformTensorMesh"]
torch.set_default_dtype(torch.float64)


class TestCC1D_InhomogeneousDirichlet(OrderTest):
    name = "1D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(torch.pi * x)
        q_fun = lambda x: -(torch.pi**2) * torch.cos(torch.pi * x)

        mesh = self.M

        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        boundary_faces = mesh.boundary_faces

        phi_bc = phi(boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)
        M_bf = mesh.boundary_face_scalar_integral

        V = utils.sdiag(mesh.cell_volumes)
        G = -torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # Since the xc_bc is known, move it to the RHS!
        A = V @ D @ MfI @ G
        rhs = (
            V @ q_ana
            - V @ D @ MfI @ torch.sparse.mm(M_bf, phi_bc.unsqueeze(-1)).squeeze()
        )

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())
        err = torch.linalg.norm((phi_test - phi_ana)) / torch.sqrt(
            torch.tensor(mesh.n_cells, dtype=torch.float32)
        )

        return err.item()

    def test_orderX(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC2D_InhomogeneousDirichlet(OrderTest):
    name = "2D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
        q_fun = lambda x: -2 * (torch.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)
        M_bf = mesh.boundary_face_scalar_integral

        V = utils.sdiag(mesh.cell_volumes)
        G = -torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # Since the xc_bc is known, move it to the RHS!
        A = V @ D @ MfI @ G
        rhs = (
            V @ q_ana
            - V @ D @ MfI @ torch.sparse.mm(M_bf, phi_bc.unsqueeze(-1)).squeeze()
        )

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())
        err = torch.linalg.norm(phi_test - phi_ana) / torch.sqrt(
            torch.tensor(mesh.n_cells, dtype=torch.float32)
        )

        return err.item()

    def test_orderX(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC1D_InhomogeneousNeumann(OrderTest):
    name = "1D - Neumann"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: torch.sin(torch.pi * x)
        j_fun = lambda x: torch.pi * torch.cos(torch.pi * x)
        q_fun = lambda x: -(torch.pi**2) * torch.sin(torch.pi * x)

        mesh = self.M
        xc_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)
        j_ana = j_fun(mesh.faces_x)

        phi_bc = phi(mesh.boundary_faces)
        j_bc = j_fun(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = 0.0
        beta = 1.0
        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals
        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        # Compute the analytical solution for j
        # print the shape of the matrices for debugging
        print(
            f"Shape of MfI: {MfI.shape}, G: {G.shape}, B_bc: {B_bc.shape}, b_bc: {b_bc.shape}"
        )

        j = MfI @ ((-G + B_bc) @ xc_ana + b_bc)

        # Since the xc_bc is known, move it to the RHS!
        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        if self.myTest == "j":
            err = torch.linalg.norm((j - j_ana), ord=float("inf"))
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                j = MfI @ ((-G + B_bc) @ xc + b_bc)
                err = torch.linalg.norm((j - j_ana)) / torch.sqrt(
                    torch.tensor(mesh.n_edges, dtype=torch.float32)
                )
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")

        return err.item() if isinstance(err, torch.Tensor) else err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousNeumann_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousNeumann_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class TestCC2D_InhomogeneousNeumann(OrderTest):
    name = "2D - Neumann"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        phi = lambda x: torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
        j_funX = (
            lambda x: torch.pi
            * torch.cos(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
        )
        j_funY = (
            lambda x: torch.pi
            * torch.sin(torch.pi * x[:, 0])
            * torch.cos(torch.pi * x[:, 1])
        )
        q_fun = lambda x: -2 * (torch.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        j_bc = torch.stack([jx_bc, jy_bc], dim=-1)

        j_bc_dot_n = torch.sum(j_bc * mesh.boundary_face_outward_normals, dim=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = 0.0
        beta = 1.0
        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        try:
            phi_test = torch.linalg.lstsq(
                A.to_dense(), rhs.to_dense(), rcond=1e-6
            ).solution
            phi_test -= phi_test.mean()
            phi_ana -= phi_ana.mean()

            err = torch.linalg.norm((phi_test - phi_ana), ord=float("inf"))
        except Exception as e:
            print(f"Solve failed with error: {type(e).__name__}: {str(e)}")
            print("Setting error to infinity")
            err = float("inf")

        return err.item() if isinstance(err, torch.Tensor) else err

    def test_orderX(self):
        self.name = "2D - InhomogeneousNeumann_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC1D_InhomogeneousMixed(OrderTest):
    name = "1D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(0.5 * torch.pi * x)
        j_fun = lambda x: -0.5 * torch.pi * torch.sin(0.5 * torch.pi * x)
        q_fun = lambda x: -0.25 * (torch.pi**2) * torch.cos(0.5 * torch.pi * x)

        mesh = self.M
        xc_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)
        j_ana = j_fun(mesh.faces_x)

        phi_bc = phi(mesh.boundary_faces)
        j_bc = j_fun(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = torch.tensor([1.0, 0.0])
        beta = torch.tensor([0.0, 1.0])
        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals
        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        if self.myTest == "xc":
            xc = torch.linalg.solve(A.to_dense(), rhs.to_dense())
            err = torch.linalg.norm(xc - xc_ana) / torch.sqrt(
                torch.tensor(mesh.n_cells, dtype=torch.float32)
            )
        elif self.myTest == "xcJ":
            xc = torch.linalg.solve(A.to_dense(), rhs.to_dense())
            j = MfI @ ((-G + B_bc) @ xc + b_bc)
            err = torch.linalg.norm(j - j_ana, ord=float("inf"))

        return err.item()

    def test_orderX(self):
        self.name = "1D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousMixed_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class TestCC2D_InhomogeneousMixed(OrderTest):
    name = "2D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16]

    def getError(self):
        # Test function
        phi = lambda x: torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
        j_funX = (
            lambda x: torch.pi
            * torch.cos(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
        )
        j_funY = (
            lambda x: torch.pi
            * torch.sin(torch.pi * x[:, 0])
            * torch.cos(torch.pi * x[:, 1])
        )
        q_fun = lambda x: -2 * (torch.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        j_bc = torch.stack([jx_bc, jy_bc], dim=-1)

        j_bc_dot_n = torch.sum(j_bc * mesh.boundary_face_outward_normals, dim=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # construct matrix with robin operator
        # get indices of x0 boundary and y0 boundary
        n_boundary_faces = len(j_bc_dot_n)
        dirichlet_locs = torch.any(mesh.boundary_faces == 0.0, dim=1)

        alpha = torch.zeros(n_boundary_faces)
        alpha[dirichlet_locs] = 1.0

        beta = torch.zeros(n_boundary_faces)
        beta[~dirichlet_locs] = 1.0

        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())

        err = torch.linalg.norm((phi_test - phi_ana)) / torch.sqrt(
            torch.tensor(mesh.n_cells, dtype=torch.float32)
        )

        return err.item()

    def test_orderX(self):
        self.name = "2D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC3D_InhomogeneousMixed(OrderTest):
    name = "3D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 3
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16]

    def getError(self):
        # Test function
        phi = (
            lambda x: torch.sin(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
            * torch.sin(torch.pi * x[:, 2])
        )

        j_funX = (
            lambda x: torch.pi
            * torch.cos(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
            * torch.sin(torch.pi * x[:, 2])
        )
        j_funY = (
            lambda x: torch.pi
            * torch.sin(torch.pi * x[:, 0])
            * torch.cos(torch.pi * x[:, 1])
            * torch.sin(torch.pi * x[:, 2])
        )
        j_funZ = (
            lambda x: torch.pi
            * torch.sin(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
            * torch.cos(torch.pi * x[:, 2])
        )

        q_fun = lambda x: -3 * (torch.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        jz_bc = j_funZ(mesh.boundary_faces)
        j_bc = torch.stack([jx_bc, jy_bc, jz_bc], dim=-1)

        j_bc_dot_n = torch.sum(j_bc * mesh.boundary_face_outward_normals, dim=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = torch.sparse.mm(mesh.face_divergence.t(), V)
        D = mesh.face_divergence

        # construct matrix with robin operator
        # get indices of x0 boundary, y0, and z0 boundary
        n_boundary_faces = len(j_bc_dot_n)
        dirichlet_locs = torch.any(mesh.boundary_faces == 0.0, dim=1)

        alpha = torch.zeros(n_boundary_faces)
        alpha[dirichlet_locs] = 1.0

        beta = torch.zeros(n_boundary_faces)
        beta[~dirichlet_locs] = 1.0

        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())

        err = torch.linalg.norm(phi_test - phi_ana) / torch.sqrt(
            torch.tensor(mesh.n_cells, dtype=torch.float32)
        )

        return err.item()

    def test_orderX(self):
        self.name = "3D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestN1D_boundaries(OrderTest):
    name = "1D - Boundaries"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: torch.sin(torch.pi * x)
        j_fun = lambda x: torch.pi * torch.cos(torch.pi * x)
        q_fun = lambda x: -1 * (torch.pi**2) * phi(x)

        mesh = self.M
        mesh.origin = torch.tensor([-0.25])

        phi_ana = phi(mesh.nodes)
        q_ana = q_fun(mesh.nodes)

        phi_bc = phi(mesh.boundary_nodes)
        j_bc = j_fun(mesh.boundary_nodes)

        # construct matrix with robin operator
        beta = 1.0
        if self.boundary_type == "Robin":
            alpha = 1.0
        elif self.boundary_type == "Mixed":
            alpha = torch.tensor([1.0, 0.0])
        else:
            alpha = 0.0

        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals

        Me = mesh.get_edge_inner_product()
        Mn = utils.sdiag(
            torch.sparse.mm(
                mesh.average_node_to_cell.t(), mesh.cell_volumes.unsqueeze(-1)
            ).squeeze()
        )
        G = mesh.nodal_gradient

        B_bc, b_bc = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)

        A = -torch.sparse.mm(torch.sparse.mm(G.t(), Me), G) + B_bc
        rhs = torch.sparse.mm(Mn, q_ana.unsqueeze(-1)).squeeze() - b_bc

        if self.boundary_type == "Neumann":
            # put a single dirichlet node on the boundary
            # Mimic sp.eye(mesh.n_nodes)[:, 0] and sp.eye(mesh.n_nodes)[:, 1:]
            I = torch.eye(
                mesh.n_nodes, dtype=mesh.nodes.dtype, device=mesh.nodes.device
            )
            P_b = I[:, 0:1]  # First column
            P_f = I[:, 1:]  # Remaining columns

            # Convert to sparse tensors
            P_b = P_b.to_sparse_coo().coalesce()
            P_f = P_f.to_sparse_coo().coalesce()

            # P_f.T @ A @ (P_f @ x + P_b @ ana) = P_f.T @ rhs
            rhs = (
                torch.sparse.mm(P_f.t(), rhs.unsqueeze(-1)).squeeze()
                - torch.sparse.mm(
                    torch.sparse.mm(P_f.t(), torch.sparse.mm(A, P_b)),
                    phi_ana[0:1].unsqueeze(-1),
                ).squeeze()
            )
            A = torch.sparse.mm(torch.sparse.mm(P_f.t(), A), P_f)

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())

        if self.boundary_type == "Neumann":
            phi_test = (
                torch.sparse.mm(P_f, phi_test.unsqueeze(-1)).squeeze()
                + torch.sparse.mm(P_b, phi_ana[0:1].unsqueeze(-1)).squeeze()
            )

        err = torch.linalg.norm(phi_test - phi_ana, ord=float("inf"))

        return err.item()

    def test_orderNeumannX(self):
        self.name = "1D - NeumannBoundary_Inverse"
        self.boundary_type = "Neumann"
        self.orderTest()

    def test_orderRobinX(self):
        self.name = "1D - RobinBoundary_Inverse"
        self.boundary_type = "Robin"
        self.orderTest()

    def test_orderMixed(self):
        self.name = "1D - MixedBoundary_Inverse"
        self.boundary_type = "Mixed"
        self.orderTest()


class TestN2D_boundaries(OrderTest):
    name = "2D - Boundaries"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    tolerance = 0.8
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        phi = lambda x: torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])
        j_funX = (
            lambda x: torch.pi
            * torch.cos(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
        )
        j_funY = (
            lambda x: torch.pi
            * torch.cos(torch.pi * x[:, 1])
            * torch.sin(torch.pi * x[:, 0])
        )
        q_fun = lambda x: -2 * (torch.pi**2) * phi(x)

        mesh = self.M
        mesh.origin = torch.tensor([-0.25, -0.25])

        phi_ana = phi(mesh.nodes)
        q_ana = q_fun(mesh.nodes)

        if self.boundary_type == "Neumann":
            # Neumann with J defined at boundary nodes
            jx_bc = j_funX(mesh.boundary_nodes)
            jy_bc = j_funY(mesh.boundary_nodes)
            j_bc = torch.stack([jx_bc, jy_bc], dim=-1)

            M_bn = mesh.boundary_node_vector_integral

            B_bc = torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]),
                torch.tensor([0.0]),
                (mesh.n_nodes, mesh.n_nodes),
            ).coalesce()
            # Use Fortran-order reshape to match NumPy order="F"
            j_bc_flat = torch.stack(
                [j_bc[:, i] for i in range(j_bc.shape[1])]
            ).flatten()
            b_bc = torch.sparse.mm(M_bn, j_bc_flat.unsqueeze(-1)).squeeze()
        else:
            phi_bc = phi(mesh.boundary_faces)
            jx_bc = j_funX(mesh.boundary_faces)
            jy_bc = j_funY(mesh.boundary_faces)
            j_bc = torch.stack([jx_bc, jy_bc], dim=-1)

            j_bc_dot_n = torch.sum(j_bc * mesh.boundary_face_outward_normals, dim=-1)

            # construct matrix with robin operator
            if self.boundary_type == "Robin":
                alpha = 1.0
            else:
                # get indices of x0 boundary and y0 boundary
                n_boundary_faces = len(j_bc_dot_n)
                robin_locs = torch.any(mesh.boundary_faces == -0.25, dim=1)

                alpha = torch.zeros(n_boundary_faces)
                alpha[robin_locs] = 1.0

            beta = 1.0
            gamma = alpha * phi_bc + beta * j_bc_dot_n

            B_bc, b_bc = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)

        Me = mesh.get_edge_inner_product()
        Mn = utils.sdiag(
            torch.sparse.mm(
                mesh.average_node_to_cell.t(), mesh.cell_volumes.unsqueeze(-1)
            ).squeeze()
        )
        G = mesh.nodal_gradient

        A = -torch.sparse.mm(torch.sparse.mm(G.t(), Me), G) + B_bc
        rhs = torch.sparse.mm(Mn, q_ana.unsqueeze(-1)).squeeze() - b_bc

        if self.boundary_type == "Neumann":
            # put a single dirichlet node on the boundary
            # Mimic sp.eye(mesh.n_nodes)[:, 0] and sp.eye(mesh.n_nodes)[:, 1:]
            I = torch.eye(
                mesh.n_nodes, dtype=mesh.nodes.dtype, device=mesh.nodes.device
            )
            P_b = I[:, 0:1]  # First column
            P_f = I[:, 1:]  # Remaining columns

            # Convert to sparse tensors
            P_b = P_b.to_sparse_coo().coalesce()
            P_f = P_f.to_sparse_coo().coalesce()

            # P_f.T @ A @ (P_f @ x + P_b @ ana) = P_f.T @ rhs
            rhs = (
                torch.sparse.mm(P_f.t(), rhs.unsqueeze(-1)).squeeze()
                - torch.sparse.mm(
                    torch.sparse.mm(P_f.t(), torch.sparse.mm(A, P_b)),
                    phi_ana[0:1].unsqueeze(-1),
                ).squeeze()
            )
            A = torch.sparse.mm(torch.sparse.mm(P_f.t(), A), P_f)

        phi_test = torch.linalg.solve(A.to_dense(), rhs.to_dense())

        if self.boundary_type == "Neumann":
            phi_test = (
                torch.sparse.mm(P_f, phi_test.unsqueeze(-1)).squeeze()
                + torch.sparse.mm(P_b, phi_ana[0:1].unsqueeze(-1)).squeeze()
            )

        err = torch.linalg.norm(phi_test - phi_ana, ord=float("inf"))
        return err.item()

    def test_orderNeumannX(self):
        self.name = "2D - NeumannBoundary_Inverse"
        self.boundary_type = "Neumann"
        self.orderTest()

    def test_orderRobinX(self):
        self.name = "2D - RobinBoundary_Inverse"
        self.boundary_type = "Robin"
        self.orderTest()

    def test_orderMixed(self):
        self.name = "2D - MixedBoundary_Inverse"
        self.boundary_type = "Mixed"
