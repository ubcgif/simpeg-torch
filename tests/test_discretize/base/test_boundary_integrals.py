import torch
from simpegtorch.discretize.tests import OrderTest
from simpegtorch.discretize.utils import mkvc


def u(*args):
    if len(args) == 1:
        x = args[0]
        return x**3
    if len(args) == 2:
        x, y = args
        return x**3 + y**2
    x, y, z = args
    return x**3 + y**2 + z**4


def v(*args):
    if len(args) == 1:
        x = args[0]
        return 2 * x**2
    if len(args) == 2:
        x, y = args
        return torch.column_stack([2 * x**2, 3 * y**3])
    x, y, z = args
    return torch.column_stack([2 * x**2, 3 * y**3, -4 * z**2])


def w(*args):
    if len(args) == 2:
        x, y = args
        return torch.column_stack([(y - 2) ** 2, (x + 2) ** 2])
    x, y, z = args
    return torch.column_stack(
        [(y - 2) ** 2 + z**2, (x + 2) ** 2 - (z - 4) ** 2, y**2 - x**2]
    )


# mesh will be on [0, 1] square

# 1D
# int_V grad_u dot v dV = 6/5
# int_V u dot div v dV = 4/5

# 2D
# square vals:
# int_V grad_u dot v dV = 12/5
# int_V u div_v dV = 241/60
# int_v curl_w dot v dV = -173/30

# 3D square vals:

# int_V grad_u dot v dV = -4/15
# int_V u div_v dV = 27/20
# int_v curl_w dot v dV = 17/6


class Test1DBoundaryIntegral(OrderTest):
    name = "1D Boundary Integrals"
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            u_cc = u(mesh.cell_centers)
            v_f = v(mesh.nodes)
            u_bf = u(mesh.boundary_faces)

            D = mesh.face_divergence
            M_c = torch.diag(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            # reverse the order of tensors
            v_f = v_f.permute(*torch.arange(v_f.ndim - 1, -1, -1))
            D = D.permute(*torch.arange(D.ndim - 1, -1, -1))

            discrete_val = -(v_f @ D) @ M_c @ u_cc + v_f @ (M_bf @ u_bf)
            true_val = 6 / 5

        if self.myTest == "edge_div":
            u_n = u(mesh.nodes)
            v_e = v(mesh.edges)
            v_bn = mkvc(v(mesh.boundary_nodes))

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            # reverse the order of tensors
            u_n = u_n.permute(*torch.arange(u_n.ndim - 1, -1, -1))
            G = G.permute(*torch.arange(G.ndim - 1, -1, -1))

            discrete_val = -(u_n @ G) @ M_e @ v_e + u_n @ (M_bn @ v_bn)
            true_val = 4 / 5
        return torch.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "1D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "1D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()


class Test2DBoundaryIntegral(OrderTest):
    name = "2D Boundary Integrals"
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = torch.diag(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            # reverse the order of tensors
            v_f = v_f.permute(*torch.arange(v_f.ndim - 1, -1, -1))
            D = D.permute(*torch.arange(D.ndim - 1, -1, -1))

            discrete_val = -(v_f @ D) @ M_c @ u_cc + v_f @ (M_bf @ u_bf)
            true_val = 12 / 5

        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = mkvc(v(*mesh.boundary_nodes.T))

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            # reverse the order of tensors
            u_n = u_n.permute(*torch.arange(u_n.ndim - 1, -1, -1))
            G = G.permute(*torch.arange(G.ndim - 1, -1, -1))

            discrete_val = -(u_n @ G) @ M_e @ v_e + u_n @ (M_bn @ v_bn)
            true_val = 241 / 60

        elif self.myTest == "face_curl":
            w_e = mesh.project_edge_vector(w(*mesh.edges.T))
            u_c = u(*mesh.cell_centers.T)
            u_be = u(*mesh.boundary_edges.T)

            M_c = torch.diag(mesh.cell_volumes)
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            Curl = Curl.permute(*torch.arange(Curl.ndim - 1, -1, -1))
            w_e = w_e.permute(*torch.arange(w_e.ndim - 1, -1, -1))

            discrete_val = (w_e @ Curl) @ M_c @ u_c - w_e @ (M_be @ u_be)
            true_val = -173 / 30

        return torch.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "2D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "2D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "2D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()


class Test3DBoundaryIntegral(OrderTest):
    name = "3D Boundary Integrals"
    meshDimension = 3
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = torch.diag(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            # reverse the order of tensors
            v_f = v_f.permute(*torch.arange(v_f.ndim - 1, -1, -1))
            D = D.permute(*torch.arange(D.ndim - 1, -1, -1))

            discrete_val = -(v_f @ D) @ M_c @ u_cc + v_f @ (M_bf @ u_bf)
            true_val = -4 / 15

        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = mkvc(v(*mesh.boundary_nodes.T))

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            # reverse the order of tensors
            u_n = u_n.permute(*torch.arange(u_n.ndim - 1, -1, -1))
            G = G.permute(*torch.arange(G.ndim - 1, -1, -1))

            discrete_val = -(u_n @ G) @ M_e @ v_e + u_n @ (M_bn @ v_bn)
            true_val = 27 / 20

        elif self.myTest == "face_curl":
            w_f = mesh.project_face_vector(w(*mesh.faces.T))
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            w_be = mkvc(w(*mesh.boundary_edges.T))

            M_f = mesh.get_face_inner_product()
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            # reverse the order of tensors
            Curl = Curl.permute(*torch.arange(Curl.ndim - 1, -1, -1))
            v_e = v_e.permute(*torch.arange(v_e.ndim - 1, -1, -1))

            discrete_val = (v_e @ Curl) @ M_f @ w_f - v_e @ (M_be @ w_be)
            true_val = -79 / 6

        return torch.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "3D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest(random_seed=51235)

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "3D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest(random_seed=51123)

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "3D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest(random_seed=5522)
