# flake8: noqa: E731
import torch
import unittest
from simpegtorch.discretize import utils
from simpegtorch.discretize.tests import OrderTest
from torch.linalg import solve


MESHTYPES = ["uniformTensorMesh"]


class Test1D_InhomogeneousDirichlet(OrderTest):
    name = "1D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(torch.pi * x)
        j_fun = lambda x: -torch.pi * torch.sin(torch.pi * x)
        q_fun = lambda x: -(torch.pi**2) * torch.cos(torch.pi * x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        # TODO: Check where our boundary conditions are CCx or Nx
        # vec = self.M.nodes_x
        vec = self.M.cell_centers_x

        phi_bc = phi(vec[[0, -1]])
        j_bc = j_fun(vec[[0, -1]])

        P, Pin, Pout = self.M.get_BC_projections([["dirichlet", "dirichlet"]])

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        V = utils.sdiag(self.M.cell_volumes)
        G = -Pin.t() @ Pin @ self.M.face_divergence.t() @ V
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ phi_bc)
        q = D @ Pin.t() @ Pin @ j + D @ Pout.t() @ j_bc

        # Rearrange if we know q to solve for x
        A = V @ D @ Pin.t() @ Pin @ McI @ G
        rhs = (
            V @ q_ana
            - V @ D @ Pin.t() @ Pin @ McI @ P @ phi_bc
            - V @ D @ Pout.t() @ j_bc
        )
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc

        if self.myTest == "j":
            err = torch.linalg.norm((j - j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            # TODO: fix the null space
            xc = solve(A.to_dense(), rhs.to_dense())
            print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            xc = solve(A.to_dense(), rhs.to_dense())
            print(torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            j = McI @ (G @ xc + P @ phi_bc)
            err = torch.linalg.norm((j - j_ana), ord=float("inf"))

        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousDirichlet_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousDirichlet_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderX(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class Test2D_InhomogeneousDirichlet(OrderTest):
    name = "2D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
        j_funX = (
            lambda x: -torch.pi
            * torch.sin(torch.pi * x[:, 0])
            * torch.cos(torch.pi * x[:, 1])
        )
        j_funY = (
            lambda x: -torch.pi
            * torch.cos(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
        )
        q_fun = lambda x: -2 * (torch.pi**2) * phi(x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = torch.cat([jX_ana, jY_ana])

        # TODO: Check where our boundary conditions are CCx or Nx
        # fxm,fxp,fym,fyp = self.M.face_boundary_indices
        # gBFx = self.M.gridFx[(fxm|fxp),:]
        # gBFy = self.M.gridFy[(fym|fyp),:]
        fxm, fxp, fym, fyp = self.M.cell_boundary_indices
        gBFx = self.M.gridCC[(fxm | fxp), :]
        gBFy = self.M.gridCC[(fym | fyp), :]

        bc = phi(torch.cat([gBFx, gBFy]))

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.get_BC_projections("dirichlet")

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        G = -self.M.face_divergence.t() @ utils.sdiag(self.M.cell_volumes)
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ bc)
        q = D @ j

        # self.M.plot_image(j, 'FxFy', show_it=True)

        # Rearrange if we know q to solve for x
        A = D @ McI @ G
        rhs = q_ana - D @ McI @ P @ bc

        if self.myTest == "j":
            err = torch.linalg.norm((j - j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            xc = solve(A.to_dense(), rhs.to_dense())
            err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
        elif self.myTest == "xcJ":
            xc = solve(A.to_dense(), rhs.to_dense())
            j = McI @ (G @ xc + P @ bc)
            err = torch.linalg.norm((j - j_ana), ord=float("inf"))

        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousDirichlet_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousDirichlet_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderX(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class Test1D_InhomogeneousNeumann(OrderTest):
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

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        # TODO: Check where our boundary conditions are CCx or Nx
        vecN = self.M.nodes_x
        vecC = self.M.cell_centers_x

        phi_bc = phi(vecC[[0, -1]])
        j_bc = j_fun(vecN[[0, -1]])

        P, Pin, Pout = self.M.get_BC_projections([["neumann", "neumann"]])

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        V = utils.sdiag(self.M.cell_volumes)
        G = -Pin.t() @ Pin @ self.M.face_divergence.t() @ V
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ phi_bc)
        q = V @ D @ Pin.t() @ Pin @ j + V @ D @ Pout.t() @ j_bc

        # Rearrange if we know q to solve for x
        A = V @ D @ Pin.t() @ Pin @ McI @ G
        rhs = (
            V @ q_ana
            - V @ D @ Pin.t() @ Pin @ McI @ P @ phi_bc
            - V @ D @ Pout.t() @ j_bc
        )
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc

        if self.myTest == "j":
            err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - V @ q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            # TODO: fix the null space
            # PyTorch doesn't have minres, using lstsq as alternative
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                j = McI @ (G @ xc + P @ phi_bc)
                err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousNeumann_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousNeumann_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousNeumann_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class Test2D_InhomogeneousNeumann(OrderTest):
    name = "2D - Neumann"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]
    # meshSizes = [4]

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

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = torch.cat([jX_ana, jY_ana])

        # TODO: Check where our boundary conditions are CCx or Nx

        cxm, cxp, cym, cyp = self.M.cell_boundary_indices
        fxm, fxp, fym, fyp = self.M.face_boundary_indices

        gBFx = self.M.gridFx[(fxm | fxp), :]
        gBFy = self.M.gridFy[(fym | fyp), :]

        phi_bc = phi(torch.cat([gBFx, gBFy]))
        j_bc = torch.cat([j_funX(gBFx), j_funY(gBFy)])

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.get_BC_projections("neumann")

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        V = utils.sdiag(self.M.cell_volumes)
        G = -Pin.t() @ Pin @ self.M.face_divergence.t() @ V
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ phi_bc)
        q = V @ D @ Pin.t() @ Pin @ j + V @ D @ Pout.t() @ j_bc

        # Rearrange if we know q to solve for x
        A = V @ D @ Pin.t() @ Pin @ McI @ G
        rhs = (
            V @ q_ana
            - V @ D @ Pin.t() @ Pin @ McI @ P @ phi_bc
            - V @ D @ Pout.t() @ j_bc
        )

        if self.myTest == "j":
            err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - V @ q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                j = McI @ (G @ xc + P @ phi_bc)
                err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousNeumann_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousNeumann_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousNeumann_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class Test1D_InhomogeneousMixed(OrderTest):
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

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        # TODO: Check where our boundary conditions are CCx or Nx
        vecN = self.M.nodes_x
        vecC = self.M.cell_centers_x

        phi_bc = phi(vecC[[0, -1]])
        j_bc = j_fun(vecN[[0, -1]])

        P, Pin, Pout = self.M.get_BC_projections([["dirichlet", "neumann"]])

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        V = utils.sdiag(self.M.cell_volumes)
        G = -Pin.t() @ Pin @ self.M.face_divergence.t() @ V
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ phi_bc)
        q = V @ D @ Pin.t() @ Pin @ j + V @ D @ Pout.t() @ j_bc

        # Rearrange if we know q to solve for x
        A = V @ D @ Pin.t() @ Pin @ McI @ G
        rhs = (
            V @ q_ana
            - V @ D @ Pin.t() @ Pin @ McI @ P @ phi_bc
            - V @ D @ Pout.t() @ j_bc
        )
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc

        if self.myTest == "j":
            err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - V @ q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                j = McI @ (G @ xc + P @ phi_bc)
                err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousMixed_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousMixed_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousMixed_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class Test2D_InhomogeneousMixed(OrderTest):
    name = "2D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = lambda x: torch.cos(0.5 * torch.pi * x[:, 0]) * torch.cos(
            0.5 * torch.pi * x[:, 1]
        )
        j_funX = (
            lambda x: -0.5
            * torch.pi
            * torch.sin(0.5 * torch.pi * x[:, 0])
            * torch.cos(0.5 * torch.pi * x[:, 1])
        )
        j_funY = (
            lambda x: -0.5
            * torch.pi
            * torch.cos(0.5 * torch.pi * x[:, 0])
            * torch.sin(0.5 * torch.pi * x[:, 1])
        )
        q_fun = lambda x: -2 * ((0.5 * torch.pi) ** 2) * phi(x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = torch.cat([jX_ana, jY_ana])

        # TODO: Check where our boundary conditions are CCx or Nx

        cxm, cxp, cym, cyp = self.M.cell_boundary_indices
        fxm, fxp, fym, fyp = self.M.face_boundary_indices

        gBFx = self.M.gridFx[(fxm | fxp), :]
        gBFy = self.M.gridFy[(fym | fyp), :]

        gBCx = self.M.gridCC[(cxm | cxp), :]
        gBCy = self.M.gridCC[(cym | cyp), :]

        phi_bc = phi(torch.cat([gBCx, gBCy]))
        j_bc = torch.cat([j_funX(gBFx), j_funY(gBFy)])

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.get_BC_projections(
            [["dirichlet", "neumann"], ["dirichlet", "neumann"]]
        )

        Mc = self.M.get_face_inner_product()
        McI = utils.sdinv(Mc)
        V = utils.sdiag(self.M.cell_volumes)
        G = -Pin.t() @ Pin @ self.M.face_divergence.t() @ V
        D = self.M.face_divergence
        j = McI @ (G @ xc_ana + P @ phi_bc)
        q = V @ D @ Pin.t() @ Pin @ j + V @ D @ Pout.t() @ j_bc

        # Rearrange if we know q to solve for x
        A = V @ D @ Pin.t() @ Pin @ McI @ G
        rhs = (
            V @ q_ana
            - V @ D @ Pin.t() @ Pin @ McI @ P @ phi_bc
            - V @ D @ Pout.t() @ j_bc
        )

        if self.myTest == "j":
            err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
        elif self.myTest == "q":
            err = torch.linalg.norm((q - V @ q_ana), ord=float("inf"))
        elif self.myTest == "xc":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                err = torch.linalg.norm((xc - xc_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            try:
                xc = torch.linalg.lstsq(
                    A.to_dense(), rhs.to_dense(), rcond=1e-6
                ).solution
                j = McI @ (G @ xc + P @ phi_bc)
                err = torch.linalg.norm((Pin @ j - Pin @ j_ana), ord=float("inf"))
                print("ACCURACY", torch.linalg.norm(utils.mkvc(A @ xc) - rhs))
            except Exception as e:
                print(f"XCJ Solve failed with error: {type(e).__name__}: {str(e)}")
                print("Setting error to infinity")
                err = float("inf")
        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousMixed_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousMixed_Forward q"
        self.myTest = "q"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousMixed_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


if __name__ == "__main__":
    unittest.main()
