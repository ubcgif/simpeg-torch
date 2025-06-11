import torch
from simpegtorch.torchmatsolver import TorchMatSolver
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import sdiag
from scipy import sparse


class DCStaticSimulationCellCentered:
    def __init__(
        self,
        mesh: TensorMesh,
        sources=None,
        recievers=None,
    ):
        self.mesh = mesh
        self.sources = sources
        self.recievers = recievers

    def setBC(self):
        mesh = self.mesh
        V = sdiag(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T
        print("Default BC is always Dirichilet for Now")

    def getA(self, resistivity):
        """
        Returns the system matrix A for the DC resistivity problem.
        If resitivity is provided, it computes the matrix based on the given resistivity values.
        """
        D = self.Div
        G = self.Grad

        MfRhoI = self.mesh.get_face_inner_product(resistivity, invert_matrix=True)

        A = D @ MfRhoI @ G
        return A

    def getRHS(self):
        """
        Returns the right-hand side vector b for the DC resistivity problem.
        If sources are provided, it computes the RHS based on the source values.
        """
        if self.sources is None:
            raise ValueError("Sources must be defined to compute RHS.")

        b = torch.zeros(self.mesh.nC, dtype=torch.float32, device=self.mesh.device)

        for src in self.sources:
            b[src.cell] += src.current

        return b

    def fields(self, resistivity):
        """
        Computes the electric fields for the DC resistivity problem.
        Uses the system matrix A and the right-hand side vector b.
        """
        A = self.getA(resistivity)
        b = self.getRHS()

        # Solve the linear system Ax = b using sparse solver
        def solve_fn(A, b):
            return sparse.linalg.spsolve(A, b)

        x = TorchMatSolver.apply(A, b, solve_fn)

        return x
