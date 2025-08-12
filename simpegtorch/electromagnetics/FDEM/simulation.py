import torch
from simpegtorch.torchmatsolver import batched_mumps_solve, batched_sparse_solve

from simpegtorch.discretize import TensorMesh

# from simpegtorch.discretize.utils import sdiag


class BaseFDEMSimulation:
    """
    Base class for FDEM simulations.
    """

    def __init__(self, mesh: TensorMesh, survey=None, **kwargs):
        self.mesh = mesh
        self.survey = survey

    def getRHS():
        """
        Get the right-hand side (RHS) vector for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getRHS must be implemented in derived classes.")

    def getA():
        """
        Get the system matrix (A) for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getA must be implemented in derived classes.")

    def fields(self, m):
        """
        Compute the fields for the given model parameters.

        Parameters
        ----------
        m : torch.Tensor
            Model parameters.

        Returns
        -------
        torch.Tensor
            Computed fields.
        """
        A = self.getA(m)
        rhs = self.getRHS(m)

        # Solve the system of equations
        if isinstance(A, torch.sparse.FloatTensor):
            return batched_sparse_solve(A, rhs)
        else:
            return batched_mumps_solve(A, rhs)

    def dpred(self, m):
        """
        Compute the predicted data for the given model parameters.

        Parameters
        ----------
        m : torch.Tensor
            Model parameters.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        fields = self.fields(m)
        return self.survey.get_data(fields) if self.survey else fields
