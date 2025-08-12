import torch
from simpegtorch.torchmatsolver import batched_mumps_solve, batched_sparse_solve

from simpegtorch.discretize import TensorMesh

from discretize.utils import (
    mkvc,
    Zero
)

# from simpegtorch.discretize.utils import sdiag


class BaseFDEMSimulation:
    """
    Base class for FDEM simulations.
    """

    def __init__(self, mesh: TensorMesh, survey=None, **kwargs):
        self.mesh = mesh
        self.survey = survey

    def getRHS(self, freq):
        """
        Get the right-hand side (RHS) vector for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getRHS must be implemented in derived classes.")

    def getA(self, freq):
        """
        Get the system matrix (A) for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getA must be implemented in derived classes.")
    
    @property
    def getA_tensor(self):
        """
        Returns a (BxNxN) tensor representing the system matrix all of the system matrices for each frequency in the survey.
        Returns
        -------
        torch.Tensor
            A tensor containing the system matrices for each frequency in the survey.
        """
        return torch.stack([self.getA(freq) for freq in self.survey.frequencies], dim=0)

    def getSourceTerm(self, freq):
        """
        Returns the discrete source terms for the frequency provided.

        This moethod computes and returns the discrete magnetics and electric source terms for all soundings at the frequency provided. The exact shape and
        implementation of the source terms when solving for the fields at each frequency
        is formulation dependent.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        s_m : numpy.ndarray
            The magnetic sources terms. (n_faces, n_sources) for EB-formulations. (n_edges, n_sources) for HJ-formulations.
        s_e : numpy.ndarray
            The electric sources terms. (n_edges, n_sources) for EB-formulations. (n_faces, n_sources) for HJ-formulations.
        """
        Srcs = self.survey.get_sources_by_frequency(freq)
        n_fields = sum(src._fields_per_source for src in Srcs)
        if self._formulation == "EB":
            s_m = torch.zeros((self.mesh.nF, n_fields), dtype=complex).clone(memory_format=torch.channels_last)
            s_e = torch.zeros((self.mesh.nE, n_fields), dtype=complex).clone(memory_format=torch.channels_last)
        elif self._formulation == "HJ":
            s_m = torch.zeros((self.mesh.nE, n_fields), dtype=complex).clone(memory_format=torch.channels_last)
            s_e = torch.zeros((self.mesh.nF, n_fields), dtype=complex).clone(memory_format=torch.channels_last)

        i = 0
        for src in Srcs:
            ii = i + src._fields_per_source
            smi, sei = src.eval(self)
            if not isinstance(smi, Zero) and smi.ndim == 1:
                smi = smi[:, None]
            if not isinstance(sei, Zero) and sei.ndim == 1:
                sei = sei[:, None]
            s_m[:, i:ii] = s_m[:, i:ii] + smi
            s_e[:, i:ii] = s_e[:, i:ii] + sei
            i = ii
        return s_m, s_e

        

    def fields(self, freq):
        """
        Compute the fields for the given frequency.

        Parameters
        ----------
        freq : float
            The frequency at which to compute the fields.

        Returns
        -------
        torch.Tensor
            The computed fields.
        """
        A_tensor = self.getA_tensor()
        rhs = self.getRHS(freq)

        # Solve the system of equations
        if isinstance(A_tensor, torch.sparse.FloatTensor):
            return batched_sparse_solve(A_tensor, rhs)
        else:
            return batched_mumps_solve(A_tensor, rhs)

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

