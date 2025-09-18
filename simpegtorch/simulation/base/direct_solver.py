import torch
from torch import nn
from .basePDE import BasePDE
from simpegtorch.torchmatsolver import batched_mumps_solve, batched_sparse_solve


class DirectSolver(nn.Module):
    """
    PyTorch-native simulation class for a direct solver on any PDE formualtion.

    This class provides a minimal, clean interface that works with any PDE. All gradient computation is handled
    automatically by PyTorch autograd.

    The entire simulation is a single forward pass which gives predicted data
    """

    def __init__(self, pde: BasePDE):
        """
        Initialize universal simulation.

        Parameters
        ----------
        pde : BasePDE
            PDE formulation defining the physics
        survey : Survey
            Survey object with sources and receivers
        solver_method : str
            Solver method ("Direct" or "Iterative")
        """
        super().__init__()
        self.pde = pde

    def forward(self) -> torch.Tensor:
        """
        Forward simulation: model parameters → predicted data.

        Solves the PDE using a direct solver through matrixn inversion.
        Total process:
        1. Gets a RHS tensor of shape 1xkxn for n parameters problem and k different sources
        2. Gets a A matrix tensor of fx1xnxn for n parameterrs and f different problems (e.g. different frequencies)
        3. Runs a direct solver over this to solve for a fxkxn field solution tensor
        4. for j different receivers projects the fields into fxj different measurements
        """
        # 1. Apply mapping if provided

        # 2. Get system matrices from PDE
        system_matrices = self.pde.get_system_matrices()

        # 3. Get RHS vectors from PDE
        rhs_tensors = self.pde.get_rhs_tensors()

        try:
            fields = batched_mumps_solve(system_matrices, rhs_tensors)
        except ImportError:
            print("MUMPS not installed, falling back")
            fields = batched_sparse_solve(system_matrices, rhs_tensors)

        # 5. Project to data and return
        return self.pde.fields_to_data(fields)
