import torch
from ..basePDE import BasePDE
from simpegtorch.discretize import TensorMesh


class BaseDCPDE(BasePDE):
    """
    DC Resistivity PDE formulation for cell-centered and nodal discretizations.

    Solves the elliptic PDE:
    ∇ · (σ ∇ φ) = -q

    where:
    - σ is conductivity (1/resistivity)
    - φ is electric potential
    - q is current source term
    """

    def __init__(self, mesh: TensorMesh, survey, mapping):
        """
        Initialize DC Resistivity PDE.

        Parameters
        ----------
        mesh : TensorMesh
            Discretization mesh
        survey : Survey
            Survey object with sources and receivers
        mapping : BaseMapping
            Parameter generating function
        """
        super().__init__(mesh, mapping)
        self.survey = survey

    def get_system_matrices(self) -> torch.Tensor:
        raise NotImplementedError("get_system_matrices method not implemented yet.")

    def get_rhs_tensors(self) -> torch.Tensor:
        # Use sources from survey
        raise NotImplementedError("get_rhs_tensors method not implemented yet.")

    def fields_to_data(self, fields: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("fields_to_data method not implemented yet.")
