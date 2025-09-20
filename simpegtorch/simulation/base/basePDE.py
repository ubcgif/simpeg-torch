from abc import ABC, abstractmethod
import torch
from simpegtorch.discretize.base import BaseMesh
from mappings import BaseMapping


class BasePDE(ABC):
    """
    Minimal PyTorch-native PDE interface for forward simulations.

    This class defines the essential interface for any PDE formulation,
    focusing on the three core operations needed for simulation:
    1. System matrix construction
    2. RHS vector construction
    3. Field-to-data projection
    4. A vec product (optional, for experimental indirect solver)

    All gradient computation is handled automatically by PyTorch autograd.
    """

    def __init__(self, mesh: BaseMesh, mapping: BaseMapping):
        self.mesh = mesh
        self.mapping = mapping

    @abstractmethod
    def get_system_matrices(self) -> torch.Tensor:
        """
        Construct system matrices for the PDE.

        Parameters
        ----------
        model_params : torch.Tensor
            Physical model parameters (conductivity, permeability, etc.)

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Single system matrix for problems like DC resistivity,
            or list of matrices for frequency-domain problems.
            Shape depends on problem type:
            - DC: [1, n_cells, n_cells]
            - FDEM: [n_frequencies, n_cells, n_cells]
        """
        pass

    @abstractmethod
    def get_rhs_tensors(self) -> torch.Tensor:
        """
        Construct right-hand side vectors for all sources.

        Parameters

        Returns
        -------
        torch.Tensor
            Batched RHS tensors. Shape depends on problem type:
            - DC: [n_sources, n_cells]
            - FDEM: [n_sources, n_cells]
        """
        pass

    @abstractmethod
    def fields_to_data(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Project solution fields to predicted data.

        Parameters
        ----------
        fields : torch.Tensor
            Solution fields from PDE solve

        Returns
        -------
        torch.Tensor
            Predicted data vector [n_data]
        """
        pass
