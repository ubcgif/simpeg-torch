import torch
import torch.nn as nn
from .direct_solver import DirectSolver
from .basePDE import BasePDE


class SimulationWrapper(nn.Module):
    """
    Wrapper class to adapt DirectSolver for use with inversion framework.

    The inversion framework expects a simulation with a `dpred(model)` method that
    computes predicted data given model parameters. This wrapper bridges the gap
    between the BasePDE/DirectSolver architecture and the inversion framework.

    Parameters
    ----------
    pde : BasePDE
        PDE formulation defining the physics
    solver : DirectSolver, optional
        Solver instance. If None, a new DirectSolver will be created.

    Examples
    --------
    >>> # Create PDE and solver
    >>> pde = DC3DCellCentered(mesh, survey, mapping)
    >>> solver = DirectSolver(pde)
    >>>
    >>> # Wrap for inversion
    >>> simulation = SimulationWrapper(pde, solver)
    >>>
    >>> # Use with data misfit
    >>> dmisfit = L2DataMisfit(simulation, observed_data)
    """

    def __init__(self, pde: BasePDE, solver: DirectSolver = None):
        super().__init__()
        self.pde = pde
        self.solver = solver if solver is not None else DirectSolver(pde)

    def dpred(self, model: torch.Tensor = None) -> torch.Tensor:
        """
        Compute predicted data for the given model.

        This method is compatible with the inversion framework's expectations.
        If model is provided, it updates the PDE's mapping parameters before
        running the forward simulation.

        Parameters
        ----------
        model : torch.Tensor, optional
            Model parameters to use for prediction. If None, uses current
            parameters from the PDE's mapping.

        Returns
        -------
        torch.Tensor
            Predicted data vector

        Notes
        -----
        The model parameter should match the size of the PDE mapping's
        trainable_parameters. If the mapping has active cells, the model
        should only contain values for the active cells.
        """
        # Update model parameters if provided
        if model is not None:
            # Update the mapping's trainable parameters
            with torch.no_grad():
                self.pde.mapping.trainable_parameters.copy_(model)

        # Run forward simulation
        return self.solver.forward()

    def forward(self) -> torch.Tensor:
        """
        Forward simulation using current model parameters.

        Returns
        -------
        torch.Tensor
            Predicted data vector
        """
        return self.solver.forward()

    @property
    def nD(self):
        """Number of data points."""
        if hasattr(self.pde, "survey"):
            return self.pde.survey.nD
        else:
            raise AttributeError("PDE does not have a survey attribute")

    @property
    def model(self):
        """
        Current model parameters (trainable parameters from mapping).

        Returns
        -------
        torch.Tensor
            Current model parameters
        """
        return self.pde.mapping.trainable_parameters

    @model.setter
    def model(self, value: torch.Tensor):
        """
        Set model parameters.

        Parameters
        ----------
        value : torch.Tensor
            New model parameters
        """
        with torch.no_grad():
            self.pde.mapping.trainable_parameters.copy_(value)


class InversionSimulation(SimulationWrapper):
    """
    Extended simulation wrapper specifically designed for inversions.

    This class provides additional functionality useful for inversions,
    including sensitivity computation and field caching.

    Parameters
    ----------
    pde : BasePDE
        PDE formulation defining the physics
    solver : DirectSolver, optional
        Solver instance. If None, a new DirectSolver will be created.
    cache_fields : bool, optional
        If True, caches fields from forward simulations for reuse (default: False)
    """

    def __init__(
        self, pde: BasePDE, solver: DirectSolver = None, cache_fields: bool = False
    ):
        super().__init__(pde, solver)
        self.cache_fields = cache_fields
        self._cached_fields = None
        self._cached_model = None

    def dpred(self, model: torch.Tensor = None, f: torch.Tensor = None) -> torch.Tensor:
        """
        Compute predicted data with optional field caching.

        Parameters
        ----------
        model : torch.Tensor, optional
            Model parameters to use for prediction
        f : torch.Tensor, optional
            Pre-computed fields. If provided, skips forward solve and projects
            these fields to data.

        Returns
        -------
        torch.Tensor
            Predicted data vector
        """
        # If fields are provided, use them directly
        if f is not None:
            return self.pde.fields_to_data(f)

        # Check if we can use cached fields
        if self.cache_fields and self._cached_fields is not None:
            if model is None or torch.equal(model, self._cached_model):
                return self.pde.fields_to_data(self._cached_fields)

        # Update model and run forward simulation
        if model is not None:
            with torch.no_grad():
                self.pde.mapping.trainable_parameters.copy_(model)

        # Solve forward problem
        # Get system matrices and RHS
        system_matrices = self.pde.get_system_matrices()
        rhs_tensors = self.pde.get_rhs_tensors()

        # Import solver (avoid circular import)
        from simpegtorch.torchmatsolver import batched_mumps_solve, batched_sparse_solve

        try:
            fields = batched_mumps_solve(system_matrices, rhs_tensors)
        except ImportError:
            fields = batched_sparse_solve(system_matrices, rhs_tensors)

        # Cache if requested
        if self.cache_fields:
            self._cached_fields = fields.detach()
            self._cached_model = (
                model.detach().clone() if model is not None else self.model.detach().clone()
            )

        # Project to data
        return self.pde.fields_to_data(fields)

    def clear_cache(self):
        """Clear cached fields and model."""
        self._cached_fields = None
        self._cached_model = None
