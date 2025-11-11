import torch


class BaseMapping(torch.nn.Module):
    """
    Base class for parameter mappings in simpeg-torch.
    A mapping stores the parameters and returns them in the shape and form used by the base pde model.
    The model parameters are stored as internal trainable parameters of the module, the mapping is created in the forward method.
    """

    def __init__(
        self, ground_parameters: torch.Tensor, active_cells: torch.Tensor = None
    ):
        super().__init__()

        if active_cells is not None:
            if active_cells.size() != ground_parameters.size():
                raise ValueError(
                    "active_cells must be the same size as ground_parameters."
                )
            self.active_cells = active_cells
            # Store static parameters as non-trainable buffer
            self.register_buffer("static_parameters", ground_parameters.clone())
            # Only the active parameters are trainable
            self.trainable_parameters = torch.nn.Parameter(
                ground_parameters[active_cells]
            )
        else:
            self.active_cells = torch.ones_like(ground_parameters, dtype=torch.bool)
            self.register_buffer(
                "static_parameters", torch.zeros_like(ground_parameters)
            )
            self.trainable_parameters = torch.nn.Parameter(ground_parameters)

    def get_full_parameters(self):
        """
        Reconstruct the full parameter tensor by combining trainable and static parts.
        """
        if self.active_cells.all():
            return self.trainable_parameters

        # Create full parameter tensor
        full_params = self.static_parameters.clone()
        full_params[self.active_cells] = self.trainable_parameters
        return full_params

    def forward(self):
        """
        Basic mapping function which transforms input x to model parameters.
        Returns the full parameter tensor combining trainable and static parts.
        Returns
        -------
        torch.Tensor
            Full model parameters tensor
        """
        return self.get_full_parameters()


class InverseMapping(BaseMapping):
    """
    Inverse mapping class for parameter transformations in simpeg-torch.

    This class extends BaseMapping to implement an inverse transformation,
    such as mapping from log-conductivity to conductivity.
    """

    def __init__(
        self, ground_parameters: torch.Tensor, active_cells: torch.Tensor = None
    ):
        super().__init__(ground_parameters, active_cells)

    def forward(self):
        """
        Apply the inverse transformation to the full parameter tensor.

        Returns
        -------
        torch.Tensor
            Transformed model parameters tensor
        """
        full_params = self.get_full_parameters()
        return 1 / full_params


class LogMapping(BaseMapping):
    """
    Log mapping class for parameter transformations in simpeg-torch.

    This class extends BaseMapping to implement a logarithmic transformation,
    such as mapping from log-conductivity to conductivity.
    """

    def __init__(
        self, ground_parameters: torch.Tensor, active_cells: torch.Tensor = None
    ):
        super().__init__(ground_parameters, active_cells)

    def forward(self):
        """
        Apply the exponential transformation to the full parameter tensor.

        Returns
        -------
        torch.Tensor
            Transformed model parameters tensor
        """
        full_params = self.get_full_parameters()
        return torch.exp(full_params)


class LogInverseMapping(BaseMapping):
    """
    Log-inverse mapping class for parameter transformations in simpeg-torch.

    This class extends BaseMapping to implement a log-inverse transformation,
    such as mapping from log-resistivity to conductivity.
    """

    def __init__(
        self, ground_parameters: torch.Tensor, active_cells: torch.Tensor = None
    ):
        super().__init__(ground_parameters, active_cells)

    def forward(self):
        """
        Apply the logarithmic transformation to the full parameter tensor.

        Returns
        -------
        torch.Tensor
            Transformed model parameters tensor
        """
        full_params = self.get_full_parameters()
        return 1 / torch.exp(full_params)
