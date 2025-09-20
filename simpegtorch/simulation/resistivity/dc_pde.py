import torch
from ..basePDE import BasePDE
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import sdiag


class DC3DCellCentered(BasePDE):
    """
    DC Resistivity PDE formulation for Cell centered discretization.

    Solves the elliptic PDE:
    ∇ · (σ ∇ φ) = -q

    where:
    - σ is conductivity (1/resistivity)
    - φ is electric potential
    - q is current source term
    """

    def __init__(
        self,
        mesh: TensorMesh,
        survey,
        mapping,
        bc_type: str = "Dirichlet",
    ):
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
        self.bc_type = bc_type
        self.setBC()

    def setBC(self):
        mesh = self.mesh
        # Standard cell-centered finite volume discretization
        # Volume scaling is needed for proper conservation
        V = sdiag(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T

        # Initialize MfRhoI here, it will be updated in getA
        self.MfRhoI = None

        if self.bc_type == "Dirichlet":
            print("Homogeneous Dirichlet is the natural BC for this CC discretization")
            return
        elif self.bc_type.lower() == "neumann":
            alpha, beta, gamma = 0, 1, 0
            B, _ = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
            self.Grad = self.Grad - B
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
            # TODO: Implement Robin and mixed boundary conditions

    def condition_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """
        Condition the system matrix for numerical stability.
        Used for Neumann boundary conditions.
        """
        # Create a sparse tensor that represents the modification
        # It will have 1 at (0,0) and 0 elsewhere
        mod_indices = torch.tensor([[0], [0]], dtype=torch.long, device=A.device)
        mod_values = torch.tensor([1.0], dtype=A.dtype, device=A.device)
        modification_matrix = torch.sparse_coo_tensor(
            mod_indices, mod_values, A.shape
        ).coalesce()

        # Create a mask for the first row of A
        first_row_mask = A.indices()[0] == 0

        # Get the values of the first row of A
        first_row_values = A.values()[first_row_mask]
        first_row_cols = A.indices()[1, first_row_mask]

        # Create a sparse tensor representing the negative of the first row of A
        neg_first_row_indices = torch.stack(
            [torch.zeros_like(first_row_cols), first_row_cols]
        )
        neg_first_row_values = -first_row_values
        neg_first_row_matrix = torch.sparse_coo_tensor(
            neg_first_row_indices, neg_first_row_values, A.shape
        ).coalesce()

        # Add the modification matrix and the negative of the first row to A
        A = A + neg_first_row_matrix + modification_matrix
        return A

    def get_system_matrices(self) -> torch.Tensor:
        """
        Construct system matrix for the PDE.
        Returns
        -------
        torch.Tensor
            Single system matrix for DC resistivity.
            Shape [1, nC, nC]
        """
        D = self.Div
        G = self.Grad

        # Use face inner product with inverse resistivity (conductivity)
        self.MfRhoI = self.mesh.get_face_inner_product(
            self.mapping.forward(), invert_matrix=True
        )

        A = D @ self.MfRhoI @ G

        if self.bc_type.lower() == "neumann":
            A = self.condition_matrix(A)

        return torch.stack([A], dim=0)  # Shape [1, nC, nC]

    def get_rhs_tensors(self) -> torch.Tensor:
        """
        Construct RHS vectors for all sources.
        Returns
        -------
        torch.Tensor
            Batched RHS tensors. Shape [n_sources, nC]
        """

        return self.survey.get_source_tensor(self.mesh, projected_grid="CC")

    def fields_to_data(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Project solution fields to predicted data.
        Parameters
        ----------
        fields : torch.Tensor
            Solution fields from PDE solve, shape [1, n_sources, nC]
        Returns
        -------
        torch.Tensor
            Predicted data vector [n_data]
        """
        # Remove frequency dimension if present (shape: [n_sources, nC])
        if fields.dim() == 3:
            fields = fields.squeeze(0)

        sources = self.survey.source_list
        data_list = []

        for i, src in enumerate(sources):
            # Get field for this source
            src_field = fields[i]

            # Get receiver projection tensor and project to data
            rx_tensor = src.build_receiver_tensor(self.mesh, "CC")
            rx_data = torch.mv(rx_tensor, src_field)

            data_list.append(rx_data)

        return torch.cat(data_list, dim=0)
