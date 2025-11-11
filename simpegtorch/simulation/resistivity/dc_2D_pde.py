import torch
import numpy as np
from scipy.optimize import minimize
from scipy.special import k0
import warnings

from simpegtorch.simulation.base.basePDE import BasePDE
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import sdiag


class DC2DCellCentered(BasePDE):
    """
    2.5D DC Resistivity PDE formulation for Cell centered discretization.

    Solves the 2.5D elliptic PDE in wavenumber domain:
    ∇ · (σ ∇ φ(ky)) + ky² σ φ(ky) = -q

    Then integrates across wavenumbers to get the 3D response.

    Parameters
    ----------
    mesh : TensorMesh
        2D discretization mesh
    survey : Survey
        Survey object with sources and receivers
    mapping : BaseMapping
        Parameter generating function (typically conductivity or resistivity)
    bc_type : str, default="Dirichlet"
        Boundary condition type ("Dirichlet" or "Neumann")
    nky : int, default=11
        Number of wavenumber quadrature points
    do_trap : bool, default=False
        If True, use trapezoidal quadrature. If False, optimize quadrature points.
    """

    def __init__(
        self,
        mesh: TensorMesh,
        survey,
        mapping,
        bc_type: str = "Dirichlet",
        nky: int = 11,
        do_trap: bool = False,
        verbose: bool = False,
    ):
        super().__init__(mesh, mapping)

        # Validate mesh is 2D
        if mesh.dim != 2:
            raise ValueError(
                f"DC2DCellCentered requires a 2D mesh, got {mesh.dim}D mesh"
            )

        self.survey = survey
        self.bc_type = bc_type
        self.nky = nky
        self.verbose = verbose

        # Set up wavenumber quadrature
        self._setup_quadrature(do_trap)

        # Set up boundary conditions
        self.setBC()

    def _setup_quadrature(self, do_trap: bool):
        """
        Set up wavenumber quadrature points and weights.

        Parameters
        ----------
        do_trap : bool
            If True, use trapezoidal rule. If False, optimize quadrature.
        """
        if not do_trap:
            # Try to find optimal quadrature points and weights
            def get_phi(r):
                e = np.ones_like(r)

                def phi(k):
                    # Use log10 transform to enforce positivity
                    k = 10**k
                    A = r[:, None] * k0(r[:, None] * k)
                    v_i = A @ np.linalg.solve(A.T @ A, A.T @ e)
                    dv = (e - v_i) / len(r)
                    return np.linalg.norm(dv)

                def g(k):
                    A = r[:, None] * k0(r[:, None] * k)
                    return np.linalg.solve(A.T @ A, A.T @ e)

                return phi, g

            # Find minimum cell spacing and maximum mesh extent
            min_r = float(torch.min(self.mesh.edge_lengths).item())
            max_extent = (
                torch.max(self.mesh.nodes, dim=0)[0]
                - torch.min(self.mesh.nodes, dim=0)[0]
            )
            max_r = float(torch.max(max_extent).item())

            # Generate test points log-spaced between these endpoints
            rs = np.logspace(np.log10(min_r / 4), np.log10(max_r * 4), 100)

            min_rinv = -np.log10(rs).max()
            max_rinv = -np.log10(rs).min()

            # Initial guess for k_i's (in log10 space)
            k_i = np.linspace(min_rinv, max_rinv, self.nky)

            # Get optimization functions
            func, g_func = get_phi(rs)

            # Optimize
            out = minimize(func, k_i)

            if self.verbose:
                print(f"Optimized ks converged? : {out['success']}")
                print(f"Estimated transform error: {out['fun']}")

            if out["success"]:
                # Transform back to normal space
                points = 10 ** out["x"]
                # Transform has 2/pi and we want 1/pi, so divide by 2
                weights = g_func(points) / 2
            else:
                warnings.warn(
                    "Falling back to trapezoidal for integration. "
                    "You may need to change nky.",
                    stacklevel=2,
                )
                do_trap = True

        if do_trap:
            if self.verbose:
                print("Using trapezoidal quadrature")

            y = 0.0
            points = np.logspace(-4, 1, self.nky)
            dky = np.diff(points) / 2
            weights = np.r_[dky, 0] + np.r_[0, dky]
            weights *= np.cos(points * y)
            # Assume constant value at 0 frequency
            weights[0] += points[0] / 2 * (1.0 + np.cos(points[0] * y))
            weights /= np.pi

        # Store as torch tensors
        self._quad_points = torch.from_numpy(points).float()
        self._quad_weights = torch.from_numpy(weights).float()

    def setBC(self):
        """Set up boundary condition operators."""
        mesh = self.mesh

        # Standard cell-centered finite volume discretization
        V = sdiag(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T

        if self.bc_type == "Dirichlet":
            # Homogeneous Dirichlet is natural BC for this discretization
            return
        elif self.bc_type == "Neumann":
            # Set up Neumann BC
            alpha, beta, gamma = 0, 1, 0
            B, _ = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
            self.Grad = self.Grad - B
        else:
            raise ValueError(
                f"Unsupported boundary condition: {self.bc_type}. "
                "Use 'Dirichlet' or 'Neumann' (Robin not yet implemented)"
            )

    def condition_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """
        Condition the system matrix for numerical stability.
        Used for Neumann boundary conditions.

        Parameters
        ----------
        A : torch.Tensor
            Sparse system matrix

        Returns
        -------
        torch.Tensor
            Conditioned system matrix
        """
        # Modify first row: A[0, :] = 0 and A[0, 0] = 1
        mod_indices = torch.tensor([[0], [0]], dtype=torch.long, device=A.device)
        mod_values = torch.tensor([1.0], dtype=A.dtype, device=A.device)
        modification_matrix = torch.sparse_coo_tensor(
            mod_indices, mod_values, A.shape
        ).coalesce()

        # Mask for first row
        first_row_mask = A.indices()[0] == 0

        # Get values and columns of first row
        first_row_values = A.values()[first_row_mask]
        first_row_cols = A.indices()[1, first_row_mask]

        # Create sparse tensor for negative of first row
        neg_first_row_indices = torch.stack(
            [torch.zeros_like(first_row_cols), first_row_cols]
        )
        neg_first_row_values = -first_row_values
        neg_first_row_matrix = torch.sparse_coo_tensor(
            neg_first_row_indices, neg_first_row_values, A.shape
        ).coalesce()

        # A = A - A[0, :] + modification
        A = A + neg_first_row_matrix + modification_matrix
        return A

    def get_system_matrices(self) -> torch.Tensor:
        """
        Construct system matrices for all wavenumbers.

        Returns
        -------
        torch.Tensor
            System matrices for each wavenumber.
            Shape [n_ky, nC, nC]
        """
        D = self.Div
        G = self.Grad

        # Get face inner product with resistivity (inverted to get conductivity)
        MfRhoI = self.mesh.get_face_inner_product(
            self.mapping.forward(), invert_matrix=True
        )

        # Get cell mass matrix with conductivity (1/resistivity)
        # MccSigma = diag(sigma * cell_volumes)
        sigma = 1.0 / self.mapping.forward()  # conductivity = 1/resistivity
        V = self.mesh.cell_volumes
        MccSigma = sdiag(sigma * V)

        # Build system matrix for each wavenumber
        A_list = []
        for ky in self._quad_points:
            # A(ky) = D @ MfRhoI @ G + ky² @ MccSigma
            A = D @ MfRhoI @ G + (ky**2) * MccSigma

            if self.bc_type == "Neumann":
                A = self.condition_matrix(A)

            A_list.append(A)

        return torch.stack(A_list, dim=0)  # Shape [n_ky, nC, nC]

    def get_rhs_tensors(self) -> torch.Tensor:
        """
        Construct RHS vectors for all sources.

        The source term is independent of wavenumber.

        Returns
        -------
        torch.Tensor
            Batched RHS tensors. Shape [n_sources, nC]
        """
        return self.survey.get_source_tensor(self.mesh, projected_grid="CC")

    def fields_to_data(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Project solution fields to predicted data.

        This involves:
        1. Integrating fields across wavenumbers using quadrature weights
        2. Projecting integrated fields to receiver locations

        Parameters
        ----------
        fields : torch.Tensor
            Solution fields from PDE solve, shape [n_ky, n_sources, nC]

        Returns
        -------
        torch.Tensor
            Predicted data vector [n_data]
        """
        # Integrate across wavenumbers: sum over ky dimension with weights
        # fields shape: [n_ky, n_sources, nC]
        # weights shape: [n_ky]
        # Result shape: [n_sources, nC]

        # Expand weights to broadcast: [n_ky, 1, 1]
        weights = self._quad_weights.view(-1, 1, 1)

        # Weighted sum over wavenumber dimension
        fields_integrated = torch.sum(fields * weights, dim=0)  # [n_sources, nC]

        # Now project to data - same as 3D case
        sources = self.survey.source_list
        data_list = []

        for i, src in enumerate(sources):
            # Get field for this source
            src_field = fields_integrated[i]

            # Get receiver projection tensor and project to data
            rx_tensor = src.build_receiver_tensor(self.mesh, "CC")

            # Handle sparse tensor matrix multiplication
            if rx_tensor.is_sparse:
                rx_tensor_2d = rx_tensor[0]  # [n_receivers, n_mesh_cells]
                rx_data = torch.sparse.mm(
                    rx_tensor_2d, src_field.unsqueeze(-1)
                ).squeeze(-1)
            else:
                rx_data = rx_tensor.squeeze(0) @ src_field

            data_list.append(rx_data)

        return torch.cat(data_list, dim=0)
