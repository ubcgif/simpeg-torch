import torch
from typing import List
from ..basePDE import BasePDE
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import speye
from discretize.utils import Zero


class FDEMPDE(BasePDE):
    """
    Frequency Domain Electromagnetic PDE formulation.

    Solves the curl-curl equation in the frequency domain:
    ∇ × (μ⁻¹ ∇ × E) + iωσE = -iωs_m - ∇ × (μ⁻¹ s_m)

    where:
    - E is electric field
    - μ is permeability
    - σ is conductivity
    - ω is angular frequency
    - s_m, s_e are magnetic and electric source terms
    """

    def __init__(
        self, mesh: TensorMesh, formulation: str = "EB", make_symmetric: bool = True
    ):
        """
        Initialize FDEM PDE.

        Parameters
        ----------
        mesh : TensorMesh
            Discretization mesh
        formulation : str
            "EB" for electric field/magnetic flux density formulation
            "HJ" for magnetic field/current density formulation
        make_symmetric : bool
            Whether to make system matrix symmetric
        """
        super().__init__(mesh)
        self.formulation = formulation
        self.make_symmetric = make_symmetric

        # Default material properties
        self.sigma = None
        self.permittivity = None
        self.mu = torch.ones(mesh.n_cells, dtype=torch.float64)

    def set_material_properties(self, sigma=None, permittivity=None, mu=None):
        """Set material properties for the PDE."""
        if sigma is not None:
            self.sigma = sigma
        if permittivity is not None:
            self.permittivity = permittivity
        if mu is not None:
            self.mu = mu

    def get_system_matrices(
        self, model_params: torch.Tensor, frequencies: List[float]
    ) -> List[torch.Tensor]:
        """
        Construct frequency-dependent system matrices.

        Parameters
        ----------
        model_params : torch.Tensor
            Conductivity values [n_cells]
        frequencies : List[float]
            List of frequencies in Hz

        Returns
        -------
        List[torch.Tensor]
            List of system matrices, one for each frequency
        """
        # Update conductivity from model parameters
        self.sigma = model_params.to(dtype=torch.complex128)

        matrices = []
        for freq in frequencies:
            A = self._get_system_matrix_at_frequency(freq)
            matrices.append(A)

        return matrices

    def _get_system_matrix_at_frequency(self, freq: float) -> torch.Tensor:
        """Get system matrix for a specific frequency."""
        if self.formulation == "EB":
            # Electric field formulation: solve for magnetic flux density B
            return self._get_A_magnetic_flux_density(freq)
        else:
            raise NotImplementedError(f"Formulation {self.formulation} not implemented")

    def _get_A_magnetic_flux_density(self, freq: float) -> torch.Tensor:
        """
        System matrix for magnetic flux density formulation.

        A = C @ MeSigmaI @ C^T @ MfMui + iω I

        where:
        - C is curl operator
        - MeSigmaI is inverse edge conductivity matrix
        - MfMui is inverse face permeability matrix
        - I is identity
        """
        # Get material property matrices
        MfMui = self._get_face_permeability_inverse()
        C = self.mesh.edge_curl.to(dtype=torch.complex128)
        iomega = (
            1j * 2 * torch.pi * freq * speye(self.mesh.n_faces, dtype=torch.complex128)
        )

        if self.permittivity is None:
            # Only conductivity, no permittivity
            MeSigmaI = self._get_edge_conductivity_inverse()
            A = C @ (MeSigmaI @ (C.T @ MfMui)) + iomega
        else:
            # Include both conductivity and permittivity (admittivity)
            MeYhatI = self._get_edge_admittivity_inverse(freq)
            A = C @ (MeYhatI @ (C.T @ MfMui)) + iomega

        if self.make_symmetric:
            return MfMui.T @ A
        return A

    def _get_face_permeability_inverse(self) -> torch.Tensor:
        """Face inner product matrix for inverse permeability."""
        return self.mesh.get_face_inner_product(1.0 / self.mu).to(
            dtype=torch.complex128
        )

    def _get_edge_conductivity_inverse(self) -> torch.Tensor:
        """Edge inner product matrix for inverse conductivity."""
        return self.mesh.get_edge_inner_product(self.sigma, invert_matrix=True).to(
            dtype=torch.complex128
        )

    def _get_edge_admittivity_inverse(self, freq: float) -> torch.Tensor:
        """Edge inner product matrix for inverse admittivity (σ + iωε)."""
        admittivity = self.sigma + 1j * self.permittivity * 2 * torch.pi * freq
        return self.mesh.get_edge_inner_product(admittivity, invert_matrix=True).to(
            dtype=torch.complex128
        )

    def get_rhs_tensors(self, survey) -> torch.Tensor:
        """
        Construct RHS vectors for all frequencies and sources.

        Parameters
        ----------
        survey : FDEMSurvey
            FDEM survey with sources and frequencies

        Returns
        -------
        torch.Tensor
            RHS tensor [n_frequencies, n_sources, n_faces] for EB formulation
        """
        frequencies = survey.frequencies
        rhs_list = []

        for freq in frequencies:
            rhs_freq = self._get_rhs_at_frequency(freq, survey)
            rhs_list.append(rhs_freq)

        return torch.stack(rhs_list, dim=0)  # [n_freq, n_sources, n_faces]

    def _get_rhs_at_frequency(self, freq: float, survey) -> torch.Tensor:
        """Get RHS for specific frequency."""
        if self.formulation == "EB":
            return self._get_rhs_magnetic_flux_density(freq, survey)
        else:
            raise NotImplementedError(f"Formulation {self.formulation} not implemented")

    def _get_rhs_magnetic_flux_density(self, freq: float, survey) -> torch.Tensor:
        """
        RHS for magnetic flux density formulation.

        RHS = s_m + C @ MeSigmaI @ s_e

        where s_m and s_e are integrated source terms.
        """
        # Get sources at this frequency
        sources = survey.get_sources_by_frequency(freq)
        n_sources = len(sources)

        # Initialize RHS tensor
        rhs_tensor = torch.zeros((n_sources, self.mesh.n_faces), dtype=torch.complex128)

        # Get curl operator and conductivity matrix
        C = self.mesh.edge_curl.to(dtype=torch.complex128)

        if self.permittivity is None:
            MeSigmaI = self._get_edge_conductivity_inverse()
        else:
            MeSigmaI = self._get_edge_admittivity_inverse(freq)

        # Evaluate each source
        for i, src in enumerate(sources):
            s_m, s_e = src.evaluate(self)

            # Handle zero sources
            if isinstance(s_m, Zero):
                s_m = torch.zeros(self.mesh.n_faces, dtype=torch.complex128)
            if isinstance(s_e, Zero):
                s_e = torch.zeros(self.mesh.n_edges, dtype=torch.complex128)

            # Ensure proper shapes
            if s_m.ndim == 1:
                s_m = s_m.unsqueeze(-1)
            if s_e.ndim == 1:
                s_e = s_e.unsqueeze(-1)

            # Compute RHS: s_m + C @ MeSigmaI @ s_e
            rhs = s_m.squeeze() + (C @ (MeSigmaI @ s_e)).squeeze()

            # Apply symmetry transformation if needed
            if self.make_symmetric:
                MfMui = self._get_face_permeability_inverse()
                rhs = MfMui.T @ rhs

            rhs_tensor[i] = rhs

        return rhs_tensor

    def fields_to_data(self, fields: torch.Tensor, survey) -> torch.Tensor:
        """
        Project solution fields to predicted data.

        Parameters
        ----------
        fields : torch.Tensor
            Solution fields [n_freq, n_sources, n_faces] for EB formulation
        survey : FDEMSurvey
            Survey with receivers

        Returns
        -------
        torch.Tensor
            Predicted data vector
        """
        frequencies = survey.frequencies
        data_list = []

        for freq_idx, freq in enumerate(frequencies):
            sources = survey.get_sources_by_frequency(freq)

            # Set current frequency for receivers that need it
            for src in sources:
                for rx in src.receiver_list:
                    if hasattr(rx, "_current_freq"):
                        rx._current_freq = freq

            fields_freq = fields[freq_idx]  # [n_sources, n_faces]

            for src_idx, src in enumerate(sources):
                src_fields = fields_freq[src_idx]  # [n_faces]

                # Extract data from each receiver
                for rx in src.receiver_list:
                    rx_data = rx.evaluate(src_fields, simulation=self)
                    data_list.append(rx_data)

        return torch.cat(data_list, dim=0)
