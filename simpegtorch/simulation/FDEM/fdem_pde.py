import torch
from simpegtorch.simulation.base.basePDE import BasePDE
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import speye


class FDEM3DMagneticFluxDensity(BasePDE):
    """
    FDEM (Frequency Domain Electromagnetics) PDE formulation for magnetic flux density.

    Solves the frequency-domain Maxwell's equations in terms of magnetic flux density:
    ∇ × (μ⁻¹ ∇ × b) + iωσb = s

    where:
    - b is magnetic flux density
    - μ is magnetic permeability
    - σ is conductivity
    - ω is angular frequency (2πf)
    - s is the source term
    """

    def __init__(
        self,
        mesh: TensorMesh,
        survey,
        mapping,
        permittivity=None,
        mu=None,
        makeASymmetric: bool = True,
    ):
        """
        Initialize FDEM PDE.

        Parameters
        ----------
        mesh : TensorMesh
            Discretization mesh
        survey : Survey
            Survey object with sources, receivers, and frequencies
        mapping : BaseMapping
            Parameter generating function (typically for conductivity)
        permittivity : torch.Tensor, optional
            Electric permittivity values (default: None, uses only conductivity)
        mu : torch.Tensor, optional
            Magnetic permeability values (default: 1.0)
        makeASymmetric : bool, optional
            If True, makes the system matrix symmetric (default: True)
        """
        super().__init__(mesh, mapping)
        self.survey = survey
        self.permittivity = permittivity
        self.mu = (
            mu
            if mu is not None
            else torch.ones(mesh.n_cells, dtype=torch.float64, device=mesh.device)
        )
        self._makeASymmetric = makeASymmetric
        self._formulation = "EB"  # Electric field, Magnetic flux density

    @property
    def MfMui(self):
        """Face inner product matrix for inverse permeability."""
        mu_inv = (1.0 / self.mu).to(dtype=torch.complex128)
        return self.mesh.get_face_inner_product(mu_inv)

    @property
    def MeSigmaI(self):
        """Edge inner product matrix for conductivity (inverted)."""
        sigma = self.mapping.forward().to(dtype=torch.complex128)
        return self.mesh.get_edge_inner_product(sigma, invert_matrix=True)

    def _get_admittivity(self, freq):
        """
        Get admittivity: σ + iωε

        Parameters
        ----------
        freq : float
            Frequency in Hz

        Returns
        -------
        torch.Tensor
            Admittivity (complex-valued)
        """
        sigma = self.mapping.forward()
        if self.permittivity is not None:
            omega = 2 * torch.pi * freq
            return sigma + 1j * self.permittivity * omega
        else:
            return sigma

    def _get_edge_admittivity_property_matrix(self, freq, invert_matrix=False):
        """
        Edge inner product matrix with admittivity.

        Parameters
        ----------
        freq : float
            Frequency in Hz
        invert_matrix : bool
            If True, invert the matrix

        Returns
        -------
        torch.Tensor
            Edge inner product matrix
        """
        yhat = self._get_admittivity(freq)
        # Admittivity is already complex-valued
        return self.mesh.get_edge_inner_product(yhat, invert_matrix=invert_matrix)

    def get_system_matrices(self) -> torch.Tensor:
        """
        Construct system matrices for all frequencies.

        For FDEM, we solve:
        [C' M_μ⁻¹ C M_σ⁻¹ + iωI] b = rhs

        Returns
        -------
        torch.Tensor
            System matrices, shape [n_frequencies, n_faces, n_faces]
        """
        frequencies = self.survey.frequencies
        n_freq = len(frequencies)
        n_faces = self.mesh.n_faces

        # Pre-compute frequency-independent matrices (already complex from properties)
        MfMui = self.MfMui
        C = self.mesh.edge_curl.to(dtype=torch.complex128)

        # Stack system matrices for all frequencies
        system_matrices = []

        for freq in frequencies:
            # iω identity matrix
            omega = 2 * torch.pi * freq
            iomega = 1j * omega * speye(n_faces, dtype=torch.complex128)

            if self.permittivity is None:
                # Simple conductivity case
                MeSigmaI = self.MeSigmaI  # Already complex
                A = C @ (MeSigmaI @ (C.T @ MfMui)) + iomega
            else:
                # Full admittivity case (conductivity + displacement current)
                yhat = self._get_admittivity(freq)  # Returns complex
                MeyhatI = self.mesh.get_edge_inner_product(yhat, invert_matrix=True)
                A = C @ (MeyhatI @ (C.T @ MfMui)) + iomega

            # Make symmetric if requested (improves conditioning)
            if self._makeASymmetric:
                A = MfMui.T @ A

            system_matrices.append(A)

        return torch.stack(system_matrices, dim=0)

    def get_rhs_tensors(self) -> torch.Tensor:
        """
        Construct RHS vectors for all sources at all frequencies.

        The RHS includes both electric and magnetic sources:
        rhs = C M_σ⁻¹ s_e + s_m

        Returns
        -------
        torch.Tensor
            Batched RHS tensors, shape [n_sources, n_faces]
        """
        sources = self.survey.source_list
        C = self.mesh.edge_curl.to(dtype=torch.complex128)

        rhs_list = []

        for src in sources:
            freq = src.frequency

            # Get source terms from the source object
            s_m, s_e = self._get_source_terms(src, freq)

            if self.permittivity is None:
                MeSigmaI = self.MeSigmaI  # Already complex
                rhs = s_m + C @ (MeSigmaI @ s_e)
            else:
                yhat = self._get_admittivity(freq)  # Returns complex
                MeyhatI = self.mesh.get_edge_inner_product(yhat, invert_matrix=True)
                rhs = s_m + C @ (MeyhatI @ s_e)

            # Make symmetric if requested
            if self._makeASymmetric:
                MfMui = self.MfMui  # Already complex
                rhs = MfMui.T @ rhs

            rhs_list.append(rhs)

        return torch.stack(rhs_list, dim=0)

    def _get_source_terms(self, src, freq):
        """
        Extract source terms from a source object.

        Parameters
        ----------
        src : BaseFDEMSrc
            Source object
        freq : float
            Frequency in Hz

        Returns
        -------
        s_m : torch.Tensor
            Magnetic source term, shape [n_faces] (for EB formulation)
        s_e : torch.Tensor
            Electric source term, shape [n_edges] (for EB formulation)
        """
        # Use the source's evaluate method
        s_m, s_e = src.evaluate(self)

        # Ensure proper dimensions
        if s_m.ndim == 1:
            s_m = s_m  # Already correct shape
        elif s_m.ndim == 2:
            s_m = s_m.squeeze(-1)  # Remove extra dimension if present

        if s_e.ndim == 1:
            s_e = s_e
        elif s_e.ndim == 2:
            s_e = s_e.squeeze(-1)

        return s_m, s_e

    def fields_to_data(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Project solution fields to predicted data.

        Parameters
        ----------
        fields : torch.Tensor
            Solution fields from PDE solve
            Shape: [n_frequencies, n_sources_per_freq, n_faces]
            or [n_sources_total, n_faces] if flattened

        Returns
        -------
        torch.Tensor
            Predicted data vector [n_data]
        """
        # Group sources by frequency
        sources_by_freq = {}
        for src in self.survey.source_list:
            freq = src.frequency
            if freq not in sources_by_freq:
                sources_by_freq[freq] = []
            sources_by_freq[freq].append(src)

        data_list = []
        source_idx = 0

        # Process each frequency
        for freq_idx, freq in enumerate(self.survey.frequencies):
            sources = sources_by_freq.get(freq, [])

            for i, src in enumerate(sources):
                # Extract field for this source
                if fields.ndim == 3:
                    # Shape: [n_frequencies, n_sources_per_freq, n_faces]
                    src_field = fields[freq_idx, i]
                else:
                    # Shape: [n_sources_total, n_faces]
                    src_field = fields[source_idx]

                # Extract data from each receiver
                for rx in src.receiver_list:
                    rx_data = rx.evaluate(src_field, simulation=self)
                    data_list.append(rx_data)

                source_idx += 1

        return torch.cat(data_list, dim=0)
