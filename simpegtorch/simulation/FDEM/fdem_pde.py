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
        sigma,
        mu=None,
        makeASymmetric=True,
    ):
        """
        Initialize the FDEM3DMagneticFluxDensity PDE.

        Parameters
        ----------
        mesh : TensorMesh
            The computational mesh.
        survey : Survey
            The survey object containing sources and receivers.
        sigma_mapping : Mapping
            Mapping from model parameters to conductivity.
        mu_mapping : Mapping, optional
            Mapping from model parameters to permeability. If None, assumes default mu_0.
        """
        super().__init__(mesh, survey)

        self.sigma = sigma
        if mu is None:
            self.mu = (
                torch.ones(self.mesh.n_cells, dtype=torch.float64)
                * 4
                * 3.141592653589793
                * 1e-7
            )
        else:
            self.mu = mu

        self.survey = survey  # Store survey for later use
        self.frequencies = torch.tensor(survey.frequencies)
        self._formulation = "EB"  # E-B formulation (magnetic flux density)

    def get_system_matrices(self):
        r"""
        Provide the system matrices for the FDEM PDE.

        This method returns a fxNxN sparse matrix representing the system for f frequencies and N cells.

        .. math::
            \mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the curl operator
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        Returns
        -------
        (f, N, N) torch.sparse.Tensor
            System matrix for f frequencies and N cells
        """

        MfMui = self.mesh.get_face_inner_product(1.0 / self.mu)
        C = self.mesh.edge_curl
        iomega = [1j * 2 * 3.141592653589793 for f in self.frequencies]
        MeSigmaI = self.mesh.get_edge_inner_product(self.sigma.forward())
        A = torch.stack(
            [C @ MeSigmaI @ C.T @ MfMui + io * speye(self.mesh.nF) for io in iomega]
        )

        return A

    def get_rhs_tensors(self):
        """
        Provide the right-hand side tensors for the FDEM PDE.

        The solver expects RHS organized for Cartesian product batching:
        - System matrices: (n_freq, n_grid, n_grid)
        - RHS tensors: (n_sources_total, n_grid)
        - Output: (n_freq, n_sources_total, n_grid)

        We organize sources by frequency then concatenate all sources.

        Returns
        -------
        torch.Tensor
            Right-hand side tensor, shape (n_sources_total, n_grid_points)
        """
        # Get source tensors organized by frequency
        source_dict = self.survey.get_source_tensor(self.mesh, self._formulation)

        # Concatenate all sources across all frequencies
        # This gives us (n_sources_total, n_grid_points)
        rhs_list = []
        for freq in self.survey.frequencies:
            rhs_list.append(source_dict[freq])

        # Stack all sources: (n_sources_total, n_grid_points)
        rhs_tensors = torch.cat(rhs_list, dim=0)

        return rhs_tensors

    def fields_to_data(self, fields):
        """
        Project fields to data space.

        The solver returns fields as a Cartesian product:
        - fields shape: (n_freq, n_sources_total, n_grid_points)

        We need to:
        1. Match each field solution to its corresponding source
        2. Project through receivers for that source
        3. Organize data in the correct order

        Parameters
        ----------
        fields : torch.Tensor
            Fields tensor shape (n_freq, n_sources_total, n_grid_points)

        Returns
        -------
        torch.Tensor
            Data vector of length nD
        """
        data_list = []
        source_idx = 0

        # Iterate through frequencies
        for freq_idx, freq in enumerate(self.survey.frequencies):
            sources = self.survey.get_sources_by_frequency(freq)

            # For each source at this frequency
            for src in sources:
                # Get the field for this source at this frequency
                # Shape: (n_grid_points,)
                src_field = fields[freq_idx, source_idx, :]

                # Project through each receiver
                for rx in src.receiver_list:
                    # Use receiver's evaluate method to get data
                    rx_data = rx.evaluate(src_field, self)
                    data_list.append(rx_data)

                source_idx += 1

        # Concatenate all data
        return torch.cat(data_list, dim=0)
