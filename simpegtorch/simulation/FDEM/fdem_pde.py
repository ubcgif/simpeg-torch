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
            self.mu = torch.ones_like(self.mesh.n_cells) * 4 * 3.141592653589793 * 1e-7
        else:
            self.mu = mu

        self.frequencies = torch.tensor(survey.frequencies)

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
        MeSigmaI = self.mesh.get_edge_inner_product(self.sigma)
        A = torch.stack(
            [C @ MeSigmaI @ C.T @ MfMui + io * speye(self.mesh.nF) for io in iomega]
        )

        return A

    def get_rhs_tensors(self):
        """ "
        Provide the right-hand side tensors for the FDEM PDE.
        Returns
        -------
        (f, n_sources, n_faces) torch.Tensor
            Right-hand side tensor for f frequencies, n_sources, and n_faces
        """
        return None

    def fields_to_data(self, fields):
        """
        Project fields to data space.
        Parameters
        ----------
        fields : (f, n_sources, n_faces) torch.Tensor
            Fields tensor for f frequencies, n_sources, and n_faces
        Returns
        -------
        (nD,) torch.Tensor
            Data vector of length nD
        """
