import torch
from simpegtorch.torchmatsolver import batched_sparse_solve

from simpegtorch.discretize import TensorMesh

from discretize.utils import Zero
from simpegtorch.discretize.utils import speye

# from simpegtorch.discretize.utils import sdiag


class BaseFDEMSimulation:
    """
    Base class for FDEM simulations.
    """

    def __init__(
        self,
        mesh: TensorMesh,
        survey=None,
        sigma=None,
        permittivity=None,
        mu=None,
        makeASymmetric=True,
    ):
        self.mesh = mesh
        self.survey = survey
        self.permittivity = permittivity
        self.mu = (
            mu if mu is not None else torch.ones(mesh.n_cells, dtype=torch.float64)
        )
        self._makeASymmetric = makeASymmetric
        self._formulation = "EB"  # Electric field, Magnetic flux density
        self.sigma = (
            sigma
            if sigma is not None
            else torch.ones(mesh.n_cells, dtype=torch.complex128)
        )

    def getRHS(self, freq):
        """
        Get the right-hand side (RHS) vector for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getRHS must be implemented in derived classes.")

    def getA(self, freq):
        """
        Get the system matrix (A) for the simulation.
        This method should be implemented in derived classes.
        """
        raise NotImplementedError("getA must be implemented in derived classes.")

    @property
    def getA_tensor(self):
        """
        Returns a (BxNxN) tensor representing the system matrix all of the system matrices for each frequency in the survey.
        Returns
        -------
        torch.Tensor
            A tensor containing the system matrices for each frequency in the survey.
        """
        return torch.stack([self.getA(freq) for freq in self.survey.frequencies], dim=0)

    @property
    def getRHS_tensor(self):
        """
        Returns a (BxNxM) tensor representing the right-hand side vectors for each frequency in the survey.

        Returns
        -------
        torch.Tensor
            A tensor containing the right-hand side vectors for each frequency in the survey.
        """
        return torch.stack(
            [self.getRHS(freq) for freq in self.survey.frequencies], dim=0
        )

    def getSourceTerm(self, freq):
        """
        Returns the discrete source terms for the frequency provided.

        This moethod computes and returns the discrete magnetics and electric source terms for all soundings at the frequency provided. The exact shape and
        implementation of the source terms when solving for the fields at each frequency
        is formulation dependent.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        s_m : numpy.ndarray
            The magnetic sources terms. (n_faces, n_sources) for EB-formulations. (n_edges, n_sources) for HJ-formulations.
        s_e : numpy.ndarray
            The electric sources terms. (n_edges, n_sources) for EB-formulations. (n_faces, n_sources) for HJ-formulations.
        """
        Srcs = self.survey.get_sources_by_frequency(freq)
        n_fields = sum(src._fields_per_source for src in Srcs)
        if self._formulation == "EB":
            s_m = torch.zeros((self.mesh.n_faces, n_fields), dtype=torch.complex128)
            s_e = torch.zeros((self.mesh.n_edges, n_fields), dtype=torch.complex128)
        elif self._formulation == "HJ":
            s_m = torch.zeros((self.mesh.n_edges, n_fields), dtype=torch.complex128)
            s_e = torch.zeros((self.mesh.n_faces, n_fields), dtype=torch.complex128)

        i = 0
        for src in Srcs:
            ii = i + src._fields_per_source
            smi, sei = src.evaluate(self)
            if not isinstance(smi, Zero) and smi.ndim == 1:
                smi = smi[:, None]
            if not isinstance(sei, Zero) and sei.ndim == 1:
                sei = sei[:, None]
            s_m[:, i:ii] = s_m[:, i:ii] + smi
            s_e[:, i:ii] = s_e[:, i:ii] + sei
            i = ii
        return s_m, s_e

    def fields(self, freq=None):
        """
        Compute the fields, if a frequency is supplied it will compute the fields at that frequency,
        otherwise it will compute the fields for all frequencies in the survey.

        Parameters
        ----------
        freq : float
            The frequency at which to compute the fields.

        Returns
        -------
        dict or torch.Tensor
            The computed fields. If freq is None, returns a dict keyed by frequency.
            If freq is provided, returns a tensor for that frequency.
        """

        if freq is None:
            # Compute fields for all frequencies separately
            fields = {}
            for f in self.survey.frequencies:
                A = self.getA(f)
                rhs = self.getRHS(f)

                # Solve for this frequency
                if isinstance(A, torch.sparse.FloatTensor):
                    fields[f] = batched_sparse_solve(A, rhs)
                else:
                    # For 2D case, solve directly without batching
                    if rhs.ndim == 1:
                        fields[f] = torch.linalg.solve(A.to_dense(), rhs)
                    else:
                        fields[f] = torch.linalg.solve(A.to_dense(), rhs)
            return fields

        else:
            # If a frequency is provided, compute fields for that specific frequency
            A = self.getA(freq)
            rhs = self.getRHS(freq)

            # Solve the system of equations
            if isinstance(A, torch.sparse.FloatTensor):
                return batched_sparse_solve(A, rhs)
            else:
                # For 2D case, solve directly without batching
                if rhs.ndim == 1:
                    return torch.linalg.solve(A.to_dense(), rhs)
                else:
                    return torch.linalg.solve(A.to_dense(), rhs)

    def _get_admittivity(self, freq):
        if self.permittivity is not None:
            return self.sigma + 1j * self.permittivity * 2 * torch.pi * (freq)
        else:
            return self.sigma

    def _get_face_admittivity_property_matrix(
        self, freq, invert_model=False, invert_matrix=False
    ):
        """
        Face inner product matrix with permittivity and resistivity
        """
        yhat = self._get_admittivity(freq)
        return self.mesh.get_face_inner_product(
            yhat, invert_model=invert_model, invert_matrix=invert_matrix
        )

    def _get_edge_admittivity_property_matrix(
        self, freq, invert_model=False, invert_matrix=False
    ):
        """
        Face inner product matrix with permittivity and resistivity
        """
        yhat = self._get_admittivity(freq)
        return self.mesh.get_edge_inner_product(
            yhat, invert_model=invert_model, invert_matrix=invert_matrix
        )

    @property
    def MfMui(self):
        """Face inner product matrix for inverse permeability"""
        return self.mesh.get_face_inner_product(1.0 / self.mu).to(
            dtype=torch.complex128
        )

    @property
    def MeSigmaI(self):
        """Edge inner product matrix for conductivity (inverted)"""
        return self.mesh.get_edge_inner_product(self.sigma, invert_matrix=True).to(
            dtype=torch.complex128
        )

    def dpred(self, m=None, f=None):
        """
        Compute predicted data for the given model.

        Parameters
        ----------
        m : torch.Tensor, optional
            Model parameters (conductivity values). If None, uses current sigma.
        f : dict or torch.Tensor, optional
            Pre-computed fields. If provided, fields won't be recomputed.

        Returns
        -------
        torch.Tensor
            Predicted data vector
        """
        # Update model if provided
        if m is not None:
            self.sigma = m

        # Compute fields if not provided
        if f is None:
            f = self.fields()

        # Initialize data storage
        data = torch.zeros(self.survey.nD, dtype=torch.complex128)
        data_idx = 0

        # Extract data from each source
        for freq in self.survey.frequencies:
            sources = self.survey.get_sources_by_frequency(freq)

            # Set current frequency for receivers that need it
            self._current_freq = freq

            # Get fields for this frequency
            if isinstance(f, dict):
                fields_freq = f[freq] if freq in f else f[list(f.keys())[0]]
            elif isinstance(f, torch.Tensor) and f.ndim == 3:
                # Batched fields tensor (n_freq, n_faces, n_sources)
                freq_idx = self.survey.frequencies.index(freq)
                fields_freq = f[freq_idx]
            else:
                # Single frequency case
                fields_freq = f

            source_idx = 0
            for src in sources:
                # Get fields for this source
                if fields_freq.ndim == 2:
                    # Multiple sources: (n_faces, n_sources)
                    src_fields = fields_freq[:, source_idx]
                else:
                    # Single source: (n_faces,)
                    src_fields = fields_freq

                # Extract data from each receiver
                for rx in src.receiver_list:
                    rx_data = rx.evaluate(src_fields, simulation=self)
                    data[data_idx : data_idx + rx.nD] = rx_data
                    data_idx += rx.nD

                source_idx += 1

        return data


class Simulation3DMagneticFluxDensity(BaseFDEMSimulation):
    """
    3D FDEM simulation in terms of magnetic flux field
    """

    def getA(self, freq):
        """
        Get the system matrix (A) for the simulation at the given frequency.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        torch.Tensor
            The system matrix for the simulation.
        """
        # This method should be implemented based on the specific formulation and mesh
        MfMui = self.MfMui
        C = self.mesh.edge_curl.to(dtype=torch.complex128)
        iomega = (
            1j * 2 * torch.pi * freq * speye(self.mesh.n_faces, dtype=torch.complex128)
        )

        if self.permittivity is None:
            MeSigmaI = self.MeSigmaI
            A = C @ (MeSigmaI @ (C.T @ MfMui)) + iomega
        else:
            MeyhatI = self._get_edge_admittivity_property_matrix(
                freq, invert_matrix=True
            ).to(dtype=torch.complex128)
            A = C @ (MeyhatI @ (C.T @ MfMui)) + iomega

        if self._makeASymmetric:
            return MfMui.T @ A
        return A

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        This method returns the right-hand sides for the frequency provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = \mathbf{C M_{e\sigma}^{-1} s_e} - i \omega \mathbf{s_m }

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_faces, n_sources) numpy.ndarray
            The right-hand sides.
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl.to(dtype=torch.complex128)

        if self.permittivity is None:
            MeSigmaI = self.MeSigmaI
            RHS = s_m + C @ (MeSigmaI @ s_e)
        else:
            MeyhatI = self._get_edge_admittivity_property_matrix(
                freq, invert_matrix=True
            ).to(dtype=torch.complex128)
            RHS = s_m + C @ (MeyhatI @ s_e)

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T @ RHS

        return RHS
