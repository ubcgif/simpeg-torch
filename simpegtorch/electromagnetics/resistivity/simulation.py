import torch
from simpegtorch.torchmatsolver import TorchMatSolver
from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import sdiag
from scipy import sparse
from typing import List, Optional
from .sources import BaseSrc
from .receivers import BaseRx


class DCStaticSimulationCellCentered:
    def __init__(
        self,
        mesh: TensorMesh,
        sources: Optional[List[BaseSrc]] = None,
        receivers: Optional[List[BaseRx]] = None,
        survey=None,
    ):
        self.mesh = mesh
        self.sources = sources
        self.receivers = receivers
        self.survey = survey

    def setBC(self):
        mesh = self.mesh
        V = sdiag(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T
        print("Default BC is always Dirichilet for Now")

    def getA(self, resistivity):
        """
        Returns the system matrix A for the DC resistivity problem.
        If resitivity is provided, it computes the matrix based on the given resistivity values.
        """
        D = self.Div
        G = self.Grad

        MfRhoI = self.mesh.get_face_inner_product(resistivity, invert_matrix=True)

        A = D @ MfRhoI @ G
        return A

    def getRHS(self, source: Optional[BaseSrc] = None):
        """
        Returns the right-hand side vector b for the DC resistivity problem.

        Parameters
        ----------
        source : BaseSrc, optional
            Specific source to get RHS for. If None, uses all sources.
        """
        if source is not None:
            # Get RHS for specific source
            return source.evaluate(self)

        if self.sources is None:
            if self.survey is not None:
                # Use sources from survey
                sources = self.survey.source_list
            else:
                raise ValueError("Sources must be defined to compute RHS.")
        else:
            sources = self.sources

        # Sum contributions from all sources
        b = torch.zeros(self.mesh.nC, dtype=torch.float64, device=self.mesh.device)
        for src in sources:
            b += src.evaluate(self)

        return b

    def fields(self, resistivity, source: Optional[BaseSrc] = None):
        """
        Computes the electric fields for the DC resistivity problem.

        Parameters
        ----------
        resistivity : torch.Tensor
            Resistivity model values
        source : BaseSrc, optional
            Specific source to solve for. If None, solves for all sources.
        """
        A = self.getA(resistivity)

        if source is not None:
            # Solve for specific source
            b = self.getRHS(source)

            def solve_fn(A, b):
                return sparse.linalg.spsolve(A, b)

            x = TorchMatSolver.apply(A, b, solve_fn)
            return x

        # Solve for all sources
        if self.sources is None and self.survey is not None:
            sources = self.survey.source_list
        else:
            sources = self.sources

        if sources is None:
            raise ValueError("No sources defined")

        # Store fields for each source
        fields = {}
        for src in sources:
            b = self.getRHS(src)

            def solve_fn(A, b):
                return sparse.linalg.spsolve(A, b)

            fields[src] = TorchMatSolver.apply(A, b, solve_fn)

        return fields

    def dpred(self, m, f=None):
        """
        Compute predicted data for the given model.

        Parameters
        ----------
        m : torch.Tensor
            Model parameters (resistivity values)
        f : dict or torch.Tensor, optional
            Pre-computed fields. If provided, fields won't be recomputed.

        Returns
        -------
        torch.Tensor
            Predicted data vector
        """
        if f is None:
            f = self.fields(m)

        # Determine sources to use
        if self.sources is None and self.survey is not None:
            sources = self.survey.source_list
        else:
            sources = self.sources

        if sources is None:
            raise ValueError("No sources defined")

        # Extract data from fields using receivers
        data = torch.zeros(self.n_data, dtype=m.dtype, device=m.device)
        data_idx = 0

        for src in sources:
            # Get field for this source
            if isinstance(f, dict):
                src_field = f[src]
            else:
                # Single field (for single source case)
                src_field = f

            for rx in src.receiver_list:
                n_rx_data = rx.nD
                rx_data = rx.evaluate(src, self.mesh, src_field)
                data[data_idx : data_idx + n_rx_data] = rx_data
                data_idx += n_rx_data

        return data

    @property
    def n_data(self):
        """Total number of data points across all sources and receivers"""
        if self.sources is None and self.survey is not None:
            return self.survey.nD
        elif self.sources is not None:
            return sum(rx.nD for src in self.sources for rx in src.receiver_list)
        else:
            return 0

    def Jtvec(self, m, v, f=None):
        r"""Compute the Jacobian transpose times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an ``(n_data, n_param)`` matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jtvec`` method computes the matrix-vector product with the adjoint-sensitivity

        .. math::
            \mathbf{u} = \mathbf{J^T \, v}

        Parameters
        ----------
        m : torch.Tensor, shape (n_param,)
            The model parameters (resistivity values).
        v : torch.Tensor, shape (n_data,)
            Vector we are multiplying.
        f : torch.Tensor, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jtvec`.

        Returns
        -------
        torch.Tensor, shape (n_param,)
            The Jacobian transpose times a vector for the model and vector provided.
        """
        # Ensure model requires gradients
        if not m.requires_grad:
            m = m.requires_grad_(True)

        # Clear any existing gradients
        if m.grad is not None:
            m.grad.zero_()

        # Forward simulation to get predicted data
        predicted_data = self.dpred(m, f=f)

        # Ensure v has same shape as predicted_data
        if v.shape != predicted_data.shape:
            raise ValueError(
                f"Vector v shape {v.shape} must match predicted data shape {predicted_data.shape}"
            )

        # Compute J^T * v via backward pass
        # This computes the gradient of (v^T * predicted_data) w.r.t. m
        predicted_data.backward(gradient=v, retain_graph=True)

        return m.grad.clone()

    def Jvec(self, m, v, f=None):
        r"""Compute the Jacobian times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an ``(n_data, n_param)`` matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jvec`` method computes the matrix-vector product

        .. math::
            \mathbf{u} = \mathbf{J \, v}

        Parameters
        ----------
        m : torch.Tensor, shape (n_param,)
            The model parameters (resistivity values).
        v : torch.Tensor, shape (n_param,)
            Vector we are multiplying.
        f : torch.Tensor, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jvec`.

        Returns
        -------
        torch.Tensor, shape (n_data,)
            The Jacobian times a vector for the model and vector provided.
        """
        # Use finite differences for Jvec due to sparse matrix autodiff limitations
        eps = 1e-7
        f_plus = self.dpred(
            m + eps * v, f=None
        )  # Can't reuse fields for perturbed model
        f_minus = self.dpred(m - eps * v, f=None)
        return (f_plus - f_minus) / (2 * eps)
