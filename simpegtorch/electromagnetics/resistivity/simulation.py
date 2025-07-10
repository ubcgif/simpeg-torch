import torch
from simpegtorch.torchmatsolver.torchMUMPSsolver_batched import batched_mumps_solve
from simpegtorch.torchmatsolver.torchmatsolver_batched import batched_sparse_solve

from simpegtorch.discretize import TensorMesh
from simpegtorch.discretize.utils import sdiag
from typing import Optional
from .sources import BaseSrc


class BaseDcSimulation:
    """Base class for DC resistivity simulations with common functionality."""

    def __init__(self, mesh: TensorMesh, survey=None, **kwargs):
        self.mesh = mesh
        self.survey = survey

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

        if self.survey is not None:
            # Use sources from survey
            sources = self.survey.source_list
        else:
            raise ValueError("Sources must be defined to compute RHS.")

        # Sum contributions from all sources
        b = torch.zeros(self.mesh.nC, dtype=torch.float64, device=self.mesh.device)
        for src in sources:
            b += src.evaluate(self)

        return b

    def fields(self, resistivity=None, source: Optional[BaseSrc] = None):
        """
        Computes the electric fields/potentials for the DC resistivity problem.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fields method")

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

        sources = self.survey.source_list

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

            # Create batched tensor of all receiver data
            rx_tensor = src.build_receiver_tensor(self.mesh, self._get_formulation())

            # Project src field to receiver data
            src_field_b = self._prepare_field_for_receivers(src_field, rx_tensor)
            rx_data = torch.bmm(rx_tensor, src_field_b)
            rx_data = rx_data.flatten(start_dim=0, end_dim=-1)

            if src.receiver_list[0].data_type == "apparent_resistivity":
                # Apply geometric factors properly to each receiver's data slice
                rx_data_idx = 0
                for rx in src.receiver_list:
                    if src.uid in rx.geometric_factor:
                        geometric_factor = rx.geometric_factor[src.uid]
                        # Slice rx_data for this receiver's data points
                        rx_slice = slice(rx_data_idx, rx_data_idx + rx.nD)
                        rx_data[rx_slice] = rx_data[rx_slice] / geometric_factor
                        rx_data_idx += rx.nD
                    else:
                        raise KeyError(
                            "Geometric factor not set for apparent resistivity. "
                            "Call survey.set_geometric_factor() first."
                        )

            # Store data for each receiver
            data[data_idx : data_idx + rx_data.shape[0]] = rx_data
            data_idx += rx_data.shape[0]

        return data

    @property
    def n_data(self):
        """Total number of data points across all sources and receivers"""
        if self.survey is not None:
            return self.survey.nD
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

    def _get_formulation(self):
        """Get the formulation string for receiver tensor building. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_formulation method")

    def _prepare_field_for_receivers(self, src_field, rx_tensor):
        """Prepare field tensor for receiver computation. Must be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _prepare_field_for_receivers method"
        )


class Simulation3DCellCentered(BaseDcSimulation):
    def __init__(
        self,
        mesh: TensorMesh,
        survey=None,
        bc_type: str = "Dirichlet",
        sigma=None,
        rho=None,
        **kwargs,
    ):
        super().__init__(mesh, survey, **kwargs)
        self.bc_type = bc_type
        if sigma is not None:
            self.sigma = torch.as_tensor(sigma)
        if rho is not None:
            self.rho = torch.as_tensor(rho)
        self.setBC()

    @property
    def bc_type(self):
        """Boundary condition type for the simulation."""
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value: str):
        """Set the boundary condition type."""
        valid_bc_types = ["Dirichlet", "Neumann"]
        if value not in valid_bc_types:
            raise ValueError(
                f"Invalid boundary condition type: {value}. Must be one of {valid_bc_types}."
            )
        # TODO implement Robin and mixed boundary conditions
        self._bc_type = value

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

    def getA(self, resistivity):
        """
        Returns the system matrix A for the DC resistivity problem.
        If resistivity is provided, it computes the matrix based on the given resistivity values.
        """
        D = self.Div
        G = self.Grad

        # Use face inner product with inverse resistivity (conductivity)
        self.MfRhoI = self.mesh.get_face_inner_product(resistivity, invert_matrix=True)

        A = D @ self.MfRhoI @ G

        if self.bc_type.lower() == "neumann":
            # Get the shape of A
            n_rows, n_cols = A.shape

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

    def _get_formulation(self):
        """Get the formulation string for receiver tensor building."""
        return "CC"  # Cell-centered formulation

    def _prepare_field_for_receivers(self, src_field, rx_tensor):
        """Prepare field tensor for receiver computation."""
        # rx_tensor shape: [n_rx, 1, n_cells], src_field shape: [n_cells]
        # Need src_field_b shape: [n_rx, n_cells, 1] for bmm
        return src_field.unsqueeze(-1).unsqueeze(0).repeat(rx_tensor.shape[0], 1, 1)

    def fields(self, resistivity=None, source: Optional[BaseSrc] = None):
        """
        Computes the electric fields for the DC resistivity problem.

        Parameters
        ----------
        resistivity : torch.Tensor, optional
            Resistivity model values. If None, uses stored sigma converted to resistivity.
        source : BaseSrc, optional
            Specific source to solve for. If None, solves for all sources.
        """
        # Use stored sigma if resistivity not provided
        if resistivity is None:
            if self.sigma is None and self.rho is None:
                raise ValueError(
                    "Either provide resistivity or set sigma in constructor"
                )
            # Convert conductivity to resistivity
            elif self.sigma is not None:
                resistivity = 1.0 / self.sigma
            else:
                resistivity = self.rho

        A = self.getA(resistivity)

        if source is not None:
            # Solve for specific source
            b = self.getRHS(source)

            try:
                return batched_mumps_solve(A, b)
            except ImportError:
                print("MUMPS not installed, falling back")
                return batched_sparse_solve(A, b)

        sources = self.survey.source_list

        if sources is None:
            raise ValueError("No sources defined")

        # Store fields for each source
        fields = {}

        b_tensor = self.survey.get_source_tensor(self)  # Cell-centered formulation

        # Use native batched solvers instead of vmap for better gradient support
        try:
            print("Using batched MUMPS solver")
            field_tensor = batched_mumps_solve(A, b_tensor)
        except ImportError:
            print("MUMPS not installed, falling back to scipy")
            field_tensor = batched_sparse_solve(A, b_tensor)

        for i, src in enumerate(sources):
            # Store field for each source
            fields[src] = field_tensor[i]

        return fields


class Simulation3DNodal(BaseDcSimulation):
    # Define formulation type for source evaluation
    _formulation = "EB"  # Electric field and magnetic flux density (nodal potentials)

    def __init__(
        self,
        mesh: TensorMesh,
        survey=None,
        bc_type: str = "Neumann",
        sigma=None,
        **kwargs,
    ):
        super().__init__(mesh, survey, **kwargs)
        self.bc_type = bc_type
        self.sigma = sigma
        self.setBC()

    @property
    def bc_type(self):
        """Boundary condition type for the simulation."""
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value: str):
        """Set the boundary condition type."""
        valid_bc_types = ["Neumann"]
        if value not in valid_bc_types:
            raise ValueError(
                f"Invalid boundary condition type: {value}. Must be one of {valid_bc_types}."
            )
        # TODO implement Robin and mixed boundary conditions
        self._bc_type = value

    def setBC(self):
        mesh = self.mesh
        V = sdiag(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T
        if self.bc_type == "Dirichlet":
            print("Homogeneous Dirichlet is the natural BC for this CC discretization")
        elif self.bc_type.lower() == "neumann":
            return
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
            # TODO: Implement Robin and mixed boundary conditions

    def _get_formulation(self):
        """Get the formulation string for receiver tensor building."""
        return "N"  # Nodal formulation

    def _prepare_field_for_receivers(self, src_field, rx_tensor):
        """Prepare field tensor for receiver computation."""
        # project src field to receiver data, rx_tensor is (n_rx, 1, n_cells),
        # src field should be (n_rx, n_cells, 1)
        return src_field.unsqueeze(1).unsqueeze(0).repeat(rx_tensor.shape[0], 1, 1)

    def getA(self, resistivity):
        """
        Returns the system matrix A for the nodal DC resistivity problem.

        For Neumann boundary conditions, the system matrix has a null space
        corresponding to a constant potential shift. We remove this null space
        by fixing one node potential to zero.

        Parameters
        ----------
        resistivity : torch.Tensor
            Resistivity values on the mesh

        Returns
        -------
        torch.Tensor
            System matrix A with null space removed for Neumann BCs
        """
        MeSigma = self.mesh.get_edge_inner_product(1.0 / resistivity)
        Grad = self.mesh.nodal_gradient
        A = Grad.T @ MeSigma @ Grad

        if self.bc_type.lower() == "neumann":
            # Get the shape of A
            n_rows, n_cols = A.shape

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

    def _apply_neumann_bc_to_rhs(self, b):
        """
        Apply Neumann boundary condition modification to RHS vector.

        For Neumann BCs, we fix the potential at the first node to zero,
        so we modify the RHS accordingly.

        Parameters
        ----------
        b : torch.Tensor
            Right-hand side vector

        Returns
        -------
        torch.Tensor
            Modified RHS vector for Neumann BCs
        """
        if self.bc_type.lower() == "neumann":
            b_modified = b.clone()
            b_modified[0] = 0.0  # Fix potential at node 0 to zero
            return b_modified
        return b

    def _apply_neumann_bc_to_rhs_tensor(self, b_tensor):
        """
        Apply Neumann boundary condition modification to RHS tensor.

        For Neumann BCs, we fix the potential at the first node to zero,
        so we modify the RHS accordingly.

        Parameters
        ----------
        b_tensor : torch.Tensor
            Right-hand side tensor

        Returns
        -------
        torch.Tensor
            Modified RHS tensor for Neumann BCs
        """
        if self.bc_type.lower() == "neumann":
            b_tensor_modified = b_tensor.clone()
            b_tensor_modified[:, 0] = 0.0
            return b_tensor_modified
        return b_tensor

    def fields(self, resistivity=None, source: Optional[BaseSrc] = None):
        """
        Computes the electric potentials for the nodal DC resistivity problem.

        Parameters
        ----------
        resistivity : torch.Tensor, optional
            Resistivity model values. If None, uses stored sigma converted to resistivity.
        source : BaseSrc, optional
            Specific source to solve for. If None, solves for all sources.
        """
        # Use stored sigma if resistivity not provided
        if resistivity is None:
            if self.sigma is None and self.rho is None:
                raise ValueError(
                    "Either provide resistivity or set sigma in constructor"
                )
            # Convert conductivity to resistivity
            elif self.sigma is not None:
                resistivity = 1.0 / self.sigma
            else:
                resistivity = self.rho

        A = self.getA(resistivity)

        if source is not None:
            # Solve for specific source
            b = self.getRHS(source)
            b = self._apply_neumann_bc_to_rhs(b)

            try:
                return batched_mumps_solve(A, b)
            except ImportError:
                print("MUMPS not installed, falling back")
                return batched_sparse_solve(A, b)

        sources = self.survey.source_list

        if sources is None:
            raise ValueError("No sources defined")

        # Store fields for each source
        fields = {}

        b_tensor = self.survey.get_source_tensor(self)  # Cell-centered formulation
        b_tensor = self._apply_neumann_bc_to_rhs_tensor(b_tensor)

        try:
            field_tensor = batched_mumps_solve(A, b_tensor)
        except ImportError:
            print("MUMPS not installed, falling back")
            field_tensor = batched_sparse_solve(A, b_tensor)

        for i, src in enumerate(sources):
            # Store field for each source
            fields[src] = field_tensor[i]

        return fields
