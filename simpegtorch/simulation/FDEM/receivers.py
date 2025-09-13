import torch


class BaseFDEMReceiver:
    """Base class for FDEM receivers."""

    def __init__(self, locations, orientation="z", component="real"):
        """
        Parameters
        ----------
        locations : array_like
            Receiver locations (n_receivers, 3)
        orientation : str or array_like
            Field component orientation. Can be 'x', 'y', 'z' or a 3-vector like [0, 0, 1]
        component : str
            Complex component ('real', 'imag')
        """
        if torch.is_tensor(locations):
            self.locations = locations.detach().clone().to(dtype=torch.float64)
        else:
            self.locations = torch.tensor(locations, dtype=torch.float64)
        if self.locations.ndim == 1:
            self.locations = self.locations.reshape(1, -1)

        # Convert orientation to vector format
        if isinstance(orientation, str):
            orient_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            self.orientation = torch.tensor(
                orient_map[orientation], dtype=torch.float64
            )
        else:
            self.orientation = torch.tensor(orientation, dtype=torch.float64)

        self.component = component

    @property
    def nD(self):
        """Number of data points"""
        return self.locations.shape[0]

    def evaluate(self, fields, simulation):
        """Extract data from fields at receiver locations"""
        raise NotImplementedError("evaluate must be implemented in derived classes")

    def _get_projection_matrix(self, mesh, projected_grid):
        """
        Get projection matrix from mesh to receivers (following SimPEG pattern).

        Parameters
        ----------
        mesh : TensorMesh
            The mesh
        projected_grid : str

        Returns
        -------
        torch.Tensor
            Projection matrix
        """
        # Initialize zero matrix
        n_grid = getattr(mesh, f"n_{projected_grid[:-1]}")
        P = torch.zeros(self.locations.shape[0], n_grid, dtype=torch.complex128)

        # Component-wise interpolation like SimPEG
        for strength, comp in zip(self.orientation, ["x", "y", "z"]):
            if strength != 0.0:
                location_type = projected_grid + comp  # e.g., "faces_x"
                P_comp = mesh.get_interpolation_matrix(
                    self.locations, location_type=location_type
                )
                P = P + strength * P_comp.to(dtype=torch.complex128)

        return P


class PointMagneticFluxDensity(BaseFDEMReceiver):
    """Point receiver for magnetic flux density measurements."""

    def evaluate(self, fields, simulation):
        """
        Extract magnetic flux density at receiver locations.

        Parameters
        ----------
        fields : torch.Tensor
            Computed fields from simulation (n_faces,)
        simulation : BaseFDEMSimulation
            FDEM simulation object

        Returns
        -------
        torch.Tensor
            Data at receiver locations
        """
        # Get projection matrix using component-wise interpolation (like SimPEG)
        P = self._get_projection_matrix(simulation.mesh, "faces_")

        # Interpolate fields to receiver locations
        b_interp = P @ fields

        # Return real or imaginary part
        if self.component == "real":
            return b_interp.real
        else:
            return b_interp.imag


class PointMagneticFluxDensitySecondary(PointMagneticFluxDensity):
    """Point receiver for secondary magnetic flux density measurements."""

    def evaluate(self, fields, simulation):
        """Extract secondary magnetic flux density (total - primary)"""
        # Get total field
        b_total = super().evaluate(fields, simulation)

        # Get primary field from source
        # For now, assume primary field is negligible (air/vacuum background)
        # In practice, you'd compute the primary field analytically
        b_primary = torch.zeros_like(b_total)

        return b_total - b_primary


class PointElectricField(BaseFDEMReceiver):
    """Point receiver for electric field measurements."""

    def evaluate(self, fields, simulation):
        """
        Extract electric field at receiver locations.

        For EB formulation, we need to compute E from B using:
        E = -iωμ⁻¹∇×B
        """
        # Get curl matrix
        C = simulation.mesh.edge_curl.to(dtype=torch.complex128)

        # Compute electric field from magnetic flux density
        # E = -iω * μ⁻¹ * C.T * B
        freq = getattr(simulation, "_current_freq", 1.0)  # Need current frequency
        omega = 2 * torch.pi * freq

        # Get permeability (assume vacuum for now)
        mu0 = 4e-7 * torch.pi

        e_field = -1j * omega / mu0 * (C.T @ fields)

        # Interpolate to receiver locations
        P = self._get_projection_matrix(simulation.mesh, "edges_")

        e_interp = P @ e_field

        # Return real or imaginary part
        if self.component == "real":
            return e_interp.real
        else:
            return e_interp.imag
