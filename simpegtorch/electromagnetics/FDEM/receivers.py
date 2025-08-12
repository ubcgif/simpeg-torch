import torch


class BaseFDEMReceiver:
    """Base class for FDEM receivers."""

    def __init__(self, locations, orientation="z", component="real"):
        """
        Parameters
        ----------
        locations : array_like
            Receiver locations (n_receivers, 3)
        orientation : str
            Field component orientation ('x', 'y', 'z')
        component : str
            Complex component ('real', 'imag')
        """
        if torch.is_tensor(locations):
            self.locations = locations.detach().clone().to(dtype=torch.float64)
        else:
            self.locations = torch.tensor(locations, dtype=torch.float64)
        if self.locations.ndim == 1:
            self.locations = self.locations.reshape(1, -1)
        self.orientation = orientation
        self.component = component

    @property
    def nD(self):
        """Number of data points"""
        return self.locations.shape[0]

    def evaluate(self, fields, simulation):
        """Extract data from fields at receiver locations"""
        raise NotImplementedError("evaluate must be implemented in derived classes")


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
        # Get interpolation matrix from faces to receiver locations
        P = simulation.mesh.get_interpolation_matrix(self.locations, location_type="F")

        # Interpolate fields to receiver locations
        b_interp = P @ fields

        # For 3D, need to extract the correct component
        if simulation.mesh.dim == 3:
            # Reshape to (n_receivers, 3) for vector field
            b_vector = b_interp.reshape(-1, 3)

            # Extract component
            comp_idx = {"x": 0, "y": 1, "z": 2}[self.orientation]
            b_comp = b_vector[:, comp_idx]
        else:
            # For 2D, assume z-component
            b_comp = b_interp

        # Return real or imaginary part
        if self.component == "real":
            return b_comp.real
        else:
            return b_comp.imag


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
        C = simulation.mesh.edge_curl

        # Compute electric field from magnetic flux density
        # E = -iω * μ⁻¹ * C.T * B
        freq = getattr(simulation, "_current_freq", 1.0)  # Need current frequency
        omega = 2 * torch.pi * freq

        # Get permeability (assume vacuum for now)
        mu0 = 4e-7 * torch.pi

        e_field = -1j * omega / mu0 * (C.T @ fields)

        # Interpolate to receiver locations
        P = simulation.mesh.get_interpolation_matrix(self.locations, location_type="E")

        e_interp = P @ e_field

        # Extract component
        if simulation.mesh.dim == 3:
            e_vector = e_interp.reshape(-1, 3)
            comp_idx = {"x": 0, "y": 1, "z": 2}[self.orientation]
            e_comp = e_vector[:, comp_idx]
        else:
            e_comp = e_interp

        # Return real or imaginary part
        if self.component == "real":
            return e_comp.real
        else:
            return e_comp.imag
