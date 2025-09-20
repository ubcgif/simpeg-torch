import torch
import numpy as np
from discretize.utils import Zero


class BaseFDEMSrc:
    """
    Base FDEM source class
    ----------
    receiver_list : list of FDEM receivers
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : array_like, default: ``None``
        Source location.
    """

    _ePrimary = None
    _bPrimary = None
    _hPrimary = None
    _jPrimary = None

    def __init__(self, receiver_list, frequency, location=None, **kwargs):
        self.receiver_list = receiver_list if receiver_list is not None else []
        self.frequency = frequency
        if location is not None:
            if torch.is_tensor(location):
                self.location = location.detach().clone().to(dtype=torch.float64)
            else:
                self.location = torch.tensor(location, dtype=torch.float64)
        else:
            self.location = None
        self._fields_per_source = 1

    @property
    def frequency(self):
        """Source frequency

        Returns
        -------
        float
            Source frequency
        """
        return self._frequency

    @frequency.setter
    def frequency(self, freq):
        self._frequency = freq

    def bPrimary(self, simulation):
        """Compute primary magnetic flux density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic flux density
        """
        if self._bPrimary is None:
            return Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary magnetic flux density times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary magnetic flux density times a vector
        """
        return Zero()

    def hPrimary(self, simulation):
        """Compute primary magnetic field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic field
        """
        if self._hPrimary is None:
            return Zero()
        return self._hPrimary

    def ePrimary(self, simulation):
        """Compute primary electric field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary electric field
        """
        if self._ePrimary is None:
            return Zero()
        return self._ePrimary

    def jPrimary(self, simulation):
        """Compute primary current density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary current density
        """
        if self._jPrimary is None:
            return Zero()
        return self._jPrimary

    def evaluate(self, simulation):
        """
        Return source terms for this source.

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            FDEM simulation object

        Returns
        -------
        s_m : torch.Tensor
            Magnetic source terms
        s_e : torch.Tensor
            Electric source terms
        """
        raise NotImplementedError("evaluate must be implemented in derived classes")


class MagneticDipole(BaseFDEMSrc):
    """
    Magnetic dipole source for FDEM simulations.

    Parameters
    ----------
    receiver_list : list
        List of receivers
    frequency : float
        Source frequency in Hz
    location : array_like
        Source location [x, y, z]
    moment : float or array_like, default: 1.0
        Magnetic moment [Am²]. Can be scalar or 3-vector for directional moment.
    orientation : str, default: 'z'
        Dipole orientation ('x', 'y', 'z') if moment is scalar
    """

    def __init__(
        self, receiver_list, frequency, location, moment=1.0, orientation="z", **kwargs
    ):
        super().__init__(receiver_list, frequency, location, **kwargs)

        if np.isscalar(moment):
            # Convert scalar moment to vector based on orientation
            moment_vec = torch.zeros(3, dtype=torch.float64)
            orient_idx = {"x": 0, "y": 1, "z": 2}[orientation]
            moment_vec[orient_idx] = moment
            self.moment = moment_vec
        else:
            if torch.is_tensor(moment):
                self.moment = moment.detach().clone().to(dtype=torch.float64)
            else:
                self.moment = torch.tensor(moment, dtype=torch.float64)

    def evaluate(self, simulation):
        """
        Evaluate magnetic dipole source terms.

        Returns source terms for the magnetic dipole at the specified location.
        For EB formulation, the magnetic dipole contributes to the magnetic source term.
        """
        # Initialize source vectors
        if simulation._formulation == "EB":
            s_m = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
        elif simulation._formulation == "HJ":
            s_m = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
        else:
            raise ValueError(f"Unknown formulation: {simulation._formulation}")

        if self.location is not None:
            # Find closest face/edge to source location
            if simulation._formulation == "EB":
                # For EB formulation, magnetic sources go on faces
                face_centers = simulation.mesh.faces
                distances = torch.norm(face_centers - self.location, dim=1)
                closest_idx = torch.argmin(distances)

                # Add magnetic dipole source (simplified - point source)
                # In practice, you'd want better interpolation
                s_m[closest_idx] = self.moment[2]  # Assuming z-oriented for now

            elif simulation._formulation == "HJ":
                # For HJ formulation, magnetic sources go on edges
                edge_centers = simulation.mesh.edges
                distances = torch.norm(edge_centers - self.location, dim=1)
                closest_idx = torch.argmin(distances)
                s_m[closest_idx] = self.moment[2]

        return s_m, s_e


class ElectricDipole(BaseFDEMSrc):
    """
    Electric dipole source for FDEM simulations.

    Parameters
    ----------
    receiver_list : list
        List of receivers
    frequency : float
        Source frequency in Hz
    location : array_like
        Source location [x, y, z]
    current : float, default: 1.0
        Electric current in Amperes
    length : float, default: 1.0
        Dipole length in meters
    orientation : str, default: 'z'
        Dipole orientation ('x', 'y', 'z')
    """

    def __init__(
        self,
        receiver_list,
        frequency,
        location,
        current=1.0,
        length=1.0,
        orientation="z",
        **kwargs,
    ):
        super().__init__(receiver_list, frequency, location, **kwargs)
        self.current = current
        self.length = length
        self.orientation = orientation

    def evaluate(self, simulation):
        """
        Evaluate electric dipole source terms.

        For EB formulation, electric dipole contributes to electric source term.
        """
        # Initialize source vectors
        if simulation._formulation == "EB":
            s_m = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
        elif simulation._formulation == "HJ":
            s_m = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
        else:
            raise ValueError(f"Unknown formulation: {simulation._formulation}")

        if self.location is not None:
            # Electric dipole moment
            dipole_moment = self.current * self.length

            if simulation._formulation == "EB":
                # For EB formulation, electric sources go on edges
                edge_centers = simulation.mesh.edges
                distances = torch.norm(edge_centers - self.location, dim=1)
                closest_idx = torch.argmin(distances)
                s_e[closest_idx] = dipole_moment

            elif simulation._formulation == "HJ":
                # For HJ formulation, electric sources go on faces
                face_centers = simulation.mesh.faces
                distances = torch.norm(face_centers - self.location, dim=1)
                closest_idx = torch.argmin(distances)
                s_e[closest_idx] = dipole_moment

        return s_m, s_e


class LoopSource(BaseFDEMSrc):
    """
    Circular loop source for FDEM simulations.

    Parameters
    ----------
    receiver_list : list
        List of receivers
    frequency : float
        Source frequency in Hz
    location : array_like
        Loop center location [x, y, z]
    radius : float, default: 1.0
        Loop radius in meters
    current : float, default: 1.0
        Loop current in Amperes
    orientation : str, default: 'z'
        Normal direction of loop plane ('x', 'y', 'z')
    n_segments : int, default: 20
        Number of segments to discretize loop
    """

    def __init__(
        self,
        receiver_list,
        frequency,
        location,
        radius=1.0,
        current=1.0,
        orientation="z",
        n_segments=20,
        **kwargs,
    ):
        super().__init__(receiver_list, frequency, location, **kwargs)
        self.radius = radius
        self.current = current
        self.orientation = orientation
        self.n_segments = n_segments

    def evaluate(self, simulation):
        """
        Evaluate loop source terms.

        Discretizes the loop into line segments and treats each as a small electric dipole.
        """
        # Initialize source vectors
        if simulation._formulation == "EB":
            s_m = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
        elif simulation._formulation == "HJ":
            s_m = torch.zeros(simulation.mesh.n_edges, dtype=torch.complex128)
            s_e = torch.zeros(simulation.mesh.n_faces, dtype=torch.complex128)
        else:
            raise ValueError(f"Unknown formulation: {simulation._formulation}")

        if self.location is not None:
            # Create loop geometry
            theta = torch.linspace(0, 2 * torch.pi, self.n_segments + 1)[:-1]

            # Define loop coordinates based on orientation
            if self.orientation == "z":
                x = self.location[0] + self.radius * torch.cos(theta)
                y = self.location[1] + self.radius * torch.sin(theta)
                z = torch.full_like(x, self.location[2])
            elif self.orientation == "y":
                x = self.location[0] + self.radius * torch.cos(theta)
                z = self.location[2] + self.radius * torch.sin(theta)
                y = torch.full_like(x, self.location[1])
            elif self.orientation == "x":
                y = self.location[1] + self.radius * torch.cos(theta)
                z = self.location[2] + self.radius * torch.sin(theta)
                x = torch.full_like(y, self.location[0])

            loop_points = torch.stack([x, y, z], dim=1)

            # Segment length
            segment_length = 2 * torch.pi * self.radius / self.n_segments

            # Add source at each segment
            for i in range(self.n_segments):
                segment_center = loop_points[i]

                if simulation._formulation == "EB":
                    edge_centers = simulation.mesh.edges
                    distances = torch.norm(edge_centers - segment_center, dim=1)
                    closest_idx = torch.argmin(distances)
                    s_e[closest_idx] += self.current * segment_length / self.n_segments

                elif simulation._formulation == "HJ":
                    face_centers = simulation.mesh.faces
                    distances = torch.norm(face_centers - segment_center, dim=1)
                    closest_idx = torch.argmin(distances)
                    s_e[closest_idx] += self.current * segment_length / self.n_segments

        return s_m, s_e
