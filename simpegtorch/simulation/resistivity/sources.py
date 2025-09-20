import torch
import numpy as np
from typing import List, Union, Optional, Tuple
import uuid


class BaseSrc:
    """Base DC resistivity source class for simpeg-torch.

    Parameters
    ----------
    receiver_list : list of BaseRx objects
        List of receivers associated with this source
    location : torch.Tensor or array_like
        Source electrode locations
    current : float or torch.Tensor, default: 1.0
        Current amplitude [A]
    """

    def __init__(
        self,
        receiver_list: Optional[List] = None,
        location: Optional[Union[torch.Tensor, np.ndarray]] = None,
        current: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ):
        if receiver_list is None:
            receiver_list = []
        self.receiver_list = receiver_list

        if location is not None:
            self.location = location
        self.current = current
        self._uid = uuid.uuid4()
        self.projection_tensor = None

    @property
    def location(self) -> torch.Tensor:
        """Source electrode locations

        Returns
        -------
        torch.Tensor
            Source electrode locations, shape (n_electrodes, dim)
        """
        return self._location

    @location.setter
    def location(self, loc: Union[torch.Tensor, np.ndarray, list]):
        if isinstance(loc, np.ndarray):
            loc = torch.from_numpy(loc).float()
        elif isinstance(loc, list):
            # Handle list of locations (for multipole sources)
            tensor_locs = []
            for l in loc:
                if isinstance(l, torch.Tensor):
                    tensor_locs.append(l.to(torch.float64))
                elif isinstance(l, np.ndarray):
                    tensor_locs.append(torch.from_numpy(l).to(torch.float64))
                else:
                    tensor_locs.append(torch.tensor(l, dtype=torch.float64))

            # Ensure each location has proper dimensions
            for i, t_loc in enumerate(tensor_locs):
                if t_loc.dim() == 1:
                    tensor_locs[i] = t_loc.unsqueeze(0)

            # Stack all locations into a single tensor
            loc = torch.cat(tensor_locs, dim=0)
        elif not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float64)

        if isinstance(loc, torch.Tensor) and loc.dim() == 1:
            loc = loc.unsqueeze(0)  # Add batch dimension for single electrode

        self._location = loc

    @property
    def current(self) -> torch.Tensor:
        """Current amplitudes for source electrodes

        Returns
        -------
        torch.Tensor
            Current amplitudes, shape (n_electrodes,)
        """
        return self._current

    @current.setter
    def current(self, current: Union[float, torch.Tensor, np.ndarray]):
        if isinstance(current, (int, float)):
            current = torch.tensor([current], dtype=torch.float64)
        elif isinstance(current, np.ndarray):
            current = torch.from_numpy(current).float()
        elif not isinstance(current, torch.Tensor):
            current = torch.tensor(current, dtype=torch.float64)

        if current.dim() == 0:
            current = current.unsqueeze(0)

        self._current = current

    @property
    def uid(self):
        """Universal unique identifier"""
        return self._uid

    @property
    def receivers(self):
        """Alias for receiver_list for compatibility"""
        return self.receiver_list

    def evaluate(self, mesh, projected_grid: str = "CC"):
        """Discretize source to mesh

        Parameters
        ----------
        mesh : TensorMesh
            Discretization mesh
        projected_grid : str, optional
            Formulation type ("CC" for cell-centered, "N" for nodal), by default "CC"

        Returns
        -------
        torch.Tensor
            Right-hand side vector for this source
        """

        indices = mesh.closest_points_index(self.location, grid_loc=projected_grid)
        q = torch.zeros(mesh.nC, dtype=self.current.dtype, device=self.current.device)
        q[indices] = self.current
        return q

    def build_receiver_tensor(self, mesh, projected_grid: str = "N"):
        """Build a batched tensor of projection matrices for all receivers associated with this source."""
        # Create cache key that includes the projected_grid parameter
        cache_key = f"{id(mesh)}_{projected_grid}"

        # Check if we have a cached tensor for this mesh and grid combination
        if (
            hasattr(self, "_projection_tensor_cache")
            and cache_key in self._projection_tensor_cache
        ):
            return self._projection_tensor_cache[cache_key]

        projection_matrices = []

        for rx in self.receiver_list:
            # Get projection matrix for this receiver
            P = rx.getP(mesh, projected_grid)
            projection_matrices.append(P)
        # Stack all projection matrices into a single tensor
        projection_tensor = torch.stack(projection_matrices, dim=0)

        # Cache the projection tensor with the proper key
        if not hasattr(self, "_projection_tensor_cache"):
            self._projection_tensor_cache = {}
        self._projection_tensor_cache[cache_key] = projection_tensor

        return projection_tensor


class Multipole(BaseSrc):
    """Generic multipole source for arbitrary electrode configurations"""

    @property
    def location_a(self) -> torch.Tensor:
        """Locations of A electrodes (positive current)"""
        return self.location

    @property
    def location_b(self) -> torch.Tensor:
        """Locations of B electrodes (return current) - NaN for multipole"""
        return torch.full_like(self.location, float("nan"))


class Dipole(BaseSrc):
    """Dipole source with A and B electrodes

    Parameters
    ----------
    receiver_list : list of BaseRx objects
        List of receivers associated with this source
    location_a : torch.Tensor or array_like
        A electrode location
    location_b : torch.Tensor or array_like
        B electrode location
    location : tuple of array_like, optional
        Alternative way to specify (location_a, location_b)
    current : float, default: 1.0
        Current amplitude [A]
    """

    def __init__(
        self,
        receiver_list: List,
        location_a: Optional[Union[torch.Tensor, np.ndarray]] = None,
        location_b: Optional[Union[torch.Tensor, np.ndarray]] = None,
        location: Optional[Tuple] = None,
        current: float = 1.0,
        **kwargs,
    ):
        # Handle location specification
        if location is None and location_a is None and location_b is None:
            raise TypeError(
                "Must specify either 'location' or both 'location_a' and 'location_b'"
            )

        if location is not None and (location_a is not None or location_b is not None):
            raise TypeError(
                "Cannot specify both 'location' and 'location_a'/'location_b'"
            )

        if location is None:
            if location_a is None or location_b is None:
                raise TypeError(
                    "When 'location' is None, both 'location_a' and 'location_b' must be provided"
                )
            location = [location_a, location_b]

        if len(location) != 2:
            raise ValueError(
                f"location must have length 2 for dipole source, got {len(location)}"
            )

        # Convert to tensors
        if isinstance(location[0], torch.Tensor):
            loc_a = location[0].clone().to(torch.float64)
        else:
            loc_a = torch.tensor(location[0], dtype=torch.float64)

        if isinstance(location[1], torch.Tensor):
            loc_b = location[1].clone().to(torch.float64)
        else:
            loc_b = torch.tensor(location[1], dtype=torch.float64)

        if loc_a.dim() == 1:
            loc_a = loc_a.unsqueeze(0)
        if loc_b.dim() == 1:
            loc_b = loc_b.unsqueeze(0)

        combined_location = torch.cat([loc_a, loc_b], dim=0)

        super().__init__(
            receiver_list=receiver_list,
            location=combined_location,
            current=torch.tensor([current, -current], dtype=torch.float64),
            **kwargs,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(a: {self.location_a}; b: {self.location_b})"

    @property
    def location_a(self) -> torch.Tensor:
        """Location of A electrode (positive current)"""
        return self.location[0]

    @property
    def location_b(self) -> torch.Tensor:
        """Location of B electrode (negative current)"""
        return self.location[1]


class Pole(BaseSrc):
    """Pole source with single electrode

    Parameters
    ----------
    receiver_list : list of BaseRx objects
        List of receivers associated with this source
    location : torch.Tensor or array_like
        Electrode location
    current : float, default: 1.0
        Current amplitude [A]
    """

    def __init__(
        self,
        receiver_list: List,
        location: Union[torch.Tensor, np.ndarray],
        current: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            receiver_list=receiver_list, location=location, current=current, **kwargs
        )

        if len(self.location) != 1:
            raise ValueError(
                f"Pole sources only have a single electrode, got {len(self.location)}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(location: {self.location_a})"

    @property
    def location_a(self) -> torch.Tensor:
        """Location of A electrode"""
        return self.location[0]

    @property
    def location_b(self) -> torch.Tensor:
        """Location of B electrode (NaN for pole)"""
        return torch.full_like(self.location[0], float("nan"))
