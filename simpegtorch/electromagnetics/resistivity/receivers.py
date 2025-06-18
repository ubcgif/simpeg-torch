import torch
import numpy as np
from typing import Union, Optional, Tuple, List
import uuid


class BaseRx:
    """Base DC resistivity receiver class for simpeg-torch.

    Parameters
    ----------
    locations : torch.Tensor or array_like
        Receiver electrode locations
    data_type : str, default: "volt"
        Type of data observed: "volt", "apparent_resistivity", "apparent_chargeability"
    orientation : str, optional
        Orientation for field measurements: None, "x", "y", "z"
    projField : str, default: "phi"
        Fields projected from mesh: "phi", "e", "j"
    storeProjections : bool, default: False
        Whether to cache projection matrices
    """

    def __init__(
        self,
        locations: Union[torch.Tensor, np.ndarray],
        data_type: str = "volt",
        orientation: Optional[str] = None,
        projField: str = "phi",
        storeProjections: bool = False,
        **kwargs,
    ):
        self.locations = locations
        self.data_type = data_type
        self.orientation = orientation
        self.projField = projField
        self.storeProjections = storeProjections
        self._geometric_factor = {}
        self._Ps = {}
        self._uid = uuid.uuid4()

    @property
    def locations(self) -> torch.Tensor:
        """Receiver electrode locations

        Returns
        -------
        torch.Tensor
            Receiver locations, shape (n_receivers, dim) or list of such tensors
        """
        return self._locations

    @locations.setter
    def locations(self, locs: Union[torch.Tensor, np.ndarray, List]):
        if isinstance(locs, np.ndarray):
            locs = torch.from_numpy(locs).float()
        elif isinstance(locs, list):
            # For dipole receivers with separate M and N locations
            locs = [
                (
                    torch.tensor(loc, dtype=torch.float64)
                    if not isinstance(loc, torch.Tensor)
                    else loc
                )
                for loc in locs
            ]
        elif not isinstance(locs, torch.Tensor):
            locs = torch.tensor(locs, dtype=torch.float64)

        self._locations = locs

    @property
    def data_type(self) -> str:
        """Type of data observed"""
        return self._data_type

    @data_type.setter
    def data_type(self, dtype: str):
        valid_types = [
            "volt",
            "potential",
            "voltage",
            "apparent_resistivity",
            "apparent_chargeability",
        ]
        if dtype not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}, got {dtype}")
        self._data_type = dtype

    @property
    def orientation(self) -> Optional[str]:
        """Orientation of receiver measurements"""
        return self._orientation

    @orientation.setter
    def orientation(self, orient: Optional[str]):
        if orient is not None and orient not in ["x", "y", "z"]:
            raise ValueError("orientation must be None, 'x', 'y', or 'z'")
        self._orientation = orient

    @property
    def projField(self) -> str:
        """Field type being projected from mesh"""
        return self._projField

    @projField.setter
    def projField(self, field: str):
        if field not in ["phi", "e", "j"]:
            raise ValueError("projField must be 'phi', 'e', or 'j'")
        self._projField = field

    @property
    def uid(self):
        """Universal unique identifier"""
        return self._uid

    @property
    def nD(self) -> int:
        """Number of data points for this receiver

        Returns
        -------
        int
            Number of data points
        """
        if isinstance(self.locations, list):
            return self.locations[0].shape[0]  # For dipole receivers
        return self.locations.shape[0]

    @property
    def geometric_factor(self):
        """Geometric factors for apparent resistivity calculations"""
        return self._geometric_factor

    def getP(self, mesh, projected_grid: str = "CC") -> torch.Tensor:
        """Get projection matrix from mesh to receivers

        Parameters
        ----------
        mesh : TensorMesh
            The computational mesh
        projected_grid : str, default: "CC"
            Grid locations to project from:
            - "CC", "cell_centers" -> cell centers
            - "N", "nodes" -> nodes
            - "Ex", "edges_x" -> x-edges, etc.

        Returns
        -------
        torch.Tensor
            Projection matrix, shape (n_receivers, n_mesh_locations)
        """
        cache_key = (id(mesh), projected_grid)
        if cache_key in self._Ps:
            return self._Ps[cache_key]

        # Get interpolation matrix from mesh
        P = mesh.get_interpolation_matrix(self.locations, projected_grid)

        if self.storeProjections:
            self._Ps[cache_key] = P

        return P

    def evaluate(self, src, mesh, f: torch.Tensor) -> torch.Tensor:
        """Extract data from fields at receiver locations

        Parameters
        ----------
        src : BaseSrc
            Source object
        mesh : TensorMesh
            Computational mesh
        f : torch.Tensor
            Field solution on mesh

        Returns
        -------
        torch.Tensor
            Data extracted at receiver locations
        """
        # Determine projected grid based on simulation formulation
        # Check if we can detect nodal formulation from field size
        if f.numel() == mesh.nN:
            # Fields are on nodes - this is nodal formulation
            projected_grid = "N"
        elif self.orientation is not None:
            projected_grid = f"edges_{self.orientation}"
        else:
            projected_grid = "CC" if self.projField == "phi" else self.projField

        # Get projection matrix
        P = self.getP(mesh, projected_grid)

        # Project fields to receiver locations
        data = P @ f

        # Handle apparent resistivity
        if self.data_type == "apparent_resistivity":
            if src.uid in self.geometric_factor:
                data = data / self.geometric_factor[src.uid]
            else:
                raise KeyError(
                    "Geometric factor not set for apparent resistivity. "
                    "Call survey.set_geometric_factor() first."
                )

        return data

    def evalDeriv(
        self,
        src,
        mesh,
        f: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        adjoint: bool = False,
    ) -> torch.Tensor:
        """Derivative of receiver evaluation w.r.t. model

        Parameters
        ----------
        src : BaseSrc
            Source object
        mesh : TensorMesh
            Computational mesh
        f : torch.Tensor
            Field solution
        v : torch.Tensor, optional
            Vector for matrix-vector product
        adjoint : bool, default: False
            Return adjoint operation

        Returns
        -------
        torch.Tensor
            Derivative operation result
        """
        # Determine projected grid
        if self.orientation is not None:
            projected_grid = f"edges_{self.orientation}"
        else:
            projected_grid = "CC" if self.projField == "phi" else self.projField

        P = self.getP(mesh, projected_grid)

        # Handle apparent resistivity scaling
        factor = None
        if self.data_type == "apparent_resistivity":
            if src.uid in self.geometric_factor:
                factor = 1.0 / self.geometric_factor[src.uid]
            else:
                raise KeyError("Geometric factor not set for apparent resistivity")

        if v is None:
            # Return projection matrix
            if factor is not None:
                P = torch.diag(factor) @ P
            return P.T if adjoint else P

        # Matrix-vector product
        if not adjoint:
            result = P @ v
            if factor is not None:
                result = factor * result
        else:
            if factor is not None:
                v = factor * v
            result = P.T @ v

        return result


class Dipole(BaseRx):
    """Dipole receiver with M and N electrodes

    Parameters
    ----------
    locations_m : torch.Tensor or array_like
        M electrode locations
    locations_n : torch.Tensor or array_like
        N electrode locations
    locations : tuple of array_like, optional
        Alternative specification as (locations_m, locations_n)
    data_type : str, default: "volt"
        Type of data observed
    **kwargs
        Additional arguments passed to BaseRx
    """

    def __init__(
        self,
        locations_m: Optional[Union[torch.Tensor, np.ndarray]] = None,
        locations_n: Optional[Union[torch.Tensor, np.ndarray]] = None,
        locations: Optional[Tuple] = None,
        **kwargs,
    ):
        # Handle location specification
        if locations_m is not None or locations_n is not None:
            if locations_n is None or locations_m is None:
                raise ValueError("Both locations_m and locations_n must be provided")
            if locations is not None:
                raise ValueError(
                    "Cannot specify both 'locations' and 'locations_m/locations_n'"
                )
            locations = [locations_m, locations_n]

        if locations is None:
            raise AttributeError("Must specify receiver locations")

        if len(locations) != 2:
            raise ValueError(
                f"Dipole receiver needs 2 location arrays, got {len(locations)}"
            )

        # Convert to tensors and validate shapes
        if isinstance(locations[0], torch.Tensor):
            loc_m = locations[0].clone().detach().to(torch.float64)
        else:
            loc_m = torch.tensor(locations[0], dtype=torch.float64)

        if isinstance(locations[1], torch.Tensor):
            loc_n = locations[1].clone().detach().to(torch.float64)
        else:
            loc_n = torch.tensor(locations[1], dtype=torch.float64)

        if loc_m.shape != loc_n.shape:
            raise ValueError(
                f"M and N locations must have same shape: {loc_m.shape} vs {loc_n.shape}"
            )

        super().__init__(locations=[loc_m, loc_n], **kwargs)

    def __repr__(self):
        return ",\n".join(
            [
                f"{self.__class__.__name__}(m: {m}; n: {n})"
                for m, n in zip(self.locations_m, self.locations_n)
            ]
        )

    @property
    def locations_m(self) -> torch.Tensor:
        """M electrode locations"""
        return self.locations[0]

    @property
    def locations_n(self) -> torch.Tensor:
        """N electrode locations"""
        return self.locations[1]

    def getP(self, mesh, projected_grid: str = "CC") -> torch.Tensor:
        """Get projection matrix for dipole receiver (difference of interpolations)

        Parameters
        ----------
        mesh : TensorMesh
            Computational mesh
        projected_grid : str
            Grid locations to project from

        Returns
        -------
        torch.Tensor
            Projection matrix for M-N difference
        """
        cache_key = (id(mesh), projected_grid)
        if cache_key in self._Ps:
            return self._Ps[cache_key]

        # Get interpolation matrices for M and N electrodes
        P_m = mesh.get_interpolation_matrix(self.locations_m, projected_grid)
        P_n = mesh.get_interpolation_matrix(self.locations_n, projected_grid)

        # Dipole measurement is difference: P = P_m - P_n
        P = P_m - P_n

        if self.storeProjections:
            self._Ps[cache_key] = P

        return P


class Pole(BaseRx):
    """Pole receiver with single electrode

    Parameters
    ----------
    locations : torch.Tensor or array_like
        Electrode locations
    data_type : str, default: "volt"
        Type of data observed
    **kwargs
        Additional arguments passed to BaseRx
    """

    def __init__(self, locations: Union[torch.Tensor, np.ndarray], **kwargs):
        super().__init__(locations=locations, **kwargs)

    def __repr__(self):
        return ",\n".join(
            [f"{self.__class__.__name__}(location: {loc})" for loc in self.locations]
        )

    def getP(self, mesh, projected_grid: str = "CC") -> torch.Tensor:
        """Get projection matrix for pole receiver

        Parameters
        ----------
        mesh : TensorMesh
            Computational mesh
        projected_grid : str
            Grid locations to project from

        Returns
        -------
        torch.Tensor
            Projection matrix
        """
        cache_key = (id(mesh), projected_grid)
        if cache_key in self._Ps:
            return self._Ps[cache_key]

        P = mesh.get_interpolation_matrix(self.locations, projected_grid)

        if self.storeProjections:
            self._Ps[cache_key] = P

        return P
