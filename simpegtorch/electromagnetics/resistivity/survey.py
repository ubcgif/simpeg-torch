import torch
import uuid
from typing import List, Dict, Tuple, Union
from .sources import BaseSrc, Dipole as SrcDipole, Pole as SrcPole, Multipole
from .receivers import BaseRx, Dipole as RxDipole, Pole as RxPole


class Survey:
    """DC resistivity survey class for simpeg-torch.

    Parameters
    ----------
    source_list : list of BaseSrc objects
        List of DC sources with their associated receivers
    survey_geometry : str, default: "surface"
        Survey geometry: "surface", "borehole", or "general"
    """

    def __init__(
        self, source_list: List[BaseSrc], survey_geometry: str = "surface", **kwargs
    ):
        self.source_list = source_list
        self.survey_geometry = survey_geometry
        self._uid = uuid.uuid4()

        # Cache for electrode locations
        self._locations_a = None
        self._locations_b = None
        self._locations_m = None
        self._locations_n = None

        # Source ordering for data organization
        self._source_order = None

    @property
    def source_list(self) -> List[BaseSrc]:
        """List of sources in the survey"""
        return self._source_list

    @source_list.setter
    def source_list(self, sources: List[BaseSrc]):
        if not isinstance(sources, list):
            raise TypeError("source_list must be a list")
        for i, src in enumerate(sources):
            if not isinstance(src, BaseSrc):
                raise TypeError(f"source_list[{i}] must be a BaseSrc instance")
        self._source_list = sources
        # Reset caches when source list changes
        self._locations_a = None
        self._locations_b = None
        self._locations_m = None
        self._locations_n = None
        self._source_order = None

    @property
    def survey_geometry(self) -> str:
        """Survey geometry type"""
        return self._survey_geometry

    @survey_geometry.setter
    def survey_geometry(self, geom: str):
        if geom not in ["surface", "borehole", "general"]:
            raise ValueError(
                "survey_geometry must be 'surface', 'borehole', or 'general'"
            )
        self._survey_geometry = geom

    @property
    def uid(self):
        """Universal unique identifier"""
        return self._uid

    @property
    def nSrc(self) -> int:
        """Number of sources in the survey"""
        return len(self.source_list)

    @property
    def nD(self) -> int:
        """Total number of data points in the survey"""
        return sum(rx.nD for src in self.source_list for rx in src.receiver_list)

    def __repr__(self):
        return f"{self.__class__.__name__}(#sources: {self.nSrc}; #data: {self.nD})"

    def _set_abmn_locations(self):
        """Extract and organize A, B, M, N electrode locations from sources and receivers"""
        locations_a = []
        locations_b = []
        locations_m = []
        locations_n = []

        for source in self.source_list:
            for rx in source.receiver_list:
                nRx = rx.nD

                # Handle source electrode locations
                if isinstance(source, SrcPole):
                    # Pole source: A electrode at source location, B at infinity (NaN)
                    loc_a = source.location_a.unsqueeze(0).repeat(nRx, 1)
                    loc_b = torch.full_like(loc_a, float("nan"))
                    locations_a.append(loc_a)
                    locations_b.append(loc_b)

                elif isinstance(source, SrcDipole):
                    # Dipole source: A and B at specified locations
                    loc_a = source.location_a.unsqueeze(0).repeat(nRx, 1)
                    loc_b = source.location_b.unsqueeze(0).repeat(nRx, 1)
                    locations_a.append(loc_a)
                    locations_b.append(loc_b)

                elif isinstance(source, Multipole):
                    # Multipole source: multiple A electrodes, B at infinity
                    loc_tiled = source.location.repeat(nRx, 1)
                    locations_a.append(loc_tiled)
                    locations_b.append(torch.full_like(loc_tiled, float("nan")))

                # Handle receiver electrode locations
                if isinstance(rx, RxPole):
                    # Pole receiver: M electrode at receiver location, N at infinity
                    locations_m.append(rx.locations)
                    locations_n.append(torch.full_like(rx.locations, float("nan")))

                elif isinstance(rx, RxDipole):
                    # Dipole receiver: M and N at specified locations
                    locations_m.append(rx.locations_m)
                    locations_n.append(rx.locations_n)

        # Stack all locations
        self._locations_a = torch.vstack(locations_a)
        self._locations_b = torch.vstack(locations_b)
        self._locations_m = torch.vstack(locations_m)
        self._locations_n = torch.vstack(locations_n)

    @property
    def locations_a(self) -> torch.Tensor:
        """Locations of A (positive current) electrodes

        Returns
        -------
        torch.Tensor
            A electrode locations, shape (nD, dim)
        """
        if self._locations_a is None:
            self._set_abmn_locations()
        return self._locations_a

    @property
    def locations_b(self) -> torch.Tensor:
        """Locations of B (negative current) electrodes

        Returns
        -------
        torch.Tensor
            B electrode locations, shape (nD, dim)
        """
        if self._locations_b is None:
            self._set_abmn_locations()
        return self._locations_b

    @property
    def locations_m(self) -> torch.Tensor:
        """Locations of M (positive potential) electrodes

        Returns
        -------
        torch.Tensor
            M electrode locations, shape (nD, dim)
        """
        if self._locations_m is None:
            self._set_abmn_locations()
        return self._locations_m

    @property
    def locations_n(self) -> torch.Tensor:
        """Locations of N (negative potential) electrodes

        Returns
        -------
        torch.Tensor
            N electrode locations, shape (nD, dim)
        """
        if self._locations_n is None:
            self._set_abmn_locations()
        return self._locations_n

    @property
    def unique_electrode_locations(self) -> torch.Tensor:
        """Unique electrode locations used in the survey

        Returns
        -------
        torch.Tensor
            Unique electrode locations, shape (n_unique, dim)
        """
        # Stack all electrode locations
        all_locs = torch.vstack(
            [self.locations_a, self.locations_b, self.locations_m, self.locations_n]
        )

        # Remove NaN locations (pole configurations)
        finite_mask = torch.isfinite(all_locs).all(dim=1)
        finite_locs = all_locs[finite_mask]

        # Find unique locations
        unique_locs = torch.unique(finite_locs, dim=0)
        return unique_locs

    @property
    def source_locations(self) -> List[torch.Tensor]:
        """Source electrode locations organized by type

        Returns
        -------
        list of torch.Tensor
            [A_locations, B_locations] where each is a tensor of all source locations
        """
        src_a = []
        src_b = []

        for src in self.source_list:
            src_a.append(src.location_a)
            src_b.append(src.location_b)

        return [torch.vstack(src_a), torch.vstack(src_b)]

    def get_all_slices(self) -> Dict[Tuple[BaseSrc, BaseRx], slice]:
        """Get data slices for each source-receiver pair

        Returns
        -------
        dict
            Mapping from (source, receiver) to data slice
        """
        slices = {}
        data_idx = 0

        for src in self.source_list:
            for rx in src.receiver_list:
                slices[(src, rx)] = slice(data_idx, data_idx + rx.nD)
                data_idx += rx.nD

        return slices

    def set_geometric_factor(self, space_type: str = "halfspace") -> torch.Tensor:
        """Set geometric factors for apparent resistivity calculations

        Parameters
        ----------
        space_type : str, default: "halfspace"
            Space type for geometric factor calculation: "halfspace" or "wholespace"

        Returns
        -------
        torch.Tensor
            Geometric factors for each data point
        """
        if space_type not in ["halfspace", "wholespace"]:
            raise ValueError("space_type must be 'halfspace' or 'wholespace'")

        # Calculate geometric factors
        geometric_factors = self._calculate_geometric_factor(space_type)

        # Set geometric factors for each receiver
        survey_slices = self.get_all_slices()
        for source in self.source_list:
            for rx in source.receiver_list:
                if rx.data_type == "apparent_resistivity":
                    src_rx_slice = survey_slices[source, rx]
                    rx._geometric_factor[source.uid] = geometric_factors[src_rx_slice]

        return geometric_factors

    def _calculate_geometric_factor(self, space_type: str) -> torch.Tensor:
        """Calculate geometric factors for the survey

        Parameters
        ----------
        space_type : str
            "halfspace" or "wholespace"

        Returns
        -------
        torch.Tensor
            Geometric factors, shape (nD,)
        """
        # Get electrode locations
        loc_a = self.locations_a
        loc_b = self.locations_b
        loc_m = self.locations_m
        loc_n = self.locations_n

        # Calculate distances
        r_am = torch.norm(loc_a - loc_m, dim=1)
        r_bm = torch.norm(loc_b - loc_m, dim=1)
        r_an = torch.norm(loc_a - loc_n, dim=1)
        r_bn = torch.norm(loc_b - loc_n, dim=1)

        # Handle infinite electrodes (poles) by setting large distance
        inf_mask_b = torch.isnan(loc_b).any(dim=1)
        inf_mask_n = torch.isnan(loc_n).any(dim=1)

        r_bm[inf_mask_b] = 1e10  # Very large distance for infinite electrode
        r_bn[inf_mask_b] = 1e10
        r_an[inf_mask_n] = 1e10
        r_bn[inf_mask_n] = 1e10

        # Calculate geometric factor
        # G = (1/C) * [1/r_AM - 1/r_BM - 1/r_AN + 1/r_BN]
        # where C = 2π for halfspace, 4π for wholespace
        C = 2 * torch.pi if space_type == "halfspace" else 4 * torch.pi

        G = (1.0 / C) * (1.0 / r_am - 1.0 / r_bm - 1.0 / r_an + 1.0 / r_bn)

        return G

    def simulate_data(self, simulation, model: torch.Tensor) -> torch.Tensor:
        """Simulate data for the survey using the given model

        Parameters
        ----------
        simulation : DCStaticSimulationCellCentered
            The simulation object
        model : torch.Tensor
            Model parameters (resistivity)

        Returns
        -------
        torch.Tensor
            Simulated data vector
        """
        # Update simulation's sources and receivers
        simulation.sources = self.source_list

        # Compute predicted data
        return simulation.dpred(model)

    def get_source_indices(self, sources: Union[BaseSrc, List[BaseSrc]]) -> List[int]:
        """Get indices for specified sources

        Parameters
        ----------
        sources : BaseSrc or list of BaseSrc
            Source(s) to get indices for

        Returns
        -------
        list of int
            Indices of sources in the source list
        """
        if not isinstance(sources, list):
            sources = [sources]

        indices = []
        for src in sources:
            try:
                idx = next(
                    i for i, s in enumerate(self.source_list) if s.uid == src.uid
                )
                indices.append(idx)
            except StopIteration:
                raise KeyError(f"Source {src} not found in survey")

        return indices

    def get_source_tensor(self, simulation) -> torch.Tensor:
        """Create batched source tensor for all sources in survey

        Parameters
        ----------
        simulation : DCStaticSimulation
            The DC resistivity simulation object

        Returns
        -------
        torch.Tensor
            Batched RHS tensor with shape (n_mesh_points, n_sources)
        """
        if not self.source_list:
            raise ValueError("No sources defined in survey")

        rhs_vectors = []
        for src in self.source_list:
            rhs = src.evaluate(simulation)
            rhs_vectors.append(rhs)

        # Stack along source dimension (dim=0)
        return torch.stack(rhs_vectors, dim=0)

    def get_receiver_tensor(self, mesh, projected_grid: str = "N") -> torch.Tensor:
        """Create batched receiver projection tensor for all receivers

        Parameters
        ----------
        mesh : TensorMesh
            The computational mesh
        projected_grid : str, default: "N"
            Grid locations to project from ("N" for nodes, "CC" for cell centers)

        Returns
        -------
        torch.Tensor
            Batched projection tensor with shape (n_data_total, n_mesh_points)
            Each row corresponds to one data point's projection from mesh to receiver
        """
        if not self.source_list:
            raise ValueError("No sources defined in survey")

        projection_matrices = []

        for src in self.source_list:
            reciever_tensor = src.build_receiver_tensor(mesh, projected_grid)
            projection_matrices.append(reciever_tensor)

        return projection_matrices

    def get_receiver_tensor_sparse_list(self, mesh, projected_grid: str = "N"):
        """Get receiver projection matrices as a sparse list (most memory efficient)

        This method returns the receiver projection matrices as a list of sparse
        tensors without any stacking or padding. This is the most memory-efficient
        approach for batched operations.

        Parameters
        ----------
        mesh : TensorMesh
            The computational mesh
        projected_grid : str, default: "N"
            Grid locations to project from

        Returns
        -------
        list of torch.sparse.Tensor
            List of sparse projection matrices, one for each receiver
        """
        if not self.source_list:
            raise ValueError("No sources defined in survey")

        sparse_matrices = []
        for src in self.source_list:
            for rx in src.receiver_list:
                P = rx.getP(mesh, projected_grid)
                sparse_matrices.append(P)

        return sparse_matrices

    def get_mesh_size_for_grid(self, mesh, projected_grid: str) -> int:
        """Get the number of mesh points for a given projection grid

        Parameters
        ----------
        mesh : TensorMesh
            The computational mesh
        projected_grid : str
            Grid type ("N", "CC", "Ex", "Ey", "Ez", "Fx", "Fy", "Fz")

        Returns
        -------
        int
            Number of mesh points for the specified grid
        """
        grid_map = {
            "N": mesh.nN,
            "nodes": mesh.nN,
            "CC": mesh.nC,
            "cell_centers": mesh.nC,
        }

        if projected_grid in grid_map:
            return grid_map[projected_grid]
        elif projected_grid.startswith("E"):  # Edge grids
            if projected_grid in ["Ex", "edges_x"]:
                return mesh.nEx
            elif projected_grid in ["Ey", "edges_y"]:
                return mesh.nEy
            elif projected_grid in ["Ez", "edges_z"]:
                return mesh.nEz
            else:
                return mesh.nE  # Total edges
        elif projected_grid.startswith("F"):  # Face grids
            if projected_grid in ["Fx", "faces_x"]:
                return mesh.nFx
            elif projected_grid in ["Fy", "faces_y"]:
                return mesh.nFy
            elif projected_grid in ["Fz", "faces_z"]:
                return mesh.nFz
            else:
                return mesh.nF  # Total faces
        else:
            # Default fallback
            return mesh.nC
