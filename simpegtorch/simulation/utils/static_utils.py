"""
DC resistivity utility functions for simpeg-torch.

This module provides utility functions for creating DC/IP surveys, computing
pseudo-locations, apparent resistivity calculations, and plotting.
"""

import torch
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from typing import List, Union, Optional, Tuple

from ..resistivity.sources import Pole as SrcPole, Dipole as SrcDipole
from ..resistivity.receivers import Pole as RxPole, Dipole as RxDipole
from ..resistivity.survey import Survey


def generate_dcip_sources_line(
    survey_type: str,
    data_type: str,
    dimension_type: str,
    end_points: Union[torch.Tensor, np.ndarray],
    topo: Union[torch.Tensor, np.ndarray, float],
    num_rx_per_src: int,
    station_spacing: float,
) -> List:
    """
    Generate the source list for a 2D or 3D DC/IP survey line.

    This utility will create the list of DC/IP source objects for a single line of
    2D or 3D data. The topography, orientation, spacing and number of receivers
    can be specified by the user. This function can be used to define multiple lines
    of DC/IP, which can be appended to create the sources for an entire survey.

    Parameters
    ----------
    survey_type : {'dipole-dipole', 'pole-dipole', 'dipole-pole', 'pole-pole'}
        Survey type.
    data_type : {'volt', 'apparent_conductivity', 'apparent_resistivity', 'apparent_chargeability'}
        Data type.
    dimension_type : {'2D', '3D'}
        Which dimension you are using.
    end_points : torch.Tensor or numpy.array
        Horizontal end points [x1, x2] or [x1, x2, y1, y2]
    topo : torch.Tensor, numpy.ndarray, or float
        Define survey topography
    num_rx_per_src : int
        Maximum number of receivers per source
    station_spacing : float
        Distance between stations

    Returns
    -------
    list
        List of DC source objects
    """

    assert survey_type.lower() in [
        "pole-pole",
        "pole-dipole",
        "dipole-pole",
        "dipole-dipole",
    ], "survey_type must be one of 'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'"

    assert data_type.lower() in [
        "volt",
        "apparent_conductivity",
        "apparent_resistivity",
        "apparent_chargeability",
    ], "data_type must be one of 'volt', 'apparent_conductivity', 'apparent_resistivity', 'apparent_chargeability'"

    assert dimension_type.upper() in [
        "2D",
        "2.5D",
        "3D",
    ], "dimension_type must be one of '2D' or '3D'"

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0))
        return r

    # Convert to numpy for compatibility with interpolation functions
    if isinstance(end_points, torch.Tensor):
        end_points = end_points.detach().cpu().numpy()
    if isinstance(topo, torch.Tensor):
        topo = topo.detach().cpu().numpy()

    # Compute horizontal locations of sources and receivers
    x1 = end_points[0]
    x2 = end_points[1]

    if dimension_type == "3D":
        # Station locations
        y1 = end_points[2]
        y2 = end_points[3]
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / station_spacing) + 1)
        dl_x = (x2 - x1) / L
        dl_y = (y2 - y1) / L
        stn_x = x1 + np.array(range(int(nstn))) * dl_x * station_spacing
        stn_y = y1 + np.array(range(int(nstn))) * dl_y * station_spacing

        # Station xyz locations
        P = np.c_[stn_x, stn_y]
        if np.size(topo) == 1:
            P = np.c_[P, topo * np.ones((nstn))]
        else:
            fun_interp = LinearNDInterpolator(topo[:, 0:2], topo[:, -1])
            P = np.c_[P, fun_interp(P)]

    else:
        # Station locations
        y1 = 0.0
        y2 = 0.0
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / station_spacing) + 1)
        stn_x = x1 + np.array(range(int(nstn))) * station_spacing

        # Station xyz locations
        if np.size(topo) == 1:
            P = np.c_[stn_x, topo * np.ones((nstn))]
        else:
            fun_interp = interp1d(topo[:, 0], topo[:, -1])
            P = np.c_[stn_x, fun_interp(stn_x)]

    # Convert back to PyTorch tensors
    P = torch.from_numpy(P).float()

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    source_list = []

    if survey_type.lower() == "pole-pole":
        rx_shift = 0
    elif survey_type.lower() in ["pole-dipole", "dipole-pole"]:
        rx_shift = 1
    elif survey_type.lower() == "dipole-dipole":
        rx_shift = 2

    for ii in range(0, int(nstn - rx_shift)):
        if dimension_type == "3D":
            D = xy_2_r(stn_x[ii + rx_shift], x2, stn_y[ii + rx_shift], y2)
        else:
            D = xy_2_r(stn_x[ii + rx_shift], x2, y1, y2)

        # Number of receivers to fit
        nrec = int(np.min([np.floor(D / station_spacing), num_rx_per_src]))

        # Check if there is enough space, else break the loop
        if nrec <= 0:
            continue

        # Create receivers
        if survey_type.lower() in ["dipole-pole", "pole-pole"]:
            rxClass = RxPole(
                P[ii + rx_shift + 1 : ii + rx_shift + nrec + 1, :], data_type=data_type
            )
        elif survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
            rxClass = RxDipole(
                P[ii + rx_shift : ii + rx_shift + nrec, :],
                P[ii + rx_shift + 1 : ii + rx_shift + nrec + 1, :],
                data_type=data_type,
            )

        # Create sources
        if survey_type.lower() in ["pole-dipole", "pole-pole"]:
            srcClass = SrcPole([rxClass], P[ii, :])
        elif survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
            srcClass = SrcDipole([rxClass], P[ii, :], P[ii + 1, :])

        source_list.append(srcClass)

    return source_list


def pseudo_locations(survey: Survey, wenner_tolerance: float = 0.1) -> torch.Tensor:
    """
    Calculates the pseudo-sensitivity locations for 2D and 3D surveys.

    Parameters
    ----------
    survey : Survey
        A DC or IP survey
    wenner_tolerance : float, default=0.1
        If the center location for a source and receiver pair are within wenner_tolerance,
        we assume the datum was collected with a wenner configuration and the pseudo-location
        is computed based on the AB electrode spacing.

    Returns
    -------
    torch.Tensor
        Pseudo-location tensor of shape (n_data, dim)
        For 2D surveys, returns (n_data, 2) with along-line position and pseudo-depth.
        For 3D surveys, returns (n_data, 3) with (x, y, pseudo_depth).
    """

    if not isinstance(survey, Survey):
        raise TypeError(f"Input must be instance of {Survey}, not {type(survey)}")

    # Pre-allocate
    midpoints = []
    ds = []

    for source in survey.source_list:
        # Get source location array - this matches original SimPEG structure
        # where source.location contains all electrode locations
        src_loc = source.location
        src_midpoint = torch.mean(src_loc, dim=0).unsqueeze(0)

        for receiver in source.receiver_list:
            # Handle receiver locations - check if it's a dipole (list) or pole (tensor)
            rx_locs = receiver.locations
            if isinstance(rx_locs, list):
                # Dipole receiver: locations is [locations_m, locations_n]
                rx_midpoints = (rx_locs[0] + rx_locs[1]) / 2
            else:
                # Pole receiver: locations is just a tensor
                rx_midpoints = rx_locs

            n_loc = rx_midpoints.shape[0]

            # Midpoint locations
            midpoints.append((src_midpoint.repeat(n_loc, 1) + rx_midpoints) / 2)

            # Vector path from source midpoint to receiver midpoints
            ds.append(rx_midpoints - src_midpoint.repeat(n_loc, 1))

    midpoints = torch.vstack(midpoints)
    ds = torch.vstack(ds)
    pseudo_depth = torch.zeros_like(midpoints)

    # wenner-like electrode groups (are source and rx midpoints in same place)
    is_wenner = torch.sqrt(torch.sum(ds[:, :-1] ** 2, dim=1)) < wenner_tolerance

    # Pseudo depth is AB/2 for Wenner-like configurations
    if torch.any(is_wenner):
        temp = torch.abs(electrode_separations(survey, ["AB"])["AB"]) / 2
        pseudo_depth[is_wenner, -1] = temp[is_wenner]

    # Takes into account topography for other configurations
    if torch.any(~is_wenner):
        L = torch.sqrt(torch.sum(ds[~is_wenner, :] ** 2, dim=1)) / 2
        dz = ds[~is_wenner, -1]
        pseudo_depth[~is_wenner, 0] = (dz / 2) * (ds[~is_wenner, 0] / L)
        if ds.shape[1] > 2:
            pseudo_depth[~is_wenner, 1] = (dz / 2) * (ds[~is_wenner, 1] / L)
        pseudo_depth[~is_wenner, -1] = (
            torch.sqrt(torch.sum(ds[~is_wenner, :-1] ** 2, dim=1)) / 2
        )

    return midpoints - pseudo_depth


def electrode_separations(
    survey: Survey, electrode_pair: Union[str, List[str]] = "all"
) -> dict:
    """
    Calculate horizontal separation between specific or all electrodes.

    Parameters
    ----------
    survey : Survey
        A DC or IP survey object
    electrode_pair : {'all', 'AB', 'MN', 'AM', 'AN', 'BM', 'BN'} or list
        Which electrode separation pairs to compute.

    Returns
    -------
    dict
        Dictionary containing electrode separations for each requested pair.
    """

    if not isinstance(electrode_pair, list):
        if electrode_pair.lower() == "all":
            electrode_pair = ["AB", "MN", "AM", "AN", "BM", "BN"]
        elif isinstance(electrode_pair, str):
            electrode_pair = [electrode_pair.upper()]
        else:
            raise TypeError(
                "electrode_pair must be either a string, list of strings, or an "
                "ndarray containing the electrode separation distances you would "
                f"like to calculate not {type(electrode_pair)}"
            )

    elecSepDict = {}
    AB = []
    MN = []
    AM = []
    AN = []
    BM = []
    BN = []

    for src in survey.source_list:
        # pole or dipole source
        if isinstance(src, SrcDipole):
            a_loc = src.location_a
            b_loc = src.location_b
        elif isinstance(src, SrcPole):
            a_loc = src.location_a
            b_loc = torch.full_like(src.location_a, float("inf"))
        else:
            raise NotImplementedError("A_B locations undefined for multipole sources.")

        for rx in src.receiver_list:
            # pole or dipole receiver
            if isinstance(rx, RxDipole):
                M = rx.locations_m
                N = rx.locations_n
            else:
                M = rx.locations
                N = torch.full_like(rx.locations, float("-inf"))

            n_rx = M.shape[0]

            A = a_loc.unsqueeze(0).repeat(n_rx, 1)
            B = b_loc.unsqueeze(0).repeat(n_rx, 1)

            # Compute distances
            AB.append(torch.norm(A - B, dim=1))
            MN.append(torch.norm(M - N, dim=1))
            AM.append(torch.norm(A - M, dim=1))
            AN.append(torch.norm(A - N, dim=1))
            BM.append(torch.norm(B - M, dim=1))
            BN.append(torch.norm(B - N, dim=1))

    # Stack to vector and define in dictionary
    if "AB" in electrode_pair:
        if AB:
            AB = torch.hstack(AB)
        elecSepDict["AB"] = AB
    if "MN" in electrode_pair:
        if MN:
            MN = torch.hstack(MN)
        elecSepDict["MN"] = MN
    if "AM" in electrode_pair:
        if AM:
            AM = torch.hstack(AM)
        elecSepDict["AM"] = AM
    if "AN" in electrode_pair:
        if AN:
            AN = torch.hstack(AN)
        elecSepDict["AN"] = AN
    if "BM" in electrode_pair:
        if BM:
            BM = torch.hstack(BM)
        elecSepDict["BM"] = BM
    if "BN" in electrode_pair:
        if BN:
            BN = torch.hstack(BN)
        elecSepDict["BN"] = BN

    return elecSepDict


def apparent_resistivity_from_voltage(
    survey: Survey,
    volts: torch.Tensor,
    space_type: str = "half space",
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Calculate apparent resistivities from normalized voltages.

    Parameters
    ----------
    survey : Survey
        A DC survey
    volts : torch.Tensor
        Normalized voltage measurements [V/A]
    space_type : {'half space', 'whole space'}
        Compute apparent resistivity assume a half space or whole space.
    eps : float, default=1e-10
        Stabilization constant in case of a null geometric factor

    Returns
    -------
    torch.Tensor
        Apparent resistivities for all data
    """

    G = geometric_factor(survey, space_type=space_type)

    # Calculate apparent resistivity
    # absolute value is required because of the regularizer
    rhoApp = torch.abs(volts * (1.0 / (G + eps)))

    return rhoApp


def geometric_factor(survey: Survey, space_type: str = "half space") -> torch.Tensor:
    """
    Calculate geometric factor for every datum.

    Consider you have current electrodes A and B, and potential electrodes M and N.
    Let R_AM represents the scalar horizontal distance between electrodes A
    and M; likewise for R_BM, R_AN and R_BN.
    The geometric factor is given by:

    G = (1/C) * [1/R_AM - 1/R_BM - 1/R_AN + 1/R_BN]

    where C=2À for a halfspace and C=4À for a wholespace.

    Parameters
    ----------
    survey : Survey
        A DC (or IP) survey object
    space_type : {'half space', 'whole space'}
        Compute geometric factor for a halfspace or wholespace.

    Returns
    -------
    torch.Tensor
        Geometric factor for each datum
    """
    SPACE_TYPES = {
        "half space": ["half space", "half-space", "half_space", "halfspace", "half"],
        "whole space": [
            "whole space",
            "whole-space",
            "whole_space",
            "wholespace",
            "whole",
        ],
    }

    # Set factor for whole-space or half-space assumption
    if space_type.lower() in SPACE_TYPES["whole space"]:
        spaceFact = 4.0
    elif space_type.lower() in SPACE_TYPES["half space"]:
        spaceFact = 2.0
    else:
        raise TypeError("'space_type must be 'whole space' | 'half space'")

    elecSepDict = electrode_separations(survey, electrode_pair=["AM", "BM", "AN", "BN"])
    AM = elecSepDict["AM"]
    BM = elecSepDict["BM"]
    AN = elecSepDict["AN"]
    BN = elecSepDict["BN"]

    # Determine geometric factor G based on electrode separation distances.
    # For case where source and/or receivers are pole, terms will be
    # divided by infinity.
    G = 1 / AM - 1 / BM - 1 / AN + 1 / BN

    return G / (spaceFact * torch.pi)


def convert_survey_3d_to_2d_lines(
    survey: Survey,
    lineID: torch.Tensor,
    data_type: str = "volt",
    output_indexing: bool = False,
) -> Union[List[Survey], Tuple[List[Survey], List[torch.Tensor]]]:
    """
    Convert a 3D survey into a list of local 2D surveys.

    Here, the user provides a Survey whose geometry is defined
    for use in a 3D simulation and a 1D tensor which defines the
    line ID for each datum. The function returns a list of local
    2D survey objects. The change of coordinates for electrodes is
    [x, y, z] to [s, z], where s is the distance along the profile
    line. For each line, s = 0 defines the A-electrode location
    for the first source in the source list.

    Parameters
    ----------
    survey : Survey
        A DC (or IP) survey
    lineID : torch.Tensor
        Defines the corresponding line ID for each datum
    data_type : {'volt', 'apparent_resistivity', 'apparent_conductivity', 'apparent_chargeability'}
        Data type for the survey.
    output_indexing : bool, default=False
        If True output a list of indexing arrays that map from the original 3D
        data to each 2D survey line.

    Returns
    -------
    survey_list : list of Survey
        A list of 2D survey objects
    out_indices_list : list of torch.Tensor, optional
        A list of indexing arrays that map from the original 3D data to each 2D
        survey line. Will be returned only if output_indexing is set to True.
    """
    # Check if the survey is 3D
    if (ndims := survey.locations_a.shape[1]) != 3:
        raise ValueError(f"Invalid {ndims}D 'survey'. It should be a 3D survey.")

    # Checks on the passed lineID array
    if isinstance(lineID, np.ndarray):
        lineID = torch.from_numpy(lineID)
    if lineID.dim() != 1:
        raise ValueError(
            f"Invalid 'lineID' array with '{lineID.dim()}' dimensions. "
            "It should be a 1D array."
        )
    if (size := lineID.numel()) != survey.nD:
        raise ValueError(
            f"Invalid 'lineID' array with '{size}' elements. "
            "It should have the same number of elements as data "
            f"in the survey ('{survey.nD}')."
        )

    # Find all unique line id
    unique_lineID = torch.unique(lineID)

    # If you output indexing to keep track of possible sorting
    k = torch.arange(0, survey.nD)
    out_indices_list = []

    ab_locs_all = torch.cat([survey.locations_a, survey.locations_b], dim=1)
    mn_locs_all = torch.cat([survey.locations_m, survey.locations_n], dim=1)

    # For each unique lineID
    survey_list = []
    for ID in unique_lineID:
        source_list = []

        # Source locations for this line
        lineID_index = torch.where(lineID == ID)[0]
        ab_locs, ab_index = torch.unique(
            ab_locs_all[lineID_index, :], dim=0, return_inverse=True
        )
        ab_index = torch.unique(ab_index, sorted=True)

        # Find s=0 location and heading for line
        start_index = lineID_index[ab_index]
        out_indices = []
        kID = k[lineID_index]  # data indices part of this line
        r0 = ab_locs_all[start_index[0], 0:2]  # (x, y) for the survey line
        rN = ab_locs_all[start_index[-1], 0:2]  # (x, y) for last electrode
        uvec = (rN - r0) / torch.sqrt(
            torch.sum((rN - r0) ** 2)
        )  # unit vector for line orientation

        # Along line positions and elevation for electrodes on current line
        a_locs_s = torch.stack(
            [
                torch.dot(ab_locs_all[lineID_index, 0:2] - r0, uvec),
                ab_locs_all[lineID_index, 2],
            ],
            dim=1,
        )
        b_locs_s = torch.stack(
            [
                torch.dot(ab_locs_all[lineID_index, 3:5] - r0, uvec),
                ab_locs_all[lineID_index, -1],
            ],
            dim=1,
        )
        m_locs_s = torch.stack(
            [
                torch.dot(mn_locs_all[lineID_index, 0:2] - r0, uvec),
                mn_locs_all[lineID_index, 2],
            ],
            dim=1,
        )
        n_locs_s = torch.stack(
            [
                torch.dot(mn_locs_all[lineID_index, 3:5] - r0, uvec),
                mn_locs_all[lineID_index, -1],
            ],
            dim=1,
        )

        # For each source in the line
        for ind in ab_index:
            # Get source location
            src_loc_a = a_locs_s[ind, :]
            src_loc_b = b_locs_s[ind, :]

            # Get receiver locations
            rx_index = torch.where(
                (torch.abs(a_locs_s[:, 0] - src_loc_a[0]) < 1e-3)
                & (torch.abs(b_locs_s[:, 0] - src_loc_b[0]) < 1e-3)
            )[0]
            rx_loc_m = m_locs_s[rx_index, :]
            rx_loc_n = n_locs_s[rx_index, :]

            # Extract pole and dipole receivers
            k_ii = kID[rx_index]
            is_pole_rx = torch.all(torch.abs(rx_loc_m - rx_loc_n) < 1e-3, dim=1)
            rx_list = []

            if torch.any(is_pole_rx):
                rx_list += [RxPole(rx_loc_m[is_pole_rx, :], data_type=data_type)]
                out_indices.append(k_ii[is_pole_rx])

            if torch.any(~is_pole_rx):
                rx_list += [
                    RxDipole(
                        rx_loc_m[~is_pole_rx, :],
                        rx_loc_n[~is_pole_rx, :],
                        data_type=data_type,
                    )
                ]
                out_indices.append(k_ii[~is_pole_rx])

            # Define Pole or Dipole Sources
            if torch.all(torch.abs(src_loc_a - src_loc_b) < 1e-3):
                source_list.append(SrcPole(rx_list, src_loc_a))
            else:
                source_list.append(SrcDipole(rx_list, src_loc_a, src_loc_b))

        # Create a 2D survey and add to list
        survey_list.append(Survey(source_list))
        if output_indexing:
            out_indices_list.append(torch.hstack(out_indices))

    if output_indexing:
        return survey_list, out_indices_list
    else:
        return survey_list


def plot_pseudosection(
    survey: Survey,
    dobs: Optional[torch.Tensor] = None,
    plot_type: str = "contourf",
    ax=None,
    clim: Optional[List[float]] = None,
    scale: str = "linear",
    pcolor_opts: Optional[dict] = None,
    contourf_opts: Optional[dict] = None,
    scatter_opts: Optional[dict] = None,
    mask_topography: bool = False,
    create_colorbar: bool = True,
    cbar_opts: Optional[dict] = None,
    cbar_label: str = "",
    cax=None,
    data_locations: bool = False,
    data_type: Optional[str] = None,
    space_type: str = "half space",
    **kwargs,
):
    r"""
    Plot 2D DC/IP data in pseudo-section.

    This utility allows the user to image 2D DC/IP data in pseudosection as
    either a scatter plot or as a filled contour plot.

    Parameters
    ----------
    survey : Survey
        A DC or IP survey object defining a 2D survey line
    dobs : torch.Tensor, optional
        A data vector containing volts, integrated chargeabilities, apparent
        resistivities, apparent chargeabilities or data misfits.
    plot_type : {"contourf", "scatter", "pcolor"}
        Which plot type to create.
    ax : matplotlib axis, optional
        An axis for the plot
    clim : list of float, optional
        list containing the minimum and maximum value for the color range,
        i.e. [vmin, vmax]
    scale : {'linear', 'log'}
        Plot on linear or log base 10 scale.
    pcolor_opts : dict, optional
        Dictionary defining kwargs for pcolor plot if `plot_type=='pcolor'`
    contourf_opts : dict, optional
        Dictionary defining kwargs for filled contour plot if `plot_type=='contourf'`
    scatter_opts : dict, optional
        Dictionary defining kwargs for scatter plot if `plot_type=='scatter'`
    mask_topography : bool
        This feature should be set to True when there is significant topography and the user
        would like to mask interpolated locations in the filled contour plot that lie
        above the surface topography.
    create_colorbar : bool
        If True, a colorbar is automatically generated. If False, it is not.
    cbar_opts : dict
        Dictionary defining kwargs for the colorbar
    cbar_label : str
        A string stating the color bar label for the
        data; e.g. 'S/m', '$\\Omega m$', '%'
    cax : matplotlib axis, optional
        An axis object for the colorbar
    data_locations : bool
        Whether to plot data locations
    data_type : str, optional
        If dobs is None, this will transform the data vector from the survey
        from voltage to the requested data_type.
    space_type : {'half space', "whole space"}
        Space type to used for the transformation from voltage to data_type
        if dobs is None.

    Returns
    -------
    matplotlib axis
        The axis object that holds the plot
    """
    # Import matplotlib here to avoid dependency issues
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import ticker
    except ImportError:
        raise ImportError("matplotlib and scipy are required for plotting")

    if plot_type.lower() not in ["pcolor", "contourf", "scatter"]:
        raise ValueError(
            "plot_type must be 'pcolor', 'contourf', or 'scatter'. The input value of "
            f"{plot_type} is not recognized"
        )

    # Handle data transformation if needed
    if dobs is None:
        raise ValueError("dobs must be provided")

    # Convert torch tensor to numpy for plotting
    if isinstance(dobs, torch.Tensor):
        dobs = dobs.detach().cpu().numpy()

    # Transform voltage data to requested type if needed
    DATA_TYPES = {
        "apparent resistivity": [
            "apparent resistivity",
            "appresistivity",
            "apparentresistivity",
            "apparent-resistivity",
            "apparent_resistivity",
            "appres",
        ],
        "apparent conductivity": [
            "apparent conductivity",
            "appconductivity",
            "apparentconductivity",
            "apparent-conductivity",
            "apparent_conductivity",
            "appcon",
        ],
    }

    if data_type is not None:
        if data_type in (
            DATA_TYPES["apparent conductivity"] + DATA_TYPES["apparent resistivity"]
        ):
            dobs_tensor = (
                torch.from_numpy(dobs) if not isinstance(dobs, torch.Tensor) else dobs
            )
            dobs_tensor = apparent_resistivity_from_voltage(
                survey, dobs_tensor, space_type=space_type
            )
            dobs = dobs_tensor.detach().cpu().numpy()
        if data_type in DATA_TYPES["apparent conductivity"]:
            dobs = 1.0 / dobs

    # Get plotting locations from survey geometry
    try:
        locations = pseudo_locations(survey)
        if isinstance(locations, torch.Tensor):
            locations = locations.detach().cpu().numpy()
    except Exception:
        raise TypeError("The survey must be a resistivity.Survey object.")

    # Create an axis for the pseudosection if None
    if ax is None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])

    if clim is None:
        vmin = vmax = None
    else:
        vmin, vmax = clim
    # Create default norms
    if scale == "log":
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x, z = locations[:, 0], locations[:, -1]

    # Scatter plot
    if plot_type == "scatter":
        # grab a shallow copy
        if scatter_opts is None:
            s_opts = {}
        else:
            s_opts = scatter_opts.copy()
        s = s_opts.pop("s", 40)
        norm = s_opts.pop("norm", norm)
        if isinstance(norm, mpl.colors.LogNorm):
            dobs = np.abs(dobs)

        data_plot = ax.scatter(x, z, s=s, c=dobs, norm=norm, **s_opts)
    # Filled contour plot
    elif plot_type == "contourf":
        if contourf_opts is None:
            opts = {}
        else:
            opts = contourf_opts.copy()
        norm = opts.pop("norm", norm)
        if isinstance(norm, mpl.colors.LogNorm):
            dobs = np.abs(dobs)
        if scale == "log":
            try:
                levels = opts.get("levels", "auto")
                locator = ticker.MaxNLocator(levels)
                levels = locator.tick_values(np.log10(dobs.min()), np.log10(dobs.max()))
                levels = 10**levels
                opts["levels"] = levels
            except TypeError:
                pass

        data_plot = ax.tricontourf(
            x,
            z,
            dobs,
            norm=norm,
            **opts,
        )
        if data_locations:
            ax.plot(x, z, "k.", ms=1, alpha=0.4)

    elif plot_type == "pcolor":
        if pcolor_opts is None:
            opts = {}
        else:
            opts = pcolor_opts.copy()
        norm = opts.pop("norm", norm)
        if isinstance(norm, mpl.colors.LogNorm):
            dobs = np.abs(dobs)

        data_plot = ax.tripcolor(x, z, dobs, shading="gouraud", norm=norm, **opts)
        if data_locations:
            ax.plot(x, z, "k.", ms=1, alpha=0.4)

    z_top = np.max(z)
    z_bot = np.min(z)
    ax.set_ylim(z_bot - 0.03 * (z_top - z_bot), z_top + 0.03 * (z_top - z_bot))
    ax.set_xlabel("Line position (m)")
    ax.set_ylabel("Pseudo-elevation (m)")

    # Define colorbar
    if cbar_opts is None:
        cbar_opts = {}
    if create_colorbar:
        cbar = plt.colorbar(
            data_plot,
            format="%.2e",
            fraction=0.06,
            orientation="vertical",
            cax=cax,
            ax=ax,
            **cbar_opts,
        )

        vmin = np.nanmin(dobs)
        vmax = np.nanmax(dobs)
        if scale == "log":
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 7)
        else:
            ticks = np.linspace(vmin, vmax, 7)
        cbar.set_ticks(ticks)
        cbar.ax.minorticks_off()
        cbar.set_label(cbar_label, labelpad=10)
        cbar.ax.tick_params()

    return ax, data_plot
