class Survey:
    """
    FDEM Survey class

    Manages sources and receivers for frequency domain electromagnetic surveys.

    Parameters
    ----------
    source_list : list
        List of FDEM source objects
    """

    def __init__(self, source_list):
        self.source_list = source_list if source_list is not None else []

    @property
    def frequencies(self):
        """
        Get unique frequencies from all sources.

        Returns
        -------
        list
            Sorted list of unique frequencies
        """
        freqs = [src.frequency for src in self.source_list]
        return sorted(list(set(freqs)))

    @property
    def nD(self):
        """
        Total number of data points across all sources and receivers.

        Returns
        -------
        int
            Total number of data points
        """
        total = 0
        for src in self.source_list:
            for rx in src.receiver_list:
                total += rx.nD
        return total

    @property
    def n_sources(self):
        """Number of sources in survey"""
        return len(self.source_list)

    @property
    def n_receivers(self):
        """Total number of receivers across all sources"""
        total = 0
        for src in self.source_list:
            total += len(src.receiver_list)
        return total

    def get_sources_by_frequency(self, freq):
        """
        Get all sources at a specific frequency.

        Parameters
        ----------
        freq : float
            Frequency in Hz

        Returns
        -------
        list
            List of sources at the specified frequency
        """
        return [src for src in self.source_list if src.frequency == freq]

    def get_data_indices(self):
        """
        Get data indices for each source and receiver.

        Returns
        -------
        dict
            Dictionary mapping (source_idx, receiver_idx) to data slice
        """
        indices = {}
        data_idx = 0

        for src_idx, src in enumerate(self.source_list):
            for rx_idx, rx in enumerate(src.receiver_list):
                indices[(src_idx, rx_idx)] = slice(data_idx, data_idx + rx.nD)
                data_idx += rx.nD

        return indices

    def set_geometric_factor(self, geometric_factor=None):
        """
        Set geometric factors for apparent resistivity calculations.

        Parameters
        ----------
        geometric_factor : array_like, optional
            Geometric factors for each data point. If None, no geometric
            factors are applied.
        """
        self.geometric_factor = geometric_factor

    def __len__(self):
        """Number of sources"""
        return len(self.source_list)

    def __iter__(self):
        """Iterate over sources"""
        return iter(self.source_list)

    def __getitem__(self, index):
        """Get source by index"""
        return self.source_list[index]


class FDEMSurvey(Survey):
    """
    Frequency Domain Electromagnetic Survey

    Specialized survey class for FDEM with additional functionality.
    """

    def __init__(self, source_list):
        super().__init__(source_list)

    def get_frequency_data_indices(self):
        """
        Get data indices organized by frequency.

        Returns
        -------
        dict
            Dictionary mapping frequency to data indices
        """
        freq_indices = {}

        for freq in self.frequencies:
            freq_indices[freq] = []

            data_idx = 0
            for src in self.source_list:
                if src.frequency == freq:
                    for rx in src.receiver_list:
                        freq_indices[freq].append(slice(data_idx, data_idx + rx.nD))
                        data_idx += rx.nD
                else:
                    # Skip sources not at this frequency
                    for rx in src.receiver_list:
                        data_idx += rx.nD

        return freq_indices

    def get_transmitter_receiver_pairs(self):
        """
        Get all transmitter-receiver pairs in the survey.

        Returns
        -------
        list
            List of (source, receiver) tuples
        """
        pairs = []
        for src in self.source_list:
            for rx in src.receiver_list:
                pairs.append((src, rx))
        return pairs

    def plot_survey_geometry(self):
        """
        Plot survey geometry (sources and receivers).

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot sources
        for i, src in enumerate(self.source_list):
            if src.location is not None:
                loc = src.location
                ax.scatter(
                    loc[0],
                    loc[1],
                    loc[2],
                    c="red",
                    s=100,
                    marker="^",
                    label="Sources" if i == 0 else "",
                )

        # Plot receivers
        all_rx_plotted = False
        for src in self.source_list:
            for rx in src.receiver_list:
                locs = rx.locations
                ax.scatter(
                    locs[:, 0],
                    locs[:, 1],
                    locs[:, 2],
                    c="blue",
                    s=50,
                    marker="o",
                    label="Receivers" if not all_rx_plotted else "",
                )
                all_rx_plotted = True

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("FDEM Survey Geometry")

        return fig

    def summary(self):
        """
        Print survey summary.
        """
        print("FDEM Survey Summary:")
        print(f"  Number of sources: {self.n_sources}")
        print(f"  Number of receivers: {self.n_receivers}")
        print(f"  Total data points: {self.nD}")
        print(f"  Frequencies: {self.frequencies}")

        # Summary by frequency
        for freq in self.frequencies:
            sources = self.get_sources_by_frequency(freq)
            n_data_freq = sum(sum(rx.nD for rx in src.receiver_list) for src in sources)
            print(
                f"    {freq:.1f} Hz: {len(sources)} sources, {n_data_freq} data points"
            )
