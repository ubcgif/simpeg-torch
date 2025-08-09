"""Module for ``matplotlib`` interaction with ``simpegtorch.discretize``."""

import torch


def load_matplotlib():
    """Lazy load principal matplotlib routines.

    This is not beautiful. But if matplotlib is installed, but never used, it
    reduces load time significantly.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    return matplotlib, plt


class InterfaceMPL(object):
    """Class for plotting ``simpegtorch.discretize`` meshes with matplotlib.

    This interface adds three plotting methods to all ``simpegtorch.discretize`` meshes.
    :py:attr:`~InterfaceMPL.plot_grid` will plot gridded points for 2D and 3D meshes.
    :py:attr:`~InterfaceMPL.plot_image` is used for plotting models, scalars and vectors
    defined on a given mesh. And :py:attr:`~InterfaceMPL.plot_slice` is used for plotting
    models, scalars and vectors on a 2D slice through a 3D mesh.
    """

    def plot_grid(
        self,
        ax=None,
        nodes=False,
        faces=False,
        centers=False,
        edges=False,
        lines=True,
        show_it=False,
        **kwargs,
    ):
        """Plot the grid for nodal, cell-centered and staggered grids.

        For 2D and 3D meshes, this method plots the mesh grid. Additionally,
        the user can choose to denote edge, face, node and cell center locations.
        This function is built upon the ``matplotlib.pyplot.plot`` function
        and will accept associated keyword arguments.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            The axes to draw on. *None* produces a new axes.
        nodes, faces, centers, edges, lines : bool, optional
            Whether to plot the corresponding item
        show_it : bool, optional
            whether to call plt.show()
        color : Color or str, optional
            If lines=True, defines the color of the grid lines.
        linewidth : float, optional
            If lines=True, defines the thickness of the grid lines.

        Returns
        -------
        matplotlib.axes.Axes
            Axes handle for the plot

        Notes
        -----
        Excess arguments are passed on to `plot`

        Examples
        --------
        Plotting a 2D TensorMesh grid

        >>> from matplotlib import pyplot as plt
        >>> import simpegtorch.discretize as discretize
        >>> import torch
        >>> h1 = torch.linspace(.1, .5, 3)
        >>> h2 = torch.linspace(.1, .5, 5)
        >>> mesh = discretize.TensorMesh([h1, h2])
        >>> mesh.plot_grid(nodes=True, faces=True, centers=True, lines=True)
        >>> plt.show()

        Plotting a 3D TensorMesh grid

        >>> from matplotlib import pyplot as plt
        >>> import simpegtorch.discretize as discretize
        >>> import torch
        >>> h1 = torch.linspace(.1, .5, 3)
        >>> h2 = torch.linspace(.1, .5, 5)
        >>> h3 = torch.linspace(.1, .5, 7)
        >>> mesh = discretize.TensorMesh([h1, h2, h3])
        >>> mesh.plot_grid(centers=True, lines=True)
        >>> plt.show()
        """
        matplotlib, plt = load_matplotlib()
        if ax is None:
            ax = plt.gca()

        if hasattr(self, "dim"):
            if self.dim == 1:
                out = self.__plot_grid_tensor1D(
                    ax=ax,
                    nodes=nodes,
                    centers=centers,
                    lines=lines,
                    **kwargs,
                )
            elif self.dim == 2:
                out = self.__plot_grid_tensor2D(
                    ax=ax,
                    nodes=nodes,
                    centers=centers,
                    lines=lines,
                    **kwargs,
                )
            elif self.dim == 3:
                out = self.__plot_grid_tensor3D(
                    ax=ax,
                    nodes=nodes,
                    centers=centers,
                    lines=lines,
                    **kwargs,
                )
            else:
                raise Exception(
                    "grid has not been implemented for {0}D".format(self.dim)
                )
        else:
            raise NotImplementedError("plot_grid is not implemented for this mesh type")

        if show_it:
            plt.show()
        return out

    def plot_image(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        clim=None,
        show_it=False,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_thickness=None,
        stream_threshold=None,
        **kwargs,
    ):
        """Plot quantities defined on a given mesh.

        This method is primarily used to plot models, scalar quantities and vector
        quantities defined on 2D meshes. For 3D :class:`simpegtorch.discretize.TensorMesh` however,
        this method will plot the quantity for every slice of the 3D mesh.

        Parameters
        ----------
        v : torch.Tensor
            Gridded values being plotted. The length of the array depends on the quantity being
            plotted; e.g. if the quantity is a scalar value defined on mesh nodes, the
            length must be equal to the number of mesh nodes.
        v_type : {'CC','CCV', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Defines the input parameter *v*.
        view : {'real', 'imag', 'abs', 'vec'}
            For complex scalar quantities, options are included to image the real, imaginary or
            absolute value. For vector quantities, *view* must be set to 'vec'.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. *None* produces a new Axes.
        clim : tuple of float, optional
            length 2 tuple of (vmin, vmax) for the color limits
        range_x, range_y : tuple of float, optional
            length 2 tuple of (min, max) for the bounds of the plot axes.
        pcolor_opts : dict, optional
            Arguments passed on to ``pcolormesh``
        grid : bool, optional
            Whether to plot the edges of the mesh cells.
        grid_opts : dict, optional
            Arguments passed on to the ``plot_grid`` method
        stream_opts : dict, optional
            Arguments for streamline plotting (when plotting vector quantities)
        stream_thickness : int, optional
            The relative thickness of the stream lines for the plot
        stream_threshold : float, optional
            The relative threshold for the cutoff of stream line plotting

        Returns
        -------
        matplotlib.axes.Axes
            Axes handle for the plot

        Examples
        --------
        Plotting a 2D model on a 2D mesh

        >>> import torch
        >>> import simpegtorch.discretize as discretize
        >>> from matplotlib import pyplot as plt
        >>> h1 = torch.ones(20) * 5.
        >>> h2 = torch.ones(20) * 2.
        >>> mesh = discretize.TensorMesh([h1, h2])
        >>> v = torch.sin(mesh.cell_centers_x[:, 0] / 10.) * torch.cos(mesh.cell_centers_x[:, 1] / 10.)
        >>> mesh.plot_image(v)
        >>> plt.show()
        """
        # Ensure v is a torch tensor
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float64)

        matplotlib, plt = load_matplotlib()
        if ax is None:
            ax = plt.gca()

        if hasattr(self, "dim"):
            if self.dim == 1:
                out = self.__plot_image_tensor1D(
                    v,
                    v_type=v_type,
                    grid=grid,
                    view=view,
                    ax=ax,
                    clim=clim,
                    pcolor_opts=pcolor_opts,
                    grid_opts=grid_opts,
                    **kwargs,
                )
            elif self.dim == 2:
                out = self.__plot_image_tensor2D(
                    v,
                    v_type=v_type,
                    grid=grid,
                    view=view,
                    ax=ax,
                    clim=clim,
                    pcolor_opts=pcolor_opts,
                    stream_opts=stream_opts,
                    grid_opts=grid_opts,
                    range_x=range_x,
                    range_y=range_y,
                    sample_grid=sample_grid,
                    stream_thickness=stream_thickness,
                    stream_threshold=stream_threshold,
                    **kwargs,
                )
            elif self.dim == 3:
                out = self.__plot_image_tensor3D(
                    v,
                    v_type=v_type,
                    grid=grid,
                    view=view,
                    ax=ax,
                    clim=clim,
                    pcolor_opts=pcolor_opts,
                    grid_opts=grid_opts,
                    **kwargs,
                )
            else:
                raise Exception(
                    "image has not been implemented for {0}D".format(self.dim)
                )
        else:
            raise NotImplementedError(
                "plot_image is not implemented for this mesh type"
            )

        if show_it:
            plt.show()
        return out

    def plot_slice(
        self,
        v,
        v_type="CC",
        normal="Y",
        ind=None,
        slice_loc=None,
        grid=False,
        view="real",
        ax=None,
        clim=None,
        show_it=False,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_thickness=None,
        stream_threshold=None,
        **kwargs,
    ):
        """Plot a slice through a 3D mesh.

        This method is primarily used to plot models, scalar quantities and vector
        quantities on a 2D slice through a 3D mesh.

        Parameters
        ----------
        v : torch.Tensor
            Gridded values being plotted.
        v_type : {'CC','CCV', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Defines the input parameter *v*.
        normal : {'X', 'Y', 'Z'}
            The normal direction for the slice
        ind : int, optional
            The index of the slice along the normal direction
        slice_loc : float, optional
            The location of the slice along the normal direction (alternative to ind)
        view : {'real', 'imag', 'abs', 'vec'}
            For complex scalar quantities, options are included to image the real, imaginary or
            absolute value. For vector quantities, *view* must be set to 'vec'.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on.
        clim : tuple of float, optional
            length 2 tuple of (vmin, vmax) for the color limits
        range_x, range_y : tuple of float, optional
            length 2 tuple of (min, max) for the bounds of the plot axes.
        pcolor_opts : dict, optional
            Arguments passed on to ``pcolormesh``
        grid : bool, optional
            Whether to plot the edges of the mesh cells.
        grid_opts : dict, optional
            Arguments passed on to the ``plot_grid`` method
        stream_opts : dict, optional
            Arguments for streamline plotting (when plotting vector quantities)

        Returns
        -------
        matplotlib.axes.Axes
            Axes handle for the plot
        """
        # Ensure v is a torch tensor
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float64)

        matplotlib, plt = load_matplotlib()
        if ax is None:
            ax = plt.gca()

        if hasattr(self, "dim") and self.dim == 3:
            out = self.__plot_slice_tensor(
                v,
                v_type=v_type,
                normal=normal,
                ind=ind,
                slice_loc=slice_loc,
                grid=grid,
                view=view,
                ax=ax,
                clim=clim,
                pcolor_opts=pcolor_opts,
                stream_opts=stream_opts,
                grid_opts=grid_opts,
                range_x=range_x,
                range_y=range_y,
                sample_grid=sample_grid,
                stream_thickness=stream_thickness,
                stream_threshold=stream_threshold,
                **kwargs,
            )
        else:
            raise NotImplementedError("plot_slice is only implemented for 3D meshes")

        if show_it:
            plt.show()
        return out

    def __plot_grid_tensor1D(
        self, ax, nodes=False, centers=False, lines=True, **kwargs
    ):
        """1D tensor mesh grid plotting."""
        matplotlib, plt = load_matplotlib()

        # Convert PyTorch tensors to numpy for matplotlib
        nodes_x = self.nodes_x.detach().cpu().numpy()
        centers_x = self.cell_centers_x.detach().cpu().numpy()

        if lines:
            for node in nodes_x:
                ax.axvline(x=node, **kwargs)

        if nodes:
            ax.plot(
                nodes_x,
                torch.zeros_like(self.nodes_x).detach().cpu().numpy(),
                "ro",
                **kwargs,
            )

        if centers:
            ax.plot(
                centers_x,
                torch.zeros_like(self.cell_centers_x).detach().cpu().numpy(),
                "bo",
                **kwargs,
            )

        return ax

    def __plot_grid_tensor2D(
        self, ax, nodes=False, centers=False, lines=True, **kwargs
    ):
        """2D tensor mesh grid plotting."""
        matplotlib, plt = load_matplotlib()

        # Convert PyTorch tensors to numpy for matplotlib
        nodes_x = self.nodes_x.detach().cpu().numpy()
        nodes_y = self.nodes_y.detach().cpu().numpy()

        if lines:
            # Vertical lines
            for x in nodes_x:
                ax.plot([x, x], [nodes_y[0], nodes_y[-1]], "k-", **kwargs)
            # Horizontal lines
            for y in nodes_y:
                ax.plot([nodes_x[0], nodes_x[-1]], [y, y], "k-", **kwargs)

        if nodes:
            X, Y = torch.meshgrid(self.nodes_x, self.nodes_y, indexing="ij")
            ax.scatter(
                X.detach().cpu().numpy().flatten(),
                Y.detach().cpu().numpy().flatten(),
                c="r",
                marker="o",
            )

        if centers:
            X, Y = torch.meshgrid(
                self.cell_centers_x, self.cell_centers_y, indexing="ij"
            )
            ax.scatter(
                X.detach().cpu().numpy().flatten(),
                Y.detach().cpu().numpy().flatten(),
                c="b",
                marker="s",
            )

        ax.set_xlim([nodes_x[0], nodes_x[-1]])
        ax.set_ylim([nodes_y[0], nodes_y[-1]])
        ax.set_aspect("equal")

        return ax

    def __plot_grid_tensor3D(
        self, ax, nodes=False, centers=False, lines=True, **kwargs
    ):
        """3D tensor mesh grid plotting (shows wireframe of outer boundaries)."""

        _, plt = load_matplotlib()

        if not hasattr(ax, "zaxis"):
            # Create 3D axes if not already 3D
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection="3d")

        # Convert PyTorch tensors to numpy for matplotlib
        nodes_x = self.nodes_x.detach().cpu().numpy()
        nodes_y = self.nodes_y.detach().cpu().numpy()
        nodes_z = self.nodes_z.detach().cpu().numpy()

        if lines:
            # Draw wireframe of outer boundaries
            # X-Y faces at min/max Z
            for z in [nodes_z[0], nodes_z[-1]]:
                for x in nodes_x:
                    ax.plot(
                        [x, x],
                        [nodes_y[0], nodes_y[-1]],
                        [z, z],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )
                for y in nodes_y:
                    ax.plot(
                        [nodes_x[0], nodes_x[-1]],
                        [y, y],
                        [z, z],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )

            # X-Z faces at min/max Y
            for y in [nodes_y[0], nodes_y[-1]]:
                for x in nodes_x:
                    ax.plot(
                        [x, x],
                        [y, y],
                        [nodes_z[0], nodes_z[-1]],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )
                for z in nodes_z:
                    ax.plot(
                        [nodes_x[0], nodes_x[-1]],
                        [y, y],
                        [z, z],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )

            # Y-Z faces at min/max X
            for x in [nodes_x[0], nodes_x[-1]]:
                for y in nodes_y:
                    ax.plot(
                        [x, x],
                        [y, y],
                        [nodes_z[0], nodes_z[-1]],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )
                for z in nodes_z:
                    ax.plot(
                        [x, x],
                        [nodes_y[0], nodes_y[-1]],
                        [z, z],
                        "k-",
                        alpha=0.3,
                        **kwargs,
                    )

        if centers:
            centers = self.cell_centers.detach().cpu().numpy()
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                c="b",
                marker="s",
                alpha=0.3,
            )

        ax.set_xlim([nodes_x[0], nodes_x[-1]])
        ax.set_ylim([nodes_y[0], nodes_y[-1]])
        ax.set_zlim([nodes_z[0], nodes_z[-1]])

        return ax

    def __plot_image_tensor1D(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        clim=None,
        pcolor_opts=None,
        grid_opts=None,
        **kwargs,
    ):
        """1D tensor mesh image plotting."""
        matplotlib, plt = load_matplotlib()

        # Handle complex values
        if torch.is_complex(v):
            if view == "real":
                v = v.real
            elif view == "imag":
                v = v.imag
            elif view == "abs":
                v = torch.abs(v)

        v_np = v.detach().cpu().numpy()

        if v_type in ["CC", "CCV"]:
            x = self.cell_centers_x.detach().cpu().numpy()
        elif v_type == "N":
            x = self.nodes_x.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"v_type {v_type} not implemented for 1D")

        ax.plot(x, v_np, **kwargs)

        if grid and grid_opts is None:
            grid_opts = {}
        if grid:
            self.plot_grid(ax=ax, lines=True, **grid_opts)

        return ax

    def __plot_image_tensor2D(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        clim=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_thickness=None,
        stream_threshold=None,
        **kwargs,
    ):
        """2D tensor mesh image plotting."""
        matplotlib, plt = load_matplotlib()

        if pcolor_opts is None:
            pcolor_opts = {}

        # Handle complex values
        if torch.is_complex(v):
            if view == "real":
                v = v.real
            elif view == "imag":
                v = v.imag
            elif view == "abs":
                v = torch.abs(v)
            elif view == "vec":
                # Vector plotting for complex values
                pass

        if view == "vec":
            # Handle vector plotting
            if v_type in ["F", "E"]:
                # Face or edge vectors
                if stream_opts is None:
                    stream_opts = {}

                # Extract vector components - this is simplified and may need adjustment
                # based on the exact structure of vector data in simpegtorch
                if v.shape[-1] == 2 or len(v) == 2 * self.n_faces:
                    # 2D vector field
                    if len(v) == 2 * self.n_faces:
                        vx = v[: self.n_faces]
                        vy = v[self.n_faces :]
                    else:
                        vx = v[:, 0]
                        vy = v[:, 1]

                    # Convert to cell centers for plotting
                    # This is a simplified approach - exact implementation depends on mesh structure
                    centers = self.cell_centers.detach().cpu().numpy()
                    vx_np = vx.detach().cpu().numpy()
                    vy_np = vy.detach().cpu().numpy()

                    ax.quiver(centers[:, 0], centers[:, 1], vx_np, vy_np, **stream_opts)

        else:
            # Scalar plotting
            v_np = v.detach().cpu().numpy()

            if v_type in ["CC", "CCV"]:
                # Cell-centered data
                X, Y = torch.meshgrid(self.nodes_x, self.nodes_y, indexing="ij")
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()

                # Reshape data to grid
                if v_np.ndim == 1:
                    v_grid = v_np.reshape(self.shape_cells, order="F")
                else:
                    v_grid = v_np

                im = ax.pcolormesh(X_np, Y_np, v_grid, **pcolor_opts)

            elif v_type == "N":
                # Nodal data
                X, Y = torch.meshgrid(self.nodes_x, self.nodes_y, indexing="ij")
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()

                if v_np.ndim == 1:
                    v_grid = v_np.reshape(self.shape_nodes, order="F")
                else:
                    v_grid = v_np

                im = ax.pcolormesh(X_np, Y_np, v_grid, **pcolor_opts)

            if clim is not None:
                im.set_clim(clim)

            plt.colorbar(im, ax=ax)

        if grid and grid_opts is None:
            grid_opts = {}
        if grid:
            self.plot_grid(ax=ax, lines=True, **grid_opts)

        if range_x is not None:
            ax.set_xlim(range_x)
        if range_y is not None:
            ax.set_ylim(range_y)

        ax.set_aspect("equal")

        return ax

    def __plot_image_tensor3D(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        clim=None,
        pcolor_opts=None,
        grid_opts=None,
        **kwargs,
    ):
        """3D tensor mesh image plotting (shows slices)."""
        matplotlib, plt = load_matplotlib()

        # For 3D, we'll create subplots showing slices
        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax = axes

        # Handle complex values
        if torch.is_complex(v):
            if view == "real":
                v = v.real
            elif view == "imag":
                v = v.imag
            elif view == "abs":
                v = torch.abs(v)

        v_np = v.detach().cpu().numpy()

        if v_type in ["CC", "CCV"]:
            # Show middle slices in each direction
            mid_x = self.shape_cells[0] // 2
            mid_y = self.shape_cells[1] // 2
            mid_z = self.shape_cells[2] // 2

            if v_np.ndim == 1:
                v_grid = v_np.reshape(self.shape_cells, order="F")
            else:
                v_grid = v_np

            # YZ slice (fixed X)
            if isinstance(ax, list) or isinstance(ax, tuple):
                im1 = ax[0].imshow(v_grid[mid_x, :, :].T, origin="lower", **pcolor_opts)
                ax[0].set_title(f"YZ slice at X={mid_x}")
                ax[0].set_xlabel("Y")
                ax[0].set_ylabel("Z")

                # XZ slice (fixed Y)
                im2 = ax[1].imshow(v_grid[:, mid_y, :].T, origin="lower", **pcolor_opts)
                ax[1].set_title(f"XZ slice at Y={mid_y}")
                ax[1].set_xlabel("X")
                ax[1].set_ylabel("Z")

                # XY slice (fixed Z)
                im3 = ax[2].imshow(v_grid[:, :, mid_z], origin="lower", **pcolor_opts)
                ax[2].set_title(f"XY slice at Z={mid_z}")
                ax[2].set_xlabel("X")
                ax[2].set_ylabel("Y")

                if clim is not None:
                    im1.set_clim(clim)
                    im2.set_clim(clim)
                    im3.set_clim(clim)

                plt.colorbar(im1, ax=ax[0])
                plt.colorbar(im2, ax=ax[1])
                plt.colorbar(im3, ax=ax[2])
            else:
                # Single axis - show XY slice
                im = ax.imshow(v_grid[:, :, mid_z], origin="lower", **pcolor_opts)
                ax.set_title(f"XY slice at Z={mid_z}")
                if clim is not None:
                    im.set_clim(clim)
                plt.colorbar(im, ax=ax)

        return ax

    def __plot_slice_tensor(
        self,
        v,
        v_type="CC",
        normal="Y",
        ind=None,
        slice_loc=None,
        grid=False,
        view="real",
        ax=None,
        clim=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_thickness=None,
        stream_threshold=None,
        **kwargs,
    ):
        """Plot a slice through a 3D tensor mesh."""
        matplotlib, plt = load_matplotlib()

        if pcolor_opts is None:
            pcolor_opts = {}

        # Handle complex values
        if torch.is_complex(v):
            if view == "real":
                v = v.real
            elif view == "imag":
                v = v.imag
            elif view == "abs":
                v = torch.abs(v)

        v_np = v.detach().cpu().numpy()

        # Determine slice index
        if ind is None and slice_loc is not None:
            if normal == "X":
                distances = torch.abs(self.cell_centers_x - slice_loc)
                ind = torch.argmin(distances).item()
            elif normal == "Y":
                distances = torch.abs(self.cell_centers_y - slice_loc)
                ind = torch.argmin(distances).item()
            elif normal == "Z":
                distances = torch.abs(self.cell_centers_z - slice_loc)
                ind = torch.argmin(distances).item()
        elif ind is None:
            # Default to middle slice
            if normal == "X":
                ind = self.shape_cells[0] // 2
            elif normal == "Y":
                ind = self.shape_cells[1] // 2
            elif normal == "Z":
                ind = self.shape_cells[2] // 2

        if v_type in ["CC", "CCV"]:
            if v_np.ndim == 1:
                v_grid = v_np.reshape(self.shape_cells, order="F")
            else:
                v_grid = v_np

            if normal == "X":
                # YZ slice
                slice_data = v_grid[ind, :, :]
                Y, Z = torch.meshgrid(
                    self.cell_centers_y, self.cell_centers_z, indexing="ij"
                )
                X_plot, Y_plot = Y.detach().cpu().numpy(), Z.detach().cpu().numpy()
                ax.set_xlabel("Y")
                ax.set_ylabel("Z")
            elif normal == "Y":
                # XZ slice
                slice_data = v_grid[:, ind, :]
                X, Z = torch.meshgrid(
                    self.cell_centers_x, self.cell_centers_z, indexing="ij"
                )
                X_plot, Y_plot = X.detach().cpu().numpy(), Z.detach().cpu().numpy()
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
            elif normal == "Z":
                # XY slice
                slice_data = v_grid[:, :, ind]
                X, Y = torch.meshgrid(
                    self.cell_centers_x, self.cell_centers_y, indexing="ij"
                )
                X_plot, Y_plot = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

            im = ax.pcolormesh(X_plot, Y_plot, slice_data, **pcolor_opts)

            if clim is not None:
                im.set_clim(clim)

            plt.colorbar(im, ax=ax)

        if grid and grid_opts is None:
            grid_opts = {}
        if grid:
            # Plot grid for the slice - this would need to be implemented
            # for the specific slice orientation
            pass

        if range_x is not None:
            ax.set_xlim(range_x)
        if range_y is not None:
            ax.set_ylim(range_y)

        ax.set_aspect("equal")

        return ax
