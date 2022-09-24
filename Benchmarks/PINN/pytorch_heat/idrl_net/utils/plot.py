"""Plot utilities for the 2D heat diffusion problem on a plate with holes"""

import os

import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import imageio


def generate_output_across_time(u_memory,
                                plate_length,
                                diff_coef,
                                output_formats,
                                holes_list=None,
                                holes_temperature=0,
                                output_path=None,
                                error_bool=False,
                                colorbar_limits=None):
    """Outputs a figure or gif of the provided function in a set of timeframes.

    Args:
        u_memory: Saved timeframes to be plotted [[t_1, u_1], ..., [t_k, u_k]].
        t_i are the time instants respective to the plate u_i, which is a matrix
        representing the plate with each entry corresponding to the temperature
        at that given point.
        plate_length: Length of the plate edges.
        diff_coef: Diffusion coeficient considered.
        output_formats: List of formats in which we intend to represent the
        provided function.
        holes_list: List of holes (named tuples) considered
        holes_temperature: Temperature of holes boundary. The interior points of
        the hole are plotted with that temperature
        output_path: Path where the gif is saved.

    Returns:
        Plot of the inferred solution function in the prescribed timesteps.
    """

    # Initialization
    output_num_x = u_memory[0][1].shape[0]
    output_num_y = u_memory[0][1].shape[1]
    num_plots = len(u_memory)
    grid_limits = (-plate_length / 2, plate_length / 2, -plate_length / 2,
                   plate_length / 2)  # left, right, bottom, top

    # Different features between temperature and error plot
    if error_bool:
        plot_label = "Error"
        color_map = "cividis"
    else:
        plot_label = "Temperature"
        color_map = "inferno"

    # Get colorbar limits
    if colorbar_limits is None:
        min_value_grid_all_time, max_value_grid_all_time = \
            get_grid_extreme_values(u_memory)
    else:
        min_value_grid_all_time = colorbar_limits[0]
        max_value_grid_all_time = colorbar_limits[1]

    # Formatting font (to LaTeX)
    rc_params = {"font.family": "serif", "text.usetex": True}

    if "figure" in output_formats:
        # Build figure skeleton
        fig, axs = plt.subplots(math.ceil(num_plots / 5),
                                5,
                                figsize=(20, 8),
                                facecolor="w",
                                edgecolor="k")
        fig.subplots_adjust(hspace=.5, wspace=.4, right=0.8)
        axs = axs.ravel()

        # Plot for each time instant
        for time_idx in range(num_plots):
            im = plot_plate(
                axs[time_idx],
                u_memory[time_idx][0],
                u_memory[time_idx][1],
                holes_list,
                holes_temperature,
                grid_limits,
                output_num_x,
                output_num_y,
                min_value_grid_all_time,
                max_value_grid_all_time,
                diff_coef,
                color_map,
            )

        # Create color map to draw colors from
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=plot_label)

        # Save figure
        plt.savefig(os.path.join(output_path, "figure_across_time"),
                    dpi=300,
                    bbox_inches="tight")

    if "gif" in output_formats:
        # Generate new directory to save gif auxiliary figures
        frames_folder_path = os.path.join(output_path, "frames")
        if not os.path.exists(frames_folder_path):
            os.mkdir(frames_folder_path)

        for time_idx in range(num_plots):
            rc_params["font.size"] = 20
            with mpl.rc_context(rc_params):
                # Generate frame for each time instant considered considered
                fig, ax = plt.subplots()
                im = plot_plate(
                    ax,
                    u_memory[time_idx][0],
                    u_memory[time_idx][1],
                    holes_list,
                    holes_temperature,
                    grid_limits,
                    output_num_x,
                    output_num_y,
                    min_value_grid_all_time,
                    max_value_grid_all_time,
                    diff_coef,
                    color_map,
                )

                ## Create color map to draw colors from
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax, label=plot_label)

                # Save frame
                frame_name = f"{time_idx:03d}.png"
                plt.savefig(os.path.join(frames_folder_path, frame_name),
                            dpi=300,
                            bbox_inches="tight")
                plt.close(fig)

        # Generate gif
        generate_gif_from_frames_in_directory(frames_folder_path, output_path,
                                              f"{plot_label.lower()}.gif")


def get_grid_extreme_values(u_memory):
    """Get extreme values of the grid across all the times memorized.

    Args:
        u_memory: Saved timeframes to be plotted [[t_1, u_1], ..., [t_k, u_k]].
        t_i are the time instants respective to the plate u_i, which is a matrix
        representing the plate with each entry corresponding to the temperature

    Returns:
        The maximal and minimal value verified in the grid across all the
        timeframes considered.
    """
    all_u_list = [pair_time_u[1] for pair_time_u in u_memory]
    min_value = np.min(all_u_list)
    max_value = np.max(all_u_list)

    return min_value, max_value


def plot_plate(ax, instant, grid_to_plot, holes_list, holes_temperatures,
               grid_limits, output_num_x, output_num_y, scale_min_limit,
               scale_max_limit, diff_coef, color_map):
    """Plots a plate state at a given instant.

    Args:
        ax: Object where the figure is plotted
        instant: Time instant cor
        grid_to_plot: Grid state at the given time instant
        holes_list: List of holes in the plate
        grid_limits: Limits of the plate (metrics)
        output_num_x: Number of points used in the discretization of the x-axis
        output_num_y: Number of points used in the discretization of the y-axis
        scale_min_limit: Lower limit of the plot scale.
        scale_max_limit: Higher limit of the plot scale.

    Returns:
        Image generated with temperature for each point across the plate, at a
        given instant.
    """

    grid_without_holes = filter_out_holes(grid_to_plot, holes_list,
                                          holes_temperatures, grid_limits,
                                          output_num_x, output_num_y)
    im = ax.imshow(grid_without_holes,
                   aspect="equal",
                   extent=grid_limits,
                   origin="lower",
                   vmin=scale_min_limit,
                   vmax=scale_max_limit,
                   cmap=color_map)
    ax.set_title(f"$D = {diff_coef}; \\quad t = {instant:.2f}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return im


def filter_out_holes(grid_to_plot, holes_list, holes_temperatures, grid_limits,
                     output_num_x, output_num_y):
    """Filters out the grid values which were computed for point inside holes.

    This function is needed as we do not want to plot the PINN temperature
    predictions for points outside of the training domain.

    Args:
        grid_to_plot: Grid to be plotted
        holes_list: List of holes in the plate
        grid_limits: Limits of the grid to be plotted (metrics)
        output_num_x: Number of points used in the discretization of the x-axis
        output_num_y: Number of points used in the discretization of the y-axis

    Returns:
        Grid where the values inside holes are converted to holes_temperature
    """

    x_coordinates_grid = np.linspace(grid_limits[0], grid_limits[1],
                                     output_num_x)
    y_coordinates_grid = np.linspace(grid_limits[2], grid_limits[3],
                                     output_num_y)
    for x_idx, x_value in enumerate(x_coordinates_grid):
        for y_idx, y_value in enumerate(y_coordinates_grid):
            if point_within_holes(x_value, y_value, holes_list):
                # Note that x is denoted in the horizontal axis of the grid and
                # y in the vertical one. Thus, we have the following indexing:
                grid_to_plot[y_idx, x_idx] = holes_temperatures

    return grid_to_plot


def point_within_holes(x_point, y_point, holes_list):
    """Checks if a given grid point is inside the holes of the prescribed plate

    Args:
        x_point: x coordinate of the queried point
        y_point: y coordinate of the queried point
        holes_list: List of holes in the plate

    Returns:
        Boolean that indicates if the point provided is inside the holes of the
        plate
    """
    if holes_list is not None:
        for hole_coords in holes_list:
            x_center, y_center, radius = hole_coords
            if (x_point - x_center)**2 + (y_point - y_center)**2 < radius**2:
                return True
    else:
        return False


def generate_gif_from_frames_in_directory(frames_directory, movie_directory,
                                          movie_name):
    """Generates a gif from the frames stored in frames_directory.

    Args:
        frames_directory: Directory where the frames to be used for the gif are.
        movie_directory: Directory where the gif is stored.
        movie_name: Name given to the gif.

    Returns:
        A gif from the frames in frames_directory.
    """

    # Read all .png files in frames_directory
    frames_list = [
        file for file in os.listdir(frames_directory) if file.endswith(".png")
    ]

    # Order frames_list
    frames_list.sort()

    # Initialize writer
    writer = imageio.get_writer(os.path.join(movie_directory, movie_name),
                                duration=10 / len(frames_list))

    # Add all frames to writer
    for file_name in frames_list:
        frame_path = os.path.join(frames_directory, file_name)
        writer.append_data(imageio.imread(frame_path))

    # Close writer
    writer.close()
