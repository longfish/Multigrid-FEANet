"""Domain samplers implementation for the heat in plate problem"""

import idrlnet.shortcut as sc
import itertools
import numpy as np


def sample_interior_domain(geo, parameters_range, density_collocation):
    """Samples the collocation data points, with the respective targets.

    Args:
        geo: Object of type geo.
        parameters_range: Range of variable t and hot_bc.
        density_collocation: Sampling density of collocation points.

    Returns:
        Sampled collocation points and respective constraint.
    """
    points = geo.sample_interior(density_collocation,
                                 param_ranges=parameters_range)
    # it is possible to add bounds to the interior sampled In this case we want
    # to sample the whole domain, so no bounds are required.

    constraints = {"diffusion_u": 0}
    # This diffusion is obtained from the PDENode provided for the points from
    # this datanode (when you run this script, you obtain a graph figure that
    # makes the point of how the predictions are obtained for each domain
    # clear).

    return points, constraints


def sample_initial_domain(geo, t, parameters_range, initial_temp,
                          density_initial):
    """Samples the initial condition data points, with the respective targets.

    Args:
        geo: Object of type geo.
        t: Temporal variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        initial_temp: temperature of initial points (initial condition)
        density_initial: Sampling density of initial points.

    Returns:
        Sampled initial points and respective constraint.
    """
    parameters_range_initial_t = parameters_range.copy()
    parameters_range_initial_t[t] = 0.0  # Fix the time variable, while letting
    # the other parameters to range
    points = geo.sample_interior(density_initial,
                                 param_ranges=parameters_range_initial_t)
    constraints = sc.Variables({"u": initial_temp})

    return points, constraints


def sample_cold_boundary_domain(geo, x, y, parameters_range,
                                density_cold_boundary, cold_edge_temp,
                                plate_length, sieve_tolerance):
    """Samples the cold boundaries data points, with the respective targets.

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        plate_length: Length of the plate edge.
        density_cold_boundary: Sampling density of cold boundary points.
        cold_edge_temp: Temperature of cold boundaries points.
        sieve_tolerance: Tolerance to filter points from each boundary.

    Returns:
        Sampled cold boundary points and respective constraint.
    """

    points = geo.sample_boundary(
        density_cold_boundary,
        sieve=((y < plate_length / 2) &
               ((x < -plate_length / 2 + sieve_tolerance) |
                (x > plate_length / 2 - sieve_tolerance) |
                (y < -plate_length / 2 + sieve_tolerance))),
        param_ranges=parameters_range)
    constraints = sc.Variables({"u": cold_edge_temp})

    return points, constraints


def sample_hot_boundary_domain(geo, x, y, parameters_range,
                               density_hot_boundary, plate_length,
                               sieve_tolerance):
    """Samples the hot boundary data points, with the respective targets.

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        plate_length: Length of the plate edge.
        density_hot_boundary: Sampling density of hot boundary points.
        sieve_tolerance: Tolerance to filter points from each boundary.

    Returns:
        Sampled hot boundary points and respective constraint.
    """

    points = geo.sample_boundary(
        density_hot_boundary,
        sieve=((x > -plate_length / 2) & (x < plate_length / 2) &
               (y > plate_length / 2 - sieve_tolerance)),
        param_ranges=parameters_range)
    constraints = sc.Variables({"u": points["hot_bc"]})

    return points, constraints


def sample_holes_boundary_domain(geo, x, y, parameters_range,
                                 density_holes_boundary, holes_temp,
                                 plate_length):
    """Samples all the non-external boundary points and respective targets.

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        plate_length: Length of the plate edge.
        density_holes_boundary: Sampling density of hole(s) boundary points.
        holes_temp: Temperature of hole(s) boundary points.

    Returns:
        Sampled hole boundary points and respective constraint.
    """

    points = geo.sample_boundary(
        density_holes_boundary,
        sieve=((x > -plate_length / 2) & (x < plate_length / 2) &
               (y > -plate_length / 2) & (y < plate_length / 2)),
        param_ranges=parameters_range)
    constraints = sc.Variables({"u": holes_temp})

    return points, constraints


def sample_inference_domain(plate_length, output_num_x, output_num_y, num_plots,
                            t_final, hot_edge_temp_plot):
    """Samples the domain with the regular grid on which we evaluate the
    trained PINN

    Args:
        plate_length: Length of the plate edge.
        output_num_x: Number of points in the x-axis discretization.
        output_num_y: Number of points in the y-axis discretization.
        num_plots: Number of plots to be obtained.
        t_final: Final instant of time.
        hot_edge_temp_plot: Temperature of the hot edge in the PINN solution
        we intend to plot.

    Returns:
        Sampled inference points and respective constraint.
    """

    inference_points_matrix = generate_grid_matrix(plate_length, t_final,
                                                   output_num_x, output_num_y,
                                                   num_plots)
    points = {
        "x":
            inference_points_matrix[:, 0],
        "y":
            inference_points_matrix[:, 1],
        "t":
            inference_points_matrix[:, 2],
        "hot_bc":
            hot_edge_temp_plot * np.ones((inference_points_matrix.shape[0], 1))
    }
    constraints = {}

    return points, constraints


def generate_grid_matrix(plate_length, t_final, output_num_x, output_num_y,
                         num_plots):
    """Generates a matrix which has in its rows all the points of a 3D grid
        with dimensions (output_num_x, output_num_y, num_plots), that occupies
        the whole problem domain

        Args:
            plate_length: Length of the plate edge.
            t_final: Final instant of time.
            output_num_x: Number of points in the x-axis discretization.
            output_num_y: Number of points in the y-axis discretization.
            num_plots: Number of plots to be obtained.

        Returns:
            Matrix (shape: [output_num_x * output_num_y * num_plots, 3]) which
            has in its rows all the points of a 3D grid that occupies the whole
            domain.
        """
    x_list = np.linspace(-plate_length / 2, plate_length / 2, output_num_x)
    y_list = np.linspace(-plate_length / 2, plate_length / 2, output_num_y)
    t_list = np.linspace(0, t_final, num_plots)
    grid_matrix = np.matrix(list(itertools.product(*[x_list, y_list, t_list])))

    return grid_matrix
