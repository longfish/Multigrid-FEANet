"""Solves the 2D heat diffusion problem using Finite-Differences Method (FDM)

Here, we address the heat diffusion problem in a 2D square plate, where one
of the edges is kept at high temperature (hot edge) and the remaining three
are at a lower temperature (cold edges). Note that, at the beginning of the
simulation, the whole plate is at a given temperature (initial condition of
the problem).

The particular FDM method here implemented is the Forward in Time, Central in
Space method (FTCS). This method and its results are extensively discussed in
the README.md file of this directory.

Output:
Pickle file with solution of the heat equation, u(t, x, y), evaluated in a
regular spatial grid and at evenly spaced times, and its representation in a
igure or gif file.
"""

import os

import time

from absl import app
from absl import logging
from absl import flags

import jax
import jax.numpy as jnp

import numpy as np

import pickle

import utils.plot
import utils.flags

FLAGS = flags.FLAGS

# Domain definition
flags.DEFINE_float("diff_coef", 0.1, "Diffusion coefficient of the heat PDE.")
flags.DEFINE_float("plate_length",
                   1,
                   "Side length of the considered plate.",
                   lower_bound=0.)
flags.DEFINE_float("t_final", 1, "Duration of the simulation.", lower_bound=0.)

# Domain temperatures
flags.DEFINE_float("hot_edge_temp", 1,
                   "Temperature of the hot edge of the plate.")
flags.DEFINE_float("cold_edge_temp", -1,
                   "Temperature of the cold edge of the plate.")
flags.DEFINE_float("initial_temp", -1,
                   "Temperature of the initial points (i.e. t = 0).")

# FDM Hyperparameters
flags.DEFINE_integer("num_timeframes",
                     100000,
                     "Number of timeframes (discretization of time)",
                     lower_bound=1)
flags.DEFINE_integer(
    "num_x_points",
    100,
    "Number of points considered in the discretization of the x-axis",
    lower_bound=3)  # we have to have at least 2 boundary points and 1 interior
# point for the method to make sense
flags.DEFINE_integer(
    "num_y_points",
    100,
    "Number of points considered in the discretization of the y-axis",
    lower_bound=3)
flags.DEFINE_enum(
    "vectorization_strategy", "jax", ["serial", "numpy", "jax"],
    "Defines which "
    "vectorization the FDM is run with (use 'serial' for standard Python loops "
    "and 'numpy' or 'jax' for accelerated array operations)")

# Output variables
flags.DEFINE_list(
    "output_formats", ["figure", "gif"], "Defines the formats in which the "
    "output is obtained. Only two formats are available: gif and figure. "
    "If 'figure' is in the list provided, it outputs a figure with several "
    "subplots at different timesteps. If 'gif' is in the list provided, it "
    "outputs a gif with several plots at different timesteps.")
flags.DEFINE_string("output_folder", "fdm_results",
                    "Name of the directory where the output is stored.")
flags.DEFINE_integer("num_plots",
                     100,
                     "Number of plots to be obtained.",
                     lower_bound=1)
flags.DEFINE_list(
    "colorbar_limits", None, "Limits of the colorbar present in "
    "the inference output. When set to 'None' (default), those "
    "limits are adjusted automatically.")


def compute_dimension_delta(dimension_range, dimension_num_points):
    """Computes the interval between points for a given dimension.

    Args:
        dimension_range: domain extension in the dimension considered
        dimension_num_points: number of points used in the regular
        discretization of the dimension considered

    Returns:
        Interval between points in the provided dimension
    """

    delta = dimension_range / (dimension_num_points - 1)
    return delta


def compute_fdm_stability_constants(diff_coef, delta_t, delta_x, delta_y):
    """Computes FDM stability constants for the chosen sampling

    Args:
        diff_coef: diffusion coefficient of the heat PDE
        delta_t: interval between timesteps considered
        delta_x: interval between spatial points along the x-axis
        delta_y: interval between spatial points along the y-axis

    Returns:
        Stability constants of the instance provided if it is stable. Otherwise,
        raises an Exception.
    """

    alpha = diff_coef * delta_t / delta_x**2
    beta = diff_coef * delta_t / delta_y**2

    return alpha, beta


def fdm_is_unstable(alpha, beta):
    """Checks if the FDM stability condition is not verified.

    The stability of this method is guaranteed if alpha + beta <= 0.5
    (see analysis for the 2D Heat PDE:
    https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ws2016-2017/num_methods_i/heat.pdf
    - pages 11/12)

    Args:
        alpha: fdm stability constant referent to the x-axis
        beta: fdm stability constant referent to the y-axis

    Returns:
        Boolean that answers if the FDM stability condition is not verified for
        the FDM stability constants provided
    """

    return alpha + beta > .5


def initialize_plate(num_x_points, num_y_points):
    """Initializes the plate where we'll observe heat diffusion

    Args:
        num_x_points: Number of points to be considered in the x-axis
        num_y_points: Number of points to be considered in the y-axis

    Returns:
        Matrix with shape (num_x_points, num_y_points)
    """

    return np.empty((num_x_points, num_y_points))


def set_initial_conditions(u, initial_temp):
    """Sets the initial condition in the whole plate

    Args:
        u: Plate where we are going to impose the initial condition
        initial_temp: Temperature of the initial points (imposed by the initial
        condition)

    Returns:
        Matrix filled with the initial points temperature in every point
    """

    u.fill(initial_temp)

    return u


def set_canonical_boundary_conditions(u, hot_temp, cold_temp):
    """Sets the canonical boundary conditions for the setting considered

    Args:
        u: Plate where we are going to impose the boundary conditions
        hot_temp: Temperature of the hot edge of the plate
        cold_temp: Temperature of the cold edges of the plate

    Returns:
        Matrix filled with the boundary conditions imposed
    """

    u[:, 0] = cold_temp
    u[0, :] = cold_temp
    u[:, -1] = cold_temp
    u[-1, :] = hot_temp

    return u


def run_time(u, alpha, beta, delta_t, num_timeframes, num_plots,
             vectorization_strategy):
    """Runs the numerical simulation throughout time

    Args:
        u: initial state of the grid where the fdm is going to run
        alpha: fdm stability constant referent to the x-axis
        beta: fdm stability constant referent to the y-axis
        delta_t: time interval between fdm iterations
        num_timeframes: number of fdm iterations
        num_plots: number of timeframes to be saved (to be plotted)
        vectorization_strategy: vectorization strategy (serial, numpy or jax)

    Returns:
        A list of pairs [t, u], where u is the grid state at instant t.
    """

    # Initialization
    timeframes_to_save = np.rint(np.linspace(0, num_timeframes,
                                             num_plots))  # evenly spaced t_i
    u_memory = [[timeframes_to_save[0] * delta_t, u]]
    # u_memory will save pairs [[t_i, u_i], [t_j, u_j], ...]

    start = time.time()
    for t_idx in range(1, int(timeframes_to_save[-1]) + 1):

        # advance one timestep
        if vectorization_strategy == "jax":
            u = fdm_advance_time_jax(u, alpha, beta)

        elif vectorization_strategy == "numpy":
            u = fdm_advance_time_numpy(u, alpha, beta)

        elif vectorization_strategy == "serial":
            u = fdm_advance_time_serial(u, alpha, beta)

        # if we want to save that u, keep it in u_memory
        if t_idx in timeframes_to_save:
            u_memory.append([t_idx * delta_t, u])
    end = time.time()
    logging.info("Time spent in FDM: %ss", (end - start))

    return u_memory


@jax.jit
def fdm_advance_time_jax(u, alpha, beta):
    """Advances the simulation by one timestep, via FDM using JAX

    Args:
        u: grid state at instant i
        alpha: fdm stability constant referent to the x-axis
        beta: fdm stability constant referent to the y-axis

    Returns:
        Grid state at instant i+1
    """

    # Vectorized FDM
    next_u = jnp.array(u)

    # compute u(y,x) for the next instant t_{i+1}, using FDM
    vectorized_forward_step = alpha * (jnp.roll(u, 1, axis=0) +
                                        jnp.roll(u, -1, axis=0)) + \
                                beta * (jnp.roll(u, 1, axis=1) +
                                        jnp.roll(u, -1, axis=1)) + \
                                (1 - 2 * alpha - 2 * beta) * u
    next_u = next_u.at[1:-1, 1:-1].set(vectorized_forward_step[1:-1, 1:-1])

    return next_u


def fdm_advance_time_numpy(u, alpha, beta):
    """Advances the simulation by one timestep, via FDM using NumPy

    Args:
        u: grid state at instant i
        alpha: fdm stability constant referent to the x-axis
        beta: fdm stability constant referent to the y-axis

    Returns:
        Grid state at instant i+1
    """

    # Initialization
    next_u = u.copy()  # to preserve the boundary values, we copy u

    # compute u(y,x) for the next instant t_{i+1}, using FDM
    vectorized_forward_step = alpha * (np.roll(u, 1, axis=0) +
                                        np.roll(u, -1, axis=0)) + \
                                beta * (np.roll(u, 1, axis=1) +
                                        np.roll(u, -1, axis=1)) + \
                                (1 - 2 * alpha - 2 * beta) * u
    next_u[1:-1, 1:-1] = vectorized_forward_step[1:-1, 1:-1]

    return next_u


def fdm_advance_time_serial(u, alpha, beta):
    """Advances the simulation by one timestep, via FDM using nested loops

    Args:
        u: grid state at instant i
        alpha: fdm stability constant referent to the x-axis
        beta: fdm stability constant referent to the y-axis)

    Returns:
        Grid state at instant i+1
    """

    # Initialization
    next_u = u.copy()  # to preserve the boundary values, we copy u

    # Nested loops FDM
    for y in range(1, u.shape[1] - 1):
        for x in range(1, u.shape[0] - 1):
            next_u[x, y] = alpha * (u[x + 1, y] + u[x - 1, y]) + \
                        beta * (u[x, y + 1] + u[x, y - 1]) + \
                        (1 - 2 * alpha - 2 * beta) * u[x, y]

    return next_u


def main(_):
    """We solve the heat diffusion problem for a 2D plate with one hot edge and
    three cold edges"""

    # Let's print a summary of the options/parameters of this test.
    print(f"Diffusion coefficient used: {FLAGS.diff_coef}.")
    print(f"Duration of the simulation: {FLAGS.t_final}; Discretized in "
          f"{FLAGS.num_timeframes} timeframes.")
    print(f"Plate length: {FLAGS.plate_length}; Discretized in "
          f"{FLAGS.num_x_points} points along xx and "
          f"{FLAGS.num_y_points} points along yy")
    print(f"Number of plots: {FLAGS.num_plots}")
    print(f"Hot edge temperature: {FLAGS.hot_edge_temp}; Cold edge "
          f"temperature: {FLAGS.cold_edge_temp}; Initial points temperature: "
          f"{FLAGS.initial_temp}")
    print(f"Vectorization strategy: {FLAGS.vectorization_strategy}")
    print(f"The output is saved as {FLAGS.output_formats} in "
          f"{FLAGS.output_folder}.")

    # Process
    colorbar_limits = utils.flags.process_colorbar_limits_flag(
        FLAGS.colorbar_limits)

    # Compute spatial grid spacings, time step and relevant problem variables
    delta_t = compute_dimension_delta(FLAGS.t_final, FLAGS.num_timeframes)
    delta_x = compute_dimension_delta(FLAGS.plate_length, FLAGS.num_x_points)
    delta_y = compute_dimension_delta(FLAGS.plate_length, FLAGS.num_y_points)

    # Check and compute FDM stability constants for the instance provided
    alpha, beta = compute_fdm_stability_constants(FLAGS.diff_coef, delta_t,
                                                  delta_x, delta_y)
    if fdm_is_unstable(alpha, beta):
        raise Exception(
            f"Stability not guaranteed (alpha + beta > 0.5): alpha = {alpha},"
            f"beta = {beta}")
    print(f"Stability guaranteed: alpha = {alpha}; beta = {beta}")

    # Initialize plate
    u_initial = initialize_plate(FLAGS.num_x_points, FLAGS.num_y_points)
    u_initial = set_initial_conditions(u_initial, FLAGS.initial_temp)
    u_initial = set_canonical_boundary_conditions(u_initial,
                                                  FLAGS.hot_edge_temp,
                                                  FLAGS.cold_edge_temp)

    # Run FDM throughout time
    u_memorized_timeframes = run_time(u_initial, alpha, beta, delta_t,
                                      FLAGS.num_timeframes, FLAGS.num_plots,
                                      FLAGS.vectorization_strategy)

    # Generate new directory to save results
    current_directory = os.getcwd()
    results_folder_path = os.path.join(current_directory, FLAGS.output_folder)
    if not os.path.exists(results_folder_path):
        os.mkdir(results_folder_path)

    # Save results in the generated directory
    with open(os.path.join(results_folder_path, "fdm_results.pickle"),
              "wb") as f:
        pickle.dump(u_memorized_timeframes, f)

    # Plot results
    utils.plot.generate_output_across_time(
        u_memorized_timeframes,
        FLAGS.plate_length,
        FLAGS.diff_coef,
        FLAGS.output_formats,
        output_path=results_folder_path,
        colorbar_limits=colorbar_limits,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
