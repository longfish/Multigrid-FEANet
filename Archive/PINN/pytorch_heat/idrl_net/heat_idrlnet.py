"""Solves the 2D heat diffusion problem on a plate with holes using IDRLnet.

Here, we address the heat diffusion problem in a 2D square plate, where one
of the edges is kept at high temperature (hot edge) and the remaining three
are at a lower temperature (cold edges). Note that, at the beginning of the
simulation, the whole plate is at a given temperature (initial condition of
the problem). Moreover, we also explore the possiblities of the plate containing
holes inside and parametrizing the top/hot edge as an input to the PINN
considered.

As just mentioned, we use PINNs to solve the proposed problem. This method and
its results are extensively discussed in the README.md file of this directory.

Output:
Pickle file with solution of the heat equation, u(t, x, y), evaluated in a
regular spatial grid and at evenly spaced times, and its representation in a
figure or gif file.
"""

import os

from absl import app
from absl import logging
from absl import flags

import utils.idrlnet
import utils.flags
import utils.geometry
import utils.plot

import idrlnet.shortcut as sc

import sympy as sp

import pickle

FLAGS = flags.FLAGS

# Domain definition
flags.DEFINE_float("diff_coef", 0.1, "Diffusivity constant of the heat PDE.")
flags.DEFINE_float("plate_length",
                   1,
                   "Side length of the considered plate.",
                   lower_bound=0.)
flags.DEFINE_float("t_final", 1., "Duration of the simulation.", lower_bound=0.)
flags.DEFINE_list(
    "holes_list", [],
    "Circular holes coordinates as: [x1, y1, r1, ..., xk, yk, rk]")

# Domain temperatures
flags.DEFINE_list(
    "hot_edge_temp_range", ["1", "1"],
    "Range of hot edge temperatures in which the PINN is trained")
flags.DEFINE_float("hot_edge_temp_to_plot", 1,
                   "Temperature of the hot edge in the plate to be plotted.")
flags.DEFINE_float("cold_edge_temp", -1.,
                   "Temperature of the cold edge of the plate.")
flags.DEFINE_float("holes_temp", 1., "Temperature of the holes edges.")
flags.DEFINE_float("initial_temp", -1., "Temperature of the initial points.")

# Training Hyperparameters
flags.DEFINE_string(
    "pre_trained_folder", None,
    "Folder where a pre-trained networks can be found. The script continues the"
    " training of this network until a number FLAGS.max_iter of epochs.")
flags.DEFINE_integer("max_iter",
                     1000,
                     "Number of iterations in the PINN training.",
                     lower_bound=1)
flags.DEFINE_float(
    "sieve_tolerance",
    10**(-1),
    # less than this and sampler might miss the boundary (empirical
    # observation)
    "Tolerance used to filter out the sampled points that do not meet the "
    "precision criterion prescribed by this parameter")
flags.DEFINE_enum(
    "optimizer", "Adam", ["Adam", "SGD"],
    "Optimizer used for the NN training. The optimizer options are constrained "
    "to Adam and SGD for the sake of simplicity. Nevertheless, it is worth "
    "mentioning that, in practice, IDRLnet supports any optimizer from "
    "'torch.optim.Optimizer'.")
flags.DEFINE_float("learning_rate", 0.001,
                   "Learning rate used by the NN optimizer.")

# Domain densities
# The densities below are inputs required by IDRLnet that characterize domains.
# This library also uses a measure to characterize the length/area/volume of a
# domain (if it is a boundary or an interior domain). Therefore, the number
# of points sampled in a domain is given by: nr_points = density * measure.
flags.DEFINE_integer("density_collocation",
                     100000,
                     "Density of the sampled collocation points.",
                     lower_bound=1)
flags.DEFINE_integer("density_initial",
                     10000,
                     "Density of the sampled initial condition points.",
                     lower_bound=1)
flags.DEFINE_integer("density_cold_boundary",
                     10000,
                     "Density of the sampled cold boundary condition points.",
                     lower_bound=1)
flags.DEFINE_integer("density_hot_boundary",
                     10000,
                     "Density of the sampled hot boundary condition points.",
                     lower_bound=1)
flags.DEFINE_integer("density_holes_boundary",
                     100000,
                     "Density of the sampled points from the holes edges.",
                     lower_bound=1)

# Output variables
flags.DEFINE_integer(
    "output_num_x",
    100,
    "Number of points considered in the x_axis of the output grid where the "
    "trained PINN is evaluated",
    lower_bound=1)
flags.DEFINE_integer(
    "output_num_y",
    100,
    "Number of points considered in the y_axis of the output grid where the "
    "trained PINN is evaluated",
    lower_bound=1)
flags.DEFINE_integer("num_plots",
                     100,
                     "Number of plots to be obtained",
                     lower_bound=1)
flags.DEFINE_string("output_folder", "idrlnet_run",
                    "Name of the directory where the output is stored.")
flags.DEFINE_list(
    "output_formats", ["figure", "gif"], "Defines the formats in which the "
    "output is obtained. Only two formats are available: gif and figure. "
    "If 'figure' is in the list provided, it outputs a figure with several "
    "subplots at different timesteps. If 'gif' is in the list provided, it "
    "outputs a gif with several plots at different timesteps.")
flags.DEFINE_list(
    "colorbar_limits", None, "Limits of the colorbar present in "
    "the inference output. When set to 'None' (default), those "
    "limits are adjusted automatically.")


def main(_):
    """We solve the heat diffusion problem for a 2D plate with a hole """

    # Let's print a summary of the options/parameters of this test.
    print(f"Diffusion coefficient used: {FLAGS.diff_coef}.")
    print(f"Duration of the simulation: {FLAGS.t_final}, from which we "
          f"observe {FLAGS.num_plots} evenly spaced timeframes")
    print(f"Plate length: {FLAGS.plate_length}, from which we observe "
          f"{FLAGS.output_num_x} points along xx and {FLAGS.output_num_y} "
          f"points along yy.")
    print(f"Holes coordinates: {FLAGS.holes_list}")
    print(f"Sieve tolerance: {FLAGS.sieve_tolerance}")
    print(f"Optimizer used: {FLAGS.optimizer}, with learning rate "
          f"{FLAGS.learning_rate}")
    print(f"Parametrized boundary conditions: Hot edge temperature range, "
          f"{FLAGS.hot_edge_temp_range}")
    print(f"Hot edge temperature to plot: {FLAGS.hot_edge_temp_to_plot}; Cold "
          f"edge temperature: {FLAGS.cold_edge_temp}; Holes edge temperatures "
          f"{FLAGS.holes_temp}; Initial points temperature: "
          f"{FLAGS.initial_temp}")
    print(f"Maximum number of training iterations: {FLAGS.max_iter}")
    print(f"Density of collocation points used: {FLAGS.density_collocation}")
    print(f"Density of initial condition points used: {FLAGS.density_initial}")
    print(f"Density of cold boundary points used: "
          f"{FLAGS.density_cold_boundary}")
    print(f"Density of hot boundary points used: {FLAGS.density_hot_boundary}")
    print(f"The output is saved as {FLAGS.output_formats} in "
          f"{FLAGS.output_folder}. Colorbar limits: {FLAGS.colorbar_limits}.")
    print(f"Use pre-trained network in {FLAGS.pre_trained_folder} as starting "
          f"point.")

    # Flag checking
    hot_bc_range_tuple = utils.flags.process_temperature_ranges_flag(
        FLAGS.hot_edge_temp_range)
    utils.flags.check_hot_edge_temp_within_training_range(
        FLAGS.hot_edge_temp_to_plot, hot_bc_range_tuple)
    holes_list = utils.flags.process_holes_flag(FLAGS.holes_list,
                                                FLAGS.plate_length)
    colorbar_limits = utils.flags.process_colorbar_limits_flag(
        FLAGS.colorbar_limits)

    # Define variables of the problem
    x, y, t, hot_bc = sp.symbols("x y t hot_bc")
    parameters_ranges = {t: (.0, FLAGS.t_final), hot_bc: hot_bc_range_tuple}

    # Geometry of the problem
    geo = utils.geometry.generate_geometry(FLAGS.plate_length, holes_list)

    # Generate domain samplers
    interior_sampler = utils.idrlnet.generate_interior_sampler(
        geo, parameters_ranges, FLAGS.density_collocation)
    initial_sampler = utils.idrlnet.generate_initial_sampler(
        geo, t, parameters_ranges, FLAGS.density_initial, FLAGS.initial_temp)
    cold_boundary_sampler = utils.idrlnet.generate_cold_boundary_sampler(
        geo, x, y, parameters_ranges, FLAGS.density_cold_boundary,
        FLAGS.cold_edge_temp, FLAGS.plate_length, FLAGS.sieve_tolerance)
    hot_boundary_sampler = utils.idrlnet.generate_hot_boundary_sampler(
        geo, x, y, parameters_ranges, FLAGS.density_hot_boundary,
        FLAGS.plate_length, FLAGS.sieve_tolerance)
    holes_boundary_sampler = utils.idrlnet.generate_holes_boundary_sampler(
        geo, x, y, parameters_ranges, FLAGS.density_holes_boundary,
        FLAGS.holes_temp, FLAGS.plate_length, holes_list)

    # Generating the DataNodes
    training_datanodes = utils.idrlnet.generate_idrlnet_datanodes(
        interior_sampler, initial_sampler, cold_boundary_sampler,
        hot_boundary_sampler, holes_boundary_sampler)

    # Generating the NetNodes
    net_node = sc.get_net_node(inputs=("x", "y", "t", "hot_bc"),
                               outputs=("u",),
                               name="pinn",
                               arch=sc.Arch.mlp,
                               **{"seq": [4, 20, 20, 20, 20, 20, 1]})
    # The configuration of the NN architecture can be specified as the
    # following example shows:
    # https://idrlnet.readthedocs.io/en/latest/user/get_started/1_simple_poisson.html#solving-simple-poisson-equation

    # Generating the PDENodes
    # General procedure to define PDEs:
    # https://idrlnet.readthedocs.io/en/latest/user/get_started/2_euler_beam.html#eulerbernoulli-beam
    # The DiffusionNode function instantiates the PDENode with the diffusion
    # PDE:
    # u_t = \nabla . (D * \nabla . u) + Q, where \nabla denotes the del
    # differential operator.
    # Details can be found at:
    # https://en.wikipedia.org/wiki/Diffusion_equation
    # Here we consider Q = 0 and D constant, obtaining the heat diffusion PDE:
    # u_t = D * \nabla^2 . u
    pde_node = sc.DiffusionNode(T="u",
                                D=FLAGS.diff_coef,
                                Q=0.0,
                                dim=2,
                                time=True)
    # time is True by default, but just to keep in mind that it defines if
    # we want to consider the temporal derivative in the equation
    # dim refers to the spatial dimensions (thus, the number of terms that
    # are second derivatives wrt to those coordinates to be considered)

    # Generate path to pre-trained folder
    current_directory = os.getcwd()
    if FLAGS.pre_trained_folder:
        pre_trained_path = [
            os.path.join(current_directory, FLAGS.pre_trained_folder,
                         "network_dir")
        ]
    else:
        pre_trained_path = []

    # Generate output path and respective folder
    output_folder_path = os.path.join(current_directory, FLAGS.output_folder)
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # Train PINN
    trained_pinn = utils.idrlnet.train_pinn(training_datanodes, net_node,
                                            pde_node, pre_trained_path,
                                            output_folder_path, FLAGS.max_iter,
                                            FLAGS.optimizer,
                                            FLAGS.learning_rate)

    # Inference
    inference_sampler = utils.idrlnet.generate_grid_sampler(
        FLAGS.plate_length, FLAGS.output_num_x, FLAGS.output_num_y,
        FLAGS.num_plots, FLAGS.t_final, FLAGS.hot_edge_temp_to_plot)
    inference_datanode = utils.idrlnet.generate_idrlnet_datanodes(
        inference_sampler)
    u_inferred_timeframes = utils.idrlnet.infer_pinn(trained_pinn,
                                                     inference_datanode,
                                                     FLAGS.output_num_x,
                                                     FLAGS.output_num_y,
                                                     FLAGS.num_plots)

    # Generate results sub-folder inside the output one
    results_folder_path = os.path.join(output_folder_path, "results")
    if not os.path.exists(results_folder_path):
        os.mkdir(results_folder_path)

    # Plot results
    utils.plot.generate_output_across_time(
        u_inferred_timeframes,
        FLAGS.plate_length,
        FLAGS.diff_coef,
        FLAGS.output_formats,
        holes_list=holes_list,
        holes_temperature=FLAGS.holes_temp,
        output_path=results_folder_path,
        colorbar_limits=colorbar_limits,
    )

    # Save metadata
    metadata = {
        "plate_length": FLAGS.plate_length,
        "diff_coef": FLAGS.diff_coef,
    }
    with open(os.path.join(results_folder_path, "data.pickle"), "wb") as f:
        pickle.dump((u_inferred_timeframes, metadata), f)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
