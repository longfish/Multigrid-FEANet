"""IDRLnet interface functions"""

import idrlnet.shortcut as sc

from functools import partial

import utils.idrlnet_samplers
import utils.plot

import shutil


def generate_interior_sampler(geo, parameters_range, density_collocation):
    """Generates the interior domain sampler used to train the model

    Args:
        geo: Object of type geo.
        parameters_range: Range of variable t and hot_bc.
        density_collocation: Sampling density of collocation points.

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    return {
        "interior":
            partial(utils.idrlnet_samplers.sample_interior_domain, geo,
                    parameters_range, density_collocation)
    }


def generate_initial_sampler(geo, t, parameters_range, density_initial,
                             initial_temp):
    """Generates the initial domain sampler used to train the model

    Args:
        geo: Object of type geo.
        t: Temporal variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        density_collocation: Sampling density of collocation points.
        density_initial: Sampling density of initial points.
        initial_temp: Temperature of initial points.

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    return {
        "initial":
            partial(utils.idrlnet_samplers.sample_initial_domain, geo, t,
                    parameters_range, initial_temp, density_initial)
    }


def generate_cold_boundary_sampler(geo, x, y, parameters_range,
                                   density_cold_boundary, cold_edge_temp,
                                   plate_length, sieve_tolerance):
    """Generates the cold boundary domain sampler used to train the model

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        density_cold_boundary: Sampling density of cold boundary points.
        cold_edge_temp: Temperature of cold boundaries points.
        plate_length: Length of the plate edge.
        sieve_tolerance: Tolerance to filter points from each boundary.

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    return {
        "cold_boundary":
            partial(utils.idrlnet_samplers.sample_cold_boundary_domain, geo, x,
                    y, parameters_range, density_cold_boundary, cold_edge_temp,
                    plate_length, sieve_tolerance)
    }


def generate_hot_boundary_sampler(geo, x, y, parameters_range,
                                  density_hot_boundary, plate_length,
                                  sieve_tolerance):
    """Generates the hot boundary domain sampler used to train the model

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        density_hot_boundary: Sampling density of hot boundary points.
        plate_length: Length of the plate edge.
        sieve_tolerance: Tolerance to filter points from each boundary.

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    return {
        "hot_boundary":
            partial(utils.idrlnet_samplers.sample_hot_boundary_domain, geo, x,
                    y, parameters_range, density_hot_boundary, plate_length,
                    sieve_tolerance)
    }


def generate_holes_boundary_sampler(geo, x, y, parameters_range,
                                    density_holes_boundary, holes_temp,
                                    plate_length, holes_list):
    """Generates the holes boundary domain sampler used to train the model

    Args:
        geo: Object of type geo.
        x: Spatial variable of the problem.
        y: Spatial variable of the problem.
        parameters_range: Range of variable t and hot_bc.
        density_holes_boundary: Sampling density of hole(s) boundary points.
        holes_temp: Temperature of hole(s) boundary points.
        plate_length: Length of the plate edge.
        holes_exist: Boolean that indicates if there are holes prescribed.

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    holes_boundary_sampler = {}
    if holes_list:
        holes_boundary_sampler["holes_boundary"] = partial(
            utils.idrlnet_samplers.sample_holes_boundary_domain, geo, x, y,
            parameters_range, density_holes_boundary, holes_temp, plate_length)

    return holes_boundary_sampler


def generate_grid_sampler(plate_length, output_num_x, output_num_y, num_plots,
                          t_final, hot_edge_temp_plot):
    """Generates a sampler that covers the whole domain with a grid

    Args:
        plate_length: Length of the plate edge
        output_num_x: Number of points along the x-axis in the grid
        output_num_y: Number of points along the y-axis in the grid
        num_plots: Number of (evenly spaced) timeframes to be considered
        t_final: Final instant of time to be considered
        hot_edge_temp_plot: Temperature of the hot edge to be considered

    Returns:
        Dictionary where domain sampler (value) is associated to the intended
        datanode name (key).
    """

    return {
        "grid":
            partial(utils.idrlnet_samplers.sample_inference_domain,
                    plate_length, output_num_x, output_num_y, num_plots,
                    t_final, hot_edge_temp_plot)
    }


def generate_idrlnet_datanodes(*domain_samplers):
    """Generates the datanodes for this problem from the samplers provided

    Args:
        domain_samplers: arbitrary number of dictionaries where to each datanode
        name (key) is associated a domain sampler (value)

    Returns:
        Datanode tuple with as many entries as domain_samplers provided.
    """
    datanodes = []
    for domain_sampler_dict in domain_samplers:
        if domain_sampler_dict:  # empty dicts evaluate to False
            domain_identifier = next(iter(domain_sampler_dict.keys()))
            domain_sampler = domain_sampler_dict[domain_identifier]
            datanodes.append(
                (sc.datanode(domain_sampler,
                             name="datanode_" + domain_identifier)()))

    return tuple(datanodes)


def train_pinn(data_nodes, net_node, pde_node, pre_trained_path, output_path,
               max_iter, optimizer_name, learning_rate):
    """Trains the PINN on the datanodes to solve the provided PDE.

    Args:
        data_nodes: DataNodes with the training data set.
        net_node: NetNode with the NN to be trained.
        pde_node: PDENode with the PDE to be solver.
        max_iter: Maximum number of training epochs.

    Returns:
        A trained PINN.
    """
    model = sc.Solver(
        sample_domains=(data_nodes),
        netnodes=[net_node],
        pdes=[pde_node],
        max_iter=max_iter,
        init_network_dirs=pre_trained_path,
        network_dir=output_path + "/network_dir",
        summary_dir=output_path + "/network_dir",
        result_dir=output_path + "/train_domain",
        opt_config=dict(optimizer=optimizer_name, lr=learning_rate),
    )
    model.solve()

    # Save model (generate "model.ckpt" in network_dir)
    model.save()

    # Move automatically generated file "train.log" to the proper folder
    shutil.move("./train.log", output_path + "/train.log")

    return model


def infer_pinn(model, inference_datanode, output_num_x, output_num_y,
               num_plots):
    """Uses a model to infer results on the inference datanode

    Args:
        model: Model used to infer values in the domain (e.g. a trained PINN)
        inference_datanode: IDRLnet datanode where we are going to infer
        output_num_x: Number of points in the x-axis of the output grid
        output_num_y: Number of points in the y_axis of the output grid
        num_plots: Number of (evenly spaced) timeframes in which we infer values

    Returns:
        Inferred values with model in the inference datanode. Output format:
        [[t_0, u_0], ..., [t_k, u_k]], where t_i is the time respective
        to u_i, which is a grid with the inferred values across the covered
        domain
    """

    model.sample_domains = inference_datanode
    model_solution = model.infer_step({"datanode_grid": ["x", "y", "t", "u"]})
    u_inferred_timeframes = solver2grid(model_solution, output_num_x,
                                        output_num_y, num_plots)

    return u_inferred_timeframes


def solver2grid(model_solution, output_num_x, output_num_y, num_plots):
    """Rearranges solution from solver to a grid shape.

    Args:
        model_solution: inferred solution in the IDRLnet PINN format.
        output_num_x: Number of points in the x-axis discretization.
        output_num_y: Number of points in the y-axis discretization.
        num_plots: Number of plots to be obtained.

    Returns:
        Inferred solution in a grid format associated to the respective time
        instant ([[t_0, u_0], ..., [t_k, u_k]], where t_i is the time respective
        to u_i, which is a grid with the inferred values at the given points)
    """
    u_pred = torch2numpy(model_solution["datanode_grid"]["u"])
    u_layers_to_plot = u_pred.reshape((output_num_x, output_num_y, num_plots),
                                      order="C")
    t_pred = sorted(set(torch2numpy(model_solution["datanode_grid"]["t"])))
    u_inferred_timeframes = [
        [t_pred[idx], u_layers_to_plot[:, :, idx].T] for idx in range(num_plots)
    ]
    return u_inferred_timeframes


def torch2numpy(x):
    """Converts a Torch tensor to a numpy array.

    Args:
        x: Torch tensor.

    Returns:
        Numpy array corresponding to the provided tensor.
    """

    return x.cpu().detach().numpy().ravel()
