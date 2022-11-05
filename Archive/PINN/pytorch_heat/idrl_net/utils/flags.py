"""Process the list flags to an appropriate format.

The function provided by the abseil library to pass list flags only supports
lists of strings. The functions in this library convert those lists into the
appropriate format.
"""

from collections import namedtuple
import logging


def process_temperature_ranges_flag(hot_edge_temp_range):
    """Checks validity of temperature range flags and converts to float tuple.

    Args:
        hot_edge_temp_range: Temperature range in which the PINN is trained

    Returns:
        Hot edge training temperature range in a proper format: tuple of floats.
    """

    if not len(hot_edge_temp_range) == 2:
        raise Exception("Temperature range format is incorrect. "
                        "(Correct format: [lower_bound, upper_bound]). ")

    # Build the holes_list variable in clear format
    hot_edge_temp_range_tuple = (float(hot_edge_temp_range[0]),
                                 float(hot_edge_temp_range[1]))

    if hot_edge_temp_range_tuple[0] > hot_edge_temp_range_tuple[1]:
        raise Exception(
            "Lowest training temperature can not be higher than highest one.")

    return hot_edge_temp_range_tuple


def check_hot_edge_temp_within_training_range(hot_edge_temp, temp_ranges):
    """Checks that hot edge temperature is within the training range

    Args:
        hot_edge_temp: Temperature of the hot edge in the plate to be plotted
        temp_ranges: Temperature range in which the PINN is trained

    Returns:
        Exception if hot_edge_temp is not within the interval temp_ranges
    """

    if temp_ranges[0] <= hot_edge_temp <= temp_ranges[1]:
        logging.info(
            "Hot edge temperature within the training interval selected")
    else:
        raise Exception(
            f"Hot edge temperature not within the interval selected for "
            f"training {temp_ranges}!")


def process_holes_flag(holes_list, plate_length):
    """Checks validity of holes flag and converts it to a list of named tuples.

    Args:
        holes_list: List of strings ["x1","y1","r1", ..., "xk","yk","rk"].
        plate_length: Length of the plate edge.

    Returns:
        List of named tuples with fields x_center, y_center and radius,
        [(x1, y1, r1), ..., (xk, yk, rk)]
    """

    # Check the length of the input
    if not len(holes_list) % 3 == 0:
        raise Exception(
            "Holes format is incorrect. (Correct format: [x1, y1, r1, ..., "
            "xk, yk, rk])")

    # Build the holes_list variable in clear format
    hole = namedtuple("hole", ["x_center", "y_center", "radius"])
    holes_list_processed = [
        hole(x_center=float(holes_list[3 * i]),
             y_center=float(holes_list[3 * i + 1]),
             radius=float(holes_list[3 * i + 2]))
        for i in range(int(len(holes_list) / 3))
    ]

    # Check the intersection of the hole with the domain
    if not all((hole.x_center + hole.radius > -plate_length / 2 and \
                hole.x_center - hole.radius < plate_length / 2 and \
                hole.y_center + hole.radius > -plate_length / 2 and \
                hole.y_center - hole.radius < plate_length / 2 \
                for hole in holes_list_processed)):
        raise Exception("Hole(s) out of domain")

    return holes_list_processed


def process_colorbar_limits_flag(flag_colorbar_limits):
    """Checks validity of colorbar limits flag and converts to a list of floats.

    Args:
        flag_colorbar_limits: List of strings.

    Returns:
        List of two floats (colorbar upper and lower limits, respectively).
    """
    if flag_colorbar_limits is None:
        return flag_colorbar_limits

    # Check the length of the input
    if not len(flag_colorbar_limits) == 2:
        raise Exception(
            "The colorbar limits can only have two entries (upper and lower"
            " limits for the colorbar).")

    return [float(flag_colorbar_limits[0]), float(flag_colorbar_limits[1])]
