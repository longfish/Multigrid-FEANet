"""Generation of geometric objects from IDRLnet library"""

import idrlnet.shortcut as sc


def generate_geometry(plate_length, holes_list):
    """Generates the geometry of the problem.

    Args:
        plate_length: Length of the plate edge.
        holes_list: List of holes named tuples, [(x1, y1, r1), ..., (xk, yk,
        rk)].

    Returns:
        Object of type geo, containing the plate with holes.
    """
    plate = sc.Rectangle((-plate_length / 2, -plate_length / 2),
                         (plate_length / 2, plate_length / 2))
    for hole in holes_list:
        plate -= sc.Circle((hole.x_center, hole.y_center), hole.radius)
    return plate
