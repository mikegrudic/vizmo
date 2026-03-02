import numpy as np


def coordinate_grid(res: int, size: float, center: np.ndarray) -> tuple:
    """
    Instantiates a coordinate grid encoded as 2 NxN arrays representing the X and Y axis respectively, whose points
    are the centers of the grid points, which correspond to the abcissa of a rendered map.

    Initializes a grid from map arguments:

    size: side-length of the grid
    center: center of the grid
    res: number of grid cells per dimension

    These may be passed as keyword arguments.
    """

    cell_size = size / res
    X = Y = np.linspace(-0.5 * size + 0.5 * cell_size, 0.5 * size - 0.5 * cell_size, res)
    X, Y = np.meshgrid(X, Y, indexing="ij")
    X += center[0]
    Y += center[1]
    return X, Y


def coordinate_grid_from_mapargs(mapargs):
    return coordinate_grid(*mapargs)
