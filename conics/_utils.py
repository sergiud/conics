import numpy as np


def polygon_area(points: np.ndarray) -> np.ndarray:
    """Compute the signed area of a polygon using Shoelace formula.

    Parameters
    ----------
    points : array_like (..., n, 2)
        Row major collections of n-point polygons.

    Returns
    -------
    areas: np.ndarray (...,)
        The signed area of all of the polygons.
    """
    rev = np.roll(points, -1, -2)
    cross = np.linalg.det(np.stack([points, rev], -1))
    return cross.sum(axis=-1) / 2
