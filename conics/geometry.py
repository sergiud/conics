# conics - Python library for dealing with conics
#
# Copyright 2025 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numpy.typing as npt


def hnormalized(p: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Projects points from an :math:`n`-dimensional space onto
    :math:`n-1`-dimensional one.

    Parameters
    ----------
    p: numpy.ndarray (n, m)
        The :math:`n`-dimensional points to be projected stored in :math:`m`
        columns.

    Returns
    -------
    numpy.ndarray (n-1, m):
        The projected points.
    """
    return p[:-1] / p[-1]


def homogeneous(p: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Homogenizes :math:`n`-dimensional points by appendind all ones to the
    last dimension.

    Parameters
    ----------
    p: numpy.ndarray (n, m)
        The :math:`n`-dimensional points to be homogenized stored in :math:`m`
        columns.

    Returns
    -------
    numpy.ndarray (n+1, m)
        The homogenized points.
    """
    p = np.atleast_2d(p).T
    return np.vstack((p, np.ones_like(p[0])))


def line_through(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Constructs a homogeneous lines from 2-D points.

    Parameters
    ----------
    a: array_like (2, )
        The start point of the line segment.
    b: array_like (2, )
        The end point of the line segment.

    Returns
    -------
    numpy.ndarray:
        The homogeneous line that connects `a` and `b`.
    """
    start = np.append(a, 1)
    end = np.append(b, 1)

    return np.cross(start, end)


def line_intersection(l1: npt.ArrayLike, l2: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Computes the intersection between two homogeneous lines.

    Parameters
    ----------
    l1: array_like (3, )
        The first line.
    l2: array_like (3, )
        The second line.

    Returns
    -------
    numpy.ndarray:
        The homogeneous intersection point.
    """
    return np.cross(l1, l2)


def rot2d(alpha: float) -> npt.NDArray[np.float64]:
    """Constructs a 2-D rotation matrix given a specified angle.

    Parameters
    ----------
    alpha: float
        The rotation angle in the plane, in radians.

    Returns
    -------
    numpy.ndarray (2, 2):
        The 2-D rotation matrix.
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def projectively_unique(p: npt.ArrayLike, atol: float = 1e-4) -> np.ndarray:
    """Determines unique points in the projective space given a set of points.

    Parameters
    ----------
    p: array_like (n, m)
        The possibly non-unique but equivalent up to scale set
        :math:`d`-dimensional points stored in :math:`m` columns.
    atol: float
        The absolute comparison tolerance with respect to the norm of the
        cross-product of each point pair. The corresponding norm must be (close
        to) zero for points that are equivalent up to scale. The tolerance
        accounts for the corresponding round-off error.

    Returns
    -------
    numpy.ndarray:
        The subset of points that are unique in projective space.
    """
    p = np.asarray(p)
    n = p.shape[-1]
    # The indices of the strictly upper-triangular part of the adjacency
    # matrix
    i, j = np.triu_indices(n, k=1)

    # Pairwise cross-product between each intersection for filtering out
    # multiplicative multiples
    c = np.cross(p.T[i], p.T[j])

    # Compute the norm of the cross-products which is zero for a pair of
    # intersections that are equivalent up to scale
    d = np.linalg.norm(c, axis=-1)
    m = np.isclose(d, 0, atol=atol)

    # mark points as duplicate if cross product with an earlier point has 0 norm
    duplicate = np.bincount(j, m) > 0
    return p[:, ~duplicate]
