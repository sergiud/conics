
# conics - Python library for dealing with conics
#
# Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

from ._conic import Conic
from ._harker import fit_harker  # noqa: F401
from ._nievergelt import fit_nievergelt  # noqa: F401
from .geometry import hnormalized
from .geometry import line_intersection
from .geometry import line_through
import numpy as np


def fit_dlt(pts):
    """Fits an arbitrary conic using direct linear transform (DLT)
    :cite:`Hartley2004` to the specified 2-D coordinates given by `pts`.

    The resulting conic is not guaranteed to be of any specific type.

    :param pts: A set of 2-D coordinates to fit the conic to.
    :type pts: numpy.ndarray

    :return: The estimated conic.
    :rtype: conics.Conic
    """
    x, y = pts

    A = np.column_stack((x**2, x * y, y**2, x, y, np.ones_like(x)))

    u, s, vt = np.linalg.svd(A)

    return Conic(vt.T[:, -1])


def parabola_to_bezier(parabola, start, end):
    R"""Determines the control points of a quadratic Bezier curve that exactly
    represents given `parabola`.

    :param parabola: A parabola whose Bezier control points should be
        determined.
    :type parabola: conics.Parabola

    :param start: Starting 2-D coordinate on or around the curve from which the
        first control point is determined. The coordinate does not need to be
        lying exactly on the parabola. The method uses the coordinate to
        determine the shortest (orthogonal) distance contact point using
        :func:`conics.Parabola.contact`.
    :type start: numpy.ndarray

    :param end: Similar to the `start` parameter, denotes the outer point from
        which the final control point is determined.
    :type end: numpy.ndarray

    :return: A :math:`2\times3` matrix of whose columns denote the three control
        points of the Bezier curve.
    :rtype: numpy.ndarray

    :except ValueError: Thrown if the slopes on the outer contact points of the
        parabolic curve do not intersect. In this case, the parabola may be
        degenerate and correspond, e.g., to a straight line.
    """

    s1, s2 = parabola.contact(np.column_stack((start, end))).T
    C = parabola.to_conic()

    grad = C.gradient(np.column_stack((s1, s2)))

    # Normalize gradients
    grad /= np.linalg.norm(grad, axis=0)
    # Rotate vectors by 90 degrees by swapping the x/y coordinates and
    # multiplying y with -1
    grad = grad[::-1]
    grad[0] *= -1

    dxy1, dxy2 = grad.T
    # Start gradient should be facing the opposite direction of the second
    # gradient
    dxy1 *= -1

    l1 = line_through(s1, s1 + dxy1)
    l2 = line_through(s2, s2 + dxy2)

    inter = line_intersection(l1, l2)

    if np.isclose(inter[-1], 0):
        raise ValueError(
            'cannot construct a quadratic Bézier curve from the conic because the slopes at the contact points do not intersect')

    inter = hnormalized(inter)

    return np.column_stack((s1, inter, s2))
