
# conics - Python library for dealing with conics
#
# Copyright 2020 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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
from ._harker import fit_harker
from ._nievergelt import fit_nievergelt
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

    A = np.column_stack((x**2, x*y, y**2, x, y, np.ones_like(x)))

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


if __name__ == '__main__':
    from . import Parabola
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    x = [-7, -3, 0, 0, 1, 1]
    y = [9, 5, 4, 8, 3, 5]

    #x = [-6.6, -2.8, -0.2, 0.4, 1.2, 1.4]
    #y = [8.8, 5.4, 3.6, 7.8, 3.4, 4.8]

    y = [-2, -0.1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    x = [1, 0.5, 0.1, 0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.6]

    #x = [-1, 2, 5, 10, -4]
    #y = [1, -2, 3, -4, -3]

    x = [-1.1, 0, 0, 0, -0.5]
    y = [0, 1, 2, 3, 4]

    #x = np.array([-4, -2, -1, 0, 1, 2, 4])
    # y = 2*x**2+1*x+5  # +0.25*x**2+0
    #y = np.array([-4, -2, -1, 0, 1, 2, 4])
    # x = -2 * y**2 - 0.5  # +0.25*x**2+0
    #x = [1, 0, 3, 4, 5]
    #y = [3, -1, 10, 50, 100]

    #x = [-34.75, -22, -15.5, -8.0, -4, -1.0, 1.5, 4.5, 9.25, 17, 23.5, 36, 64.5]
    #y = [20.25, 17, 15.0, 13.5, 13, 12.5, 12.5, 13.0, 14.00, 16, 18.0, 21, 29.5]

    pts = np.row_stack((x, y)).astype(np.float)
    #pts[1] *= 1e-2
    #C = fit_nievergelt(pts, type='hyperbola')
    C = fit_nievergelt(pts, type='parabola', scale=True)

    C = C.constrain(pts, fix_angle=np.pi/4)

    vertex, p, alpha = C.to_parabola()
    pb = Parabola(vertex, p, alpha)
    pb1 = pb.refine(pts)

    C = pb1.to_conic()
    vertex, p, alpha = C.to_parabola()

    # plt.figure()
    # plt.axis('equal')

    #R = rot2d(alpha)
    #x = np.linspace(-10, 10)
    #y2 = 2*p*x
    #y = np.sqrt(y2)

    #x, y = R @ np.row_stack((x, y)) + vertex[..., np.newaxis]
    #plt.plot(x, y)

    # plt.show()

    plt.figure()
    plt.axis('equal')

    x, y = pts

    X, Y = np.meshgrid(np.linspace(np.min(x) - 1, np.max(x) + 1),
                       np.linspace(-1 + np.min(y), np.max(y) + 1))
    Z = C(np.row_stack((X.ravel(), Y.ravel())))

    plt.contour(X, Y, Z.reshape(X.shape), levels=0)
    plt.scatter(x, y)
    plt.scatter(*vertex)

    pb = Parabola(vertex, p, alpha)

    s1, inter, s2 = parabola_to_bezier(pb, *pts[:, [0, -1]].T).T

    pp1 = np.column_stack((s1, inter))
    pp2 = np.column_stack((s2, inter))

    plt.plot(pp1[0], pp1[1])
    plt.plot(pp2[0], pp2[1])

    path = mpatches.Path(np.row_stack((s1.T, inter, s2.T)), [mpatches.Path.MOVETO,
                                                             mpatches.Path.CURVE3, mpatches.Path.CURVE3])
    pp = mpatches.PathPatch(
        path,
        linestyle='--',
        edgecolor='blue',
        facecolor='none',
        lw=3)

    plt.gca().add_artist(pp)

    plt.scatter(*inter)
    plt.scatter(*np.column_stack((s1, s2)))

    print(C(s1))
    print(C(s2))

    plt.show()
