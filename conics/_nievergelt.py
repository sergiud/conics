
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
from scipy.linalg import solve_triangular
import numpy as np
import scipy.optimize


def _phi_frac(l, s, y):
    R"""Computer the fractional parts of

    .. math::
        \frac{{\sigma}_{2}^{2} {y}_{2}^{2}}{\left(\lambda -
        {\sigma}_{2}^{2}\right)^{2}} + \frac{{\sigma}_{1}^{2}
        {y}_{1}^{2}}{\left(\lambda - {\sigma}_{1}^{2}\right)^{2}}
    """

    num = (s * np.ravel(y))[..., np.newaxis]
    den = np.atleast_2d(l).T - np.square(s)[..., np.newaxis]

    return num, den


def _phi(l, s, y):
    R"""Computes the function

    .. math::
        \frac{{\sigma}_{2}^{2} {y}_{2}^{2}}{\left(\lambda -
        {\sigma}_{2}^{2}\right)^{2}} + \frac{{\sigma}_{1}^{2}
        {y}_{1}^{2}}{\left(\lambda - {\sigma}_{1}^{2}\right)^{2}}-1
    """
    num, den = _phi_frac(l, s, y)
    return np.sum((num / den)**2, axis=0) - 1


def _phi_prime(l, s, y):
    R"""Computes the first-order derivative of the function

    .. math::
        - \frac{2 {\sigma}_{2}^{2} {y}_{2}^{2}}{\left(\lambda -
          {\sigma}_{2}^{2}\right)^{3}} - \frac{2 {\sigma}_{1}^{2}
          {y}_{1}^{2}}{\left(\lambda - {\sigma}_{1}^{2}\right)^{3}}
    """
    num, den = _phi_frac(l, s, y)
    return -2 * np.sum(num**2 / den**3, axis=0)


def _phi_prime2(l, s, y):
    R"""Computes the second-order derivative of the function

    .. math::
        6 \left(\frac{{\sigma}_{2}^{2} {y}_{2}^{2}}{\left(\lambda -
        {\sigma}_{2}^{2}\right)^{4}} + \frac{{\sigma}_{1}^{2}
        {y}_{1}^{2}}{\left(\lambda - {\sigma}_{1}^{2}\right)^{4}}\right)
    """
    num, den = _phi_frac(l, s, y)
    return 6 * np.sum(num**2 / den**4, axis=0)


def _secular(s, y):
    s1, s2 = s
    y1, y2 = y

    assert s2 <= s1

    a = s2 * y2
    b = np.abs(s1 * y1)
    diff = s2**2 - s1**2

    if np.isclose(a, 0):
        if b <= diff:
            l = s2**2
        else:
            l = s1**2 - b
    else:
        if b <= diff:
            l = s2**2 - np.abs(a)
        else:
            l1 = s2**2 - np.sqrt(2) * np.max([b, np.abs(a)])
            # l2 = s2*(s2-np.abs(y2))
            l2 = s2**2 - np.abs(s2 * y2)

            res = scipy.optimize.root_scalar(_phi, args=(s, y), method='halley', bracket=(l1, l2),
                                             x0=l2, x1=l2, fprime=_phi_prime, fprime2=_phi_prime2)
            l = res.root

    assert l < s2**2
    # print('a b diff', a, b, diff)
    # l = -0.102179e-5
    np.testing.assert_almost_equal(_phi(l, s, y), 0)

    return l


def _geodatic(G, p):
    u, s, vt = np.linalg.svd(G, full_matrices=False)
    y = u.T @ p

    l = _secular(s, y)
    s1, s2 = s

    if l == s2**2:  # np.isclose(l, s2**2):
        z = np.array([[0], [1]])
    else:
        z = s * np.squeeze(y) / (s**2 - l)
        z = z[..., np.newaxis]

    q = vt.T @ z

    np.testing.assert_almost_equal(np.linalg.norm(q), 1)

    return q


def fit_nievergelt(pts, type='parabola', scale=False):
    R"""Linear least-squares of specific type conics.

    The method implements the approach proposed in :cite:`Nievergelt2004`.

    :param pts: 2-D array of coordinates to fit the conic to.
    :type pts: numpy.ndarray

    :param type: The desired conic type. `None` if no specific conic type is
        desired, or `ellipse`, `parabola`, or `hyperbola`.
    :type type: str, None

    :param scale: Scale points to unit standard deviation along each axis.
        Scaling generally improves the numerical robustness of the fit
        :cite:`Harker2004`. The original method does not scale the points.
    :type scale: bool
    """
    mean = np.mean(pts, axis=1)
    centered = pts - mean[..., np.newaxis]

    if scale:
        std = np.sqrt(np.mean(centered**2, axis=1))[..., np.newaxis]

        if not np.isclose(std, np.zeros_like(std)).any():
            centered /= std
        else:
            # Disable scaling due to zeros
            scale = False
    else:
        std = None

    x, y = centered
    x2 = x**2
    y2 = y**2
    xy = np.prod(centered, axis=0).T

    M23 = centered.T
    M = np.column_stack((np.ones_like(x), M23, y2 - x2, 2 * xy, x2 + y2))

    # M1 = M[:, 0]

    s = np.linalg.svd(M23, full_matrices=False, compute_uv=False)
    # Singular values are sorted in descending order

    # Condition number max/min
    k22_inv = np.divide(*s[::-1])

    if np.isclose(k22_inv, 0):
        c = 0
        w = np.zeros((3, ), dtype=pts.dtype)

        # Fit TLS line
        A = np.column_stack((x, y))
        u, s, vt = np.linalg.svd(A)
        b = -vt.T[:2, -1, np.newaxis]

        A = np.zeros((2, 2), dtype=pts.dtype)
    elif np.reciprocal(k22_inv) > 0:
        M13 = M[:, :3]
        Q3, R3 = np.linalg.qr(M13, mode='complete')

        R = Q3.T @ M
        R22 = R[3:, 3:]

        if type == 'ellipse' or type == 'hyperbola':
            u, s, vt = np.linalg.svd(R22)
            q = vt.T[..., -1, np.newaxis]

        # Not specified type or type conic is parabolic:
        if type == 'parabola':
            G = R22[:, :2]

            p = -R22 @ np.array([[0], [0], [1]])
            q12 = _geodatic(G, p)
            q = np.vstack((q12, [1]))

        R11 = R[:3, :3]
        R12 = R[:3, 3:]

        sqrt2 = np.sqrt(2)

        Z = np.array([[-1, 0, 1],
                      [0, sqrt2, 0],
                      [1, 0, 1]])
        Z /= sqrt2

        c2b = solve_triangular(R11, -R12 @ q)
        w = np.reciprocal(sqrt2) * Z @ q

        a11, a12, a22 = w.ravel()
        a12 /= sqrt2

        b = c2b[-2:] / 2
        c = c2b[0]

        A = np.array([[a11, a12], [a12, a22]]) * 2

    Q = np.block([[A, b], [b.T, c]])

    print(Q)
    C = Conic.from_homogeneous(Q)

    if scale:
        sx, sy = std
        C = C.scale(sx, sy)

    return C.translate(mean)
