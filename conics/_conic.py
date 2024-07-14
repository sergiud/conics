
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

from .geometry import rot2d
from scipy.optimize import least_squares
import itertools
import numpy as np
import scipy.linalg
import warnings


def _make_circle(x0, r):
    x0 = np.reshape(x0, (2, 1))
    C = np.block([[np.eye(2), -x0],
                  [-x0.T, x0.T @ x0 - r**2]])
    return C


def icp(A1, A2):
    e = scipy.linalg.eig(A1, A2, left=False, right=False)
    k = np.argmax(np.abs(np.median(e) - e))
    u, s, vt = np.linalg.svd(e[k] * np.linalg.inv(A1) - np.linalg.inv(A2))

    u = u[..., :2]
    s = s[:2]

    print(s.shape)

    values1 = u @ (np.sqrt(s) * np.array([1, +1j]))
    values2 = u @ (np.sqrt(s) * np.array([1, -1j]))

    print(values1)
    print(values2)

# The Common Self-polar Triangle of Concentric Circles and Its Application to
# Camera Calibration


def concentric_conics_vanishing_line(C1, C2):
    C = np.linalg.solve(C2, C1)
    evals, evecs = np.linalg.eig(C)

    evals = evals[::-1]
    evecs = evecs[..., ::-1]

    x1, x2, x3 = evecs.T
    v = np.cross(x2, x3)

    return x1, v


def g2a(x0, major_minor, alpha):
    if np.less(*major_minor):
        warnings.warn('ellipse major axis size must be larger or equal to the minor one. however, the provided major axis is smaller than the minor axis. this may cause an unintentional change of ellipise orientation', UserWarning)

    x0 = np.asarray(x0)
    R = rot2d(alpha)
    M = R @ np.diag(np.reciprocal(np.square(major_minor), dtype=R.dtype)) @ R.T
    a, b, c = np.array([1, 2, 1]) * M[np.triu_indices_from(M)]
    d, e = -2 * M @ x0
    f = x0.T @ M @ x0 - 1

    return np.vstack((a, b, c, d, e, f))


def a2g(x0, C33, f):
    factor = x0.T @ C33 @ x0 - f

    evals, evecs = np.linalg.eigh(C33)
    # TODO sqrt argument may be negative
    val = factor * np.reciprocal(evals)
    # print(val, evals)
    major_minor = np.sqrt(val)

    # TODO viz generates division by zero warning
    # alpha = np.arctan(np.divide(*evecs[::-1, 0]))
    alpha = np.arctan2(*evecs[::-1, 0]) % np.pi

    return x0, major_minor, alpha


def bracket(A, B, C):
    return np.linalg.det(np.stack((A[:, 0], B[:, 1], C[:, 2]), axis=1))


def cofactor(A):
    n, m = A.shape

    C = np.empty_like(A)

    for i in range(n):
        for j in range(m):
            minors = np.delete(np.delete(A, i, axis=0), j, axis=1)
            C[i, j] = np.linalg.det(minors)

    return C


def adjugate(C):
    det = np.linalg.det(C)

    if det == 0:
        return np.transpose(cofactor(C))

    return np.linalg.inv(C).T * det


def skew_symmetric(C):
    a, b, c = C
    return np.array([[0, c, -b], [-c, 0, a], [b, -a, 0]])


def surface_normal(Q, r=1):
    evals, evecs = np.linalg.eigh(Q)
    idxs = np.argsort(evals)[::-1]

    # Swap l1 and l2 order
    idxs[:2] = idxs[:2][::-1]

    evals = evals[idxs]
    evecs = evecs[:, idxs]

    lambda1, lambda2, lambda3 = evals
    u1, u2, u3 = evecs.T

    n = np.sqrt(lambda2 - lambda1) * u2 + np.sqrt(lambda1 - lambda3) * u3
    # Normalize to unit L2 norm
    n /= np.linalg.norm(n)
    h = np.sqrt(lambda1**3) * r

    return n, h


def projected_center(Q, n):
    """Provides the projected center of the circle in the camera coordinate
    system."""
    return np.linalg.solve(Q, n)


def estimate_pose(Q, r, alpha):
    r"""Estimates the 5-D camera pose with respect to the supporting plane of a
    circle projection :cite:`Chen2004`.

    :param Q: :math:`3\times3` symmetric matrix that defines the oblique cone
        given by  the rays passing through the center of the camera and the
        circle projection.
    :type Q: numpy.ndarray

    :param r: The radius of the projected circle.
    :type r: float

    :param alpha: The orientation of the projected circle, in radians.
    :type alpha: float

    :return: Eight possible solutions :math:`(R_i,\vec t_i,\vec n_i,\vec
        c,s_{1,i},s_{2,i},s_{3,i},m_i)`, :math:`i=1,\dotsc,8` that describe the
        pose of the camera observing the circle projection and its supporting
        plane.

        :math:`R_i\in\mathsf{SO}(3)`
            The camera rotation.

        :math:`t_i\in\mathbb{R}^3`
            The camera translation.

        :math:`n_i\in\mathbb{R}^3`
            The normal vector to the support plane of the circle projection.

        :math:`s_{1,i},s_{2,i},s_{3,i}\in\{\pm1\}`
            The signs of the possible solutions.

        :math:`m_i\in\{0,1\}`
            A mask that defines which solutions are valid.

    :rtype: tuple
    """

    evals, evecs = np.linalg.eigh(Q)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    lambda1, lambda2, lambda3 = evals

    assert lambda1 * lambda2 > 0
    assert lambda1 * lambda3 < 0
    assert np.abs(lambda1) >= np.abs(lambda2)

    den13 = lambda1 - lambda3
    g = np.sqrt((lambda2 - lambda3) / den13)
    h = np.sqrt((lambda1 - lambda2) / den13)

    # 2^3 possibilities to arrange [+1,-1] in three positions
    s1, s2, s3 = np.array(
        list(itertools.product(*itertools.repeat([+1, -1], 3)))).T
    alpha = alpha * np.ones_like(s1)

    den = np.sqrt(-lambda1 * lambda3)
    z0 = s3 * lambda2 * r / den

    right = np.array([[s2 * h], [np.zeros_like(s1)], [-s1 * g]])

    # Batch multiplication + move num axis from the end to the front
    # Basically,
    # n = np.einsum('ij,jlk->kil', evecs, right)
    n = evecs @ np.moveaxis(right, -1, 0)

    # Expand z0 dimensions to enable correct broadcasting for the multiplication with a scalar.
    left = np.array(
        [[lambda3 / lambda2], [0], [lambda1 / lambda2]])[..., np.newaxis]

    c = z0[..., np.newaxis, np.newaxis] * \
        evecs @ np.moveaxis(left * right, -1, 0)

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    R = evecs @ np.moveaxis(np.array([[g * cos_a, s1 * g * sin_a, s2 * h],
                                      [sin_a, -s1 * cos_a, np.zeros_like(s1)],
                                      [s1 * s2 * h * cos_a, s2 * h * sin_a, -s1 * g]]), -1, 0)

    factor = np.sqrt((lambda1 - lambda2) * (lambda2 - lambda3)) / lambda2
    t = np.array([[-s2 * factor * cos_a],
                  [-s1 * s2 * factor * sin_a],
                  [np.ones_like(s1)]]) * z0
    t = np.moveaxis(t, -1, 0)
    mask = np.ravel((n[:, -1, :] > 0) & (c[:, -1, :] < 0))

    return R, t, n, c, s1, s2, s3, mask


class Conic:
    """Initializes the conic using the given coefficents.

    :param args: Coeffcients of the quadratic curve.
    :type args: array-like, optional
    """

    def __init__(self, *args):
        if not (len(args) == 0 or (len(args) == 1 and np.size(*args) == 6) or len(args) == 6):
            raise ValueError(
                'unexpected number of arguments; expected 0, 1 or 6 arguments but got {}'.format(len(args)))

        self.coeffs_ = np.ravel(args)

    @property
    def __C33(self):
        a, b, c, d, e, f = self.coeffs_
        half_b = b / 2
        return np.block([[a, half_b], [half_b, c]])

    def __center(self, C33):
        d, e = self.coeffs_[-3:-1]
        return -np.linalg.solve(C33, np.vstack((d, e)) / 2)

    @property
    def center(self):
        """Returns the midpoint of a central conic :cite:`Ayoub1993`.

        :return: 2-D coordinate of the conic center.
        :rtype: numpy.ndarray
        """
        return self.__center(self.__C33)

    @property
    def homogeneous(self):
        r"""Returns the homogeneous :math:`3\times3` symmetric matrix
        that represents the conic section. The matrix is given by

        .. math::

            \tilde{Q} =
            \begin{bmatrix}
              A & B/2 & D/2
              \\
              B/2 & C & E/2
              \\
              D/2 & E/2 & F
            \end{bmatrix}

        For every point :math:`\vec p=(x,y,1)^\top` the quadratic curve can then
        be expressed by

        .. math::

            \vec p^\top \tilde{Q} \vec p=0

        """
        A, B, C, D, E, F = self.coeffs_ / Conic.__factors()

        return np.array([[A, B, D],
                         [B, C, E],
                         [D, E, F]])

    def intersect(self, other):
        r"""Computes the intersections of `self` with another conic.

        The method implements the algorithm introduced in
        :cite:`RichterGebert2011`.

        :param other: The conic for which the intersections are to be computed.
        :type other: Conic

        :return: A matrix of homogeneous 2-D points stored in column vectors of
            a :math:`3\times N` matrix consisting of :math:`0\leq N\leq 4`
            columns.
        :rtype: numpy.ndarray

        .. plot:: ../examples/intersections.py
        """

        A = self.homogeneous
        B = other.homogeneous

        alpha = np.linalg.det(A)
        beta = bracket(A, A, B) + bracket(A, B, A) + bracket(B, A, A)
        gamma = bracket(A, B, B) + bracket(B, A, B) + bracket(B, B, A)
        delta = np.linalg.det(B)

        poly = np.array([alpha, beta, gamma, delta])
        La = np.roots(poly)
        PP = np.empty_like(poly, shape=(3, 0))

        for la in La:
            P = Conic.__intersect(la, A, B)
            PP = np.column_stack((PP, P))

        # Use points that consists of real values only
        mask = ~np.any(~np.isclose(np.imag(PP), 0), axis=0)
        PP = np.unique(PP[..., mask], axis=1)

        return np.real(PP)

    def __intersect(la, A, B):
        # Set mu arbitrarily to 1 and compute the degenerate conic using the
        # pencil of conics
        C = la * A + B

        assert np.isclose(np.linalg.det(
            C), 0), 'determinant of degenerate conic must be zero'

        # Decompose the degenerate conic
        BB = adjugate(C)
        BB_diag = np.square(np.diag(BB))
        # Select the diagonal element with the largest magnitude
        i = np.argmax(BB_diag)

        # Diagonal element being zero indicates that the conic cannot be
        # decomposed since this causes a division by zero.
        if BB_diag[i] == 0:
            return np.empty_like(BB_diag, shape=(3, 0))

        # NOTE There's a typo in the book; a minus in the sqrt term is missing.
        bb2 = -BB[i, i]

        if bb2 < 0:
            bb2 = np.complex128(bb2)

        bb = np.sqrt(bb2)

        p = BB[:, i] / bb

        M_p = skew_symmetric(p)
        CC = C + M_p

        i, j = np.unravel_index(np.argmax(CC**2), CC.shape)

        if CC[i, j] == 0:
            return np.empty_like(CC, shape=(3, 0))

        # Lines consituting the degenerate conic
        g = CC[i, :]
        h = CC[:, j]

        P1 = Conic.__intersect_line(A, g)
        P2 = Conic.__intersect_line(A, h)

        if P1.shape == P2.shape and np.all(np.isclose(P1, P2)):
            P = P1
        else:
            P = np.column_stack((P1, P2))

        nonzero = ~np.isclose(P[-1, :], 0)
        return P[:, nonzero]

    def __intersect_line(A, g):
        l, u, t = g

        if t == 0:
            return np.empty_like(t, shape=(3, 0))

        M_l = skew_symmetric(g)
        # B = np.linalg.multi_dot([M_l.T, A, M_l])
        B = np.einsum('ji,jk,kl->il', M_l, A, M_l)

        D = -np.linalg.det(B[:2, :2])

        if D < 0:
            D = np.complex128(D)

        aa = np.sqrt(D) / t
        C = B + aa * M_l

        l2_r = np.linalg.norm(C, axis=1)
        l2_c = np.linalg.norm(C, axis=0)

        i = np.argmax(l2_r)
        j = np.argmax(l2_c)

        P1 = C[i, :]
        P2 = C[:, j]

        if np.all(np.isclose(P1, P2)):
            return P1

        return np.column_stack((P1, P2))

    def __factors():
        return np.array([1, 2, 1, 2, 2, 1])

    @staticmethod
    def from_ellipse(x0, major_minor, alpha):
        """Constructs a conic section from ellipse parameters.

        :param x0: The 2-D center of the ellipse.
        :type x0: array-like

        :param major_minor: The size of the half axes.
        :type major_minor: array-like

        :param angle: The orientation of the ellipse in radians.
        :type angle: float

        :return: The ellipse conic.
        :rtype: conics.Conic
        """
        return Conic(g2a(x0, major_minor, alpha))

    @staticmethod
    def from_circle(x0, r):
        """Constructs a conic from geometric representation of a circle.

        :param x0: The 2-D center of the circle.
        :type x0: array-like

        :param r: The circle radius.
        :type r: float

        :return: The circle conic.
        :rtype: conics.Conic
        """

        return Conic.from_homogeneous(_make_circle(x0, r))

    def to_ellipse(self):
        R"""Returns the geometric representation of the ellipse conic.

        :return: A tuple containing the ellipse center :math:`\vec
            x_c\in\mathbb{R}^2`, the length of the semi-major and semi-minor axes
            :math:`(a,b)\in\mathbb{R}_{>0}^2`, and the ellipse orientation
            :math:`-\pi\leq\alpha<\pi`.
        :rtype: tuple
        """

        C33 = self.__C33
        x0 = self.__center(C33)
        return a2g(x0, C33, self.coeffs_[-1])

    @staticmethod
    def from_homogeneous(Q):
        r"""Constructs a conic section from its homogeneous :math:`3\times3`
        symmetric matrix representation.

        :return: New conic section
        :rtype: conics.Conic

        :except ValueError: Raised if the `Q` is not symmetric.
        """

        if not np.isclose(Q - Q.T, np.zeros_like(Q)).all():
            raise ValueError('homogeneous conic matrix must be symmetric')

        coeffs = [Q[0, 0], Q[0, 1], Q[1, 1], Q[0, 2], Q[1, 2], Q[2, 2]]
        return Conic(np.multiply(coeffs, Conic.__factors()))

    def transform(self, R, L=None, invert=True):
        r"""Transforms the conic using a homography :math:`H\in\mathbb{R}^{3\times3}` as

        .. math::

            \tilde{C}' = (H\vec p)^\top \tilde{C}(H\vec p)
                       = \vec p^\top (H^\top \tilde{C} H) \vec p

        See :cite:`Hartley2004` for details.

        :param R: The applied homography on the right hand-side of the original conic section (unless `L` is given)
        :type R: numpy.ndarray

        :param L: The applied homography on the left hand-side of the original conic section. If not given, the transform is computed as the transpose of `R`.
        :type L: numpy.ndarray

        :param invert: Indicates whether the inverse of the homography will be
            used to transform the conic. Set to `True` if you intend to
            transform the points on the conic (default). If `False`, the
            transformation is applied as is without inversion.
        :type invert: bool

        :return: The transform conic section.
        :rtype: conics.Conic
        """

        if invert:
            R = np.linalg.inv(R)

        # (Mx)^T*C*(Mx) => x^T (M^T*C*M) x
        L = L if L is not None else np.transpose(R)
        Q = L @ self.homogeneous @ R
        return Conic.from_homogeneous(Q)

    def translate(self, t):
        """Shifts the points on the conic by a 2-D translation vector `t`.

        :param t: 2-D translation vector by which the points on the conic are
            shifted.
        :type t: array-like

        :return: The shifted conic.
        :rtype: conics.Conic
        """
        t = np.reshape(t, (2, 1))
        M = np.block([[np.eye(2), -t],
                      [0, 0, 1]])
        return self.transform(M, invert=False)

    def scale(self, sx, sy=None):
        R"""Scales the conic coordinates.

        :param sx: Scale factor along the horizontal axis:
        :type sx: float

        :param sy: Scale factor along the vertical axis. If other than `None`,
            the scaling is non-uniform. Otherwise the same factor as for the
            horizontal axis is used.
        :type sy: float, None

        :return: Scaled conic.
        :rtype: conics.Conic
        """

        s = np.stack((sx, sx if sy is None else sy)).ravel()
        d = np.stack((*np.reciprocal(s, dtype=self.coeffs_.dtype), 1))
        M = np.diag(d)

        return self.transform(M, invert=False)

    def rotate(self, angle):
        """Rotates the `points` on the conic in the counter-clockwise
        direction.

        :param angle: The counter-clockwise rotation angle, in radians.
        :type angle: float

        :return: The rotated conic section.
        :rtype: conics.Conic
        """
        M = np.block([[rot2d(-angle), np.zeros((2, 1))],
                      [0, 0, 1]])
        return self.transform(M, invert=False)

    def normalize(self, d=-1):
        r"""Normalizes the conic coefficients such that the determinant of the
        homogeneous symmetric matrix obtains the specified value `d`, i.e.,
        :math:`\det Q = d` where :math:`Q\in\mathbb{R}^{3\times3}` is the conic
        in the matrix form. The normalization scheme was proposed in
        :cite:`Kanatani1993`.

        To obtain a specific determinant value, one can exploit the determinant
        property :math:`\det(kC) = k^n \det C` where :math:`n` is the size of
        the square matrix. Here, we want to determine the factor :math:`k` that
        yields the desired determinant because

        .. math::

            k^n \det C=d \iff k^n=\frac{d}{\det C}

        Since :math:`n=3`, it follows that :math:`k=\sqrt[3]{\frac{d}{\det C}}`.

        :param d: The determinant value the matrix form the conic should obtain.
        :type d: float

        :return: The normalized conic.
        :rtype: conics.Conic
        """
        C = self.homogeneous
        k = np.cbrt(d / np.linalg.det(C))
        return Conic.from_homogeneous(k * C)

    def __repr__(self):
        return repr(self.coeffs_)

    def __sub__(self, other):
        return Conic(self.coeffs_ - other.coeffs_)

    def __plus__(self, other):
        return Conic(self.coeffs_ + other.coeffs_)

    def __call__(self, pts):
        """Evaluates the conic on the points `pts`."""

        s = np.shape(pts)

        if s[0] == 2:
            x, y = pts
            A = np.column_stack((x**2, x * y, y**2, x, y, np.ones_like(x)))
            return A @ self.coeffs_

        return np.diagonal(pts.T @ self.homogeneous @ pts)

    @staticmethod
    def from_parabola(center, p, alpha):
        R"""Constructs a conic section from the geometric representation of a
        parabola.

        The conversion uses the method from :cite:`Ahn2001`.

        :param center: The 2-D coordinate of the parabola vertex.
        :type center: numpy.ndarray

        :param p: The distance from the focus.
        :type p: float

        :param alpha: Parabola orientation (in radians).
        :type alpha: float

        :return: New conic section.
        :rtype: conics.Conic
        """

        if p < 0:
            p *= -1
            alpha -= np.copysign(np.pi, alpha)

        c = np.cos(alpha)
        s = np.sin(alpha)

        x, y = center

        A = s**2
        B = -s * c
        C = c**2
        D = -x * s**2 + y * s * c - p * c
        E = x * s * c - y * c**2 - p * s
        F = (x * s - y * c)**2 + 2 * p * (x * c + y * s)

        return Conic(np.array([A, B, C, D, E, F]) * Conic.__factors())

    def to_parabola(self):
        R"""Returns the geometric representation of the parabola given by the
        current conic.

        :return: 2-D coordinate :math:`\vec x_c\in\mathbb{R}^2` of the parabola vertex, the distance
            :math:`p>0` from the focus and the parabola orientation
            :math:`-\pi\leq\alpha<\pi`.
        :rtype: tuple
        """

        A, B, C, D, E, F = self.coeffs_ / Conic.__factors()

        alpha = np.arctan2(A, -B)

        ab = np.hypot(A, B)
        p = -(A * E - B * D) / ((A + C) * ab)

        X = np.array([[A, B],
                      [(A * D + 2 * C * D - B * E) / (A + C),
                       (C * E + 2 * A * E - B * D) / (A + C)]])
        b = -np.array([(A * D + B * E) / (A + C), F])
        vertex = np.linalg.solve(X, b)

        if p < 0:
            p *= -1
            alpha -= np.copysign(np.pi, alpha)

        return (vertex, p, alpha)

    def gradient(self, pts):
        R"""Computes the conic first-order derivative with respect to its
        coordinates:

        .. math::
            \nabla Q(x,y)
            =
            \left(
            \frac{\partial Q}{\partial x},
            \frac{\partial Q}{\partial y}
            \right)^\top
            =
            \begin{bmatrix}
            2Ax+By+D
            \\
            Bx+2Cy+E
            \end{bmatrix}
            \enspace .

        :param pts: 2-D coordinates where the gradient is evaluated.
        :type pts: numpy.ndarray

        :return: The gradient vector with respect to each 2-D coordinate.
        :rtype: numpy.ndarray
        """

        a, b, c, d, e, f = self.coeffs_
        x, y = pts

        dx = 2 * a * x + b * y + d
        dy = b * x + 2 * c * y + e

        return np.vstack((dx, dy))

    def constrain(self, pts, type='parabola', fix_angle=False):
        R"""Conditions the conic to a specific type and specific properties.

        :param pts: :math:`n` 2-D coordinates given by a :math:`2\times n` matrix
            where each coordinate is store in a column. The conditioning is
            performed with respect to the specified coordinates.
        :type pts: numpy.ndarray

        :param type: Desired conic type. Possible choice is only ``parabola``.
        :type type: str

        :param fix_angle: Specifies whether to fix the angle to the current
            configuration or use a specific value given by the argument in
            radians.
        :type fix_angle: bool, float

        :return: The constrained conic.
        :rtype: conics.Conic
        """

        if type != 'parabola':
            raise ValueError(
                'constraining conic to anything else than a parabola is not supported yet')

        return self.__constrain_parabola(pts, fix_angle)

    def __constrain_parabola(self, pts, fix_angle=False, w8=1e6, w9=1e6, w10=1e6):
        x0 = self.coeffs_ / Conic.__factors()

        if not isinstance(fix_angle, bool) or fix_angle:
            A, B, C, D, E, F = x0

            if not isinstance(fix_angle, bool):
                alpha = fix_angle
                c1 = np.sin(alpha)
                c2 = np.cos(alpha)
            else:
                den = np.hypot(A, B)
                c1 = A / den
                c2 = -B / den

            fix_angle = True

        def fun(a, pts):
            x, y = pts
            A, B, C, D, E, F = a  # / Conic.__factors()

            f = np.column_stack(
                (x**2, 2 * x * y, y**2, 2 * x, 2 * y, np.ones_like(x))) * a[np.newaxis, ...]
            f7 = np.sum(f, axis=1)

            residuals = np.stack((*f7, w8 * (B**2 - A * C), w9 * (A + C - 1)))

            if fix_angle:
                t = np.hypot(A, B)
                residuals = np.stack(
                    (*residuals, w10 * (A - c1 * t), w10 * (B + c2 * t)))

            return residuals

        def jac(a, pts):
            x, y = pts
            A, B, C, D, E, F = a  # / Conic.__factors()

            top = np.column_stack(
                (x**2, 2 * x * y, y**2, 2 * x, 2 * y, np.ones_like(x)))
            bl = np.array([
                [-w8 * C, 2 * w8 * B, -w8 * A],
                [w9, 0, w9]])
            br = np.zeros_like(bl)

            J = np.block([[top],
                          [bl, br]])

            if fix_angle:
                bl = np.array([
                    [w10 * c2**2, w10 * c1 * c2, 0],
                    [w10 * c1 * c2, w10 * c1**2, 0]])
                br = np.zeros_like(bl)

                J = np.block([[J],
                              [bl, br]])

            return J

        r = least_squares(fun, x0, args=(pts, ), jac=jac)
        print(r)

        coeffs = r.x * Conic.__factors()

        return Conic(coeffs)


if False:
    A1 = _make_circle([5, 5], 20)
    A2 = _make_circle([5.01, 5], 10)

    # icp(A1, A2)
    print(concentric_conics_vanishing_line(A2, A1))

if False:
    A1 = _make_circle([5, 5], 20)
    A2 = _make_circle([5.01, 5], 10)

    # icp(A1, A2)
    print(concentric_conics_vanishing_line(A2, A1))

if False:
    c = Conic(1, 2, 3, 4, 5, 6.)

    print(c.homogeneous)

    C1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    C2 = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 1.0])

    intersections = Conic(C1).intersect(Conic(C2))

    print(intersections[:2] / intersections[-1])

    C1 = np.array([1, 0, 0, 0, -1, 0], dtype=np.float32)
    C2 = np.array([1, 0, 0, -2, -1, 1], dtype=np.float32)

    intersections = Conic(C1).intersect(Conic(C2))

    print(intersections[:2] / intersections[-1])
