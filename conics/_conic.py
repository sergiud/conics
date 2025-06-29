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
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .geometry import projectively_unique
from .geometry import rot2d
from scipy.optimize import least_squares
import itertools
import numpy as np
import numpy.typing as npt
import scipy.linalg
import warnings


def _make_circle(x0: npt.ArrayLike, r: float) -> np.ndarray:
    x0 = np.reshape(x0, (2, 1))
    C = np.block([[np.eye(2), -x0], [-x0.T, x0.T @ x0 - r**2]])
    return C


def icp(A1: np.ndarray, A2: np.ndarray) -> tuple[float, float]:
    e = scipy.linalg.eig(A1, A2, left=False, right=False)
    k = np.argmax(np.abs(np.median(e) - e))
    u, s, vt = np.linalg.svd(e[k] * np.linalg.inv(A1) - np.linalg.inv(A2))

    u = u[..., :2]
    s = s[:2]

    values1 = u @ (np.sqrt(s) * np.array([1, +1j]))
    values2 = u @ (np.sqrt(s) * np.array([1, -1j]))

    return values1, values2


# The Common Self-polar Triangle of Concentric Circles and Its Application to
# Camera Calibration


def concentric_conics_vanishing_line(
    C1: np.ndarray, C2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    C = np.linalg.solve(C2, C1)
    evals, evecs = np.linalg.eig(C)

    evals = evals[::-1]
    evecs = evecs[..., ::-1]

    x1, x2, x3 = evecs.T
    v = np.cross(x2, x3)

    return x1, v


def g2a(x0: npt.ArrayLike, major_minor: npt.ArrayLike, alpha: float) -> np.ndarray:
    if np.less(*major_minor):
        warnings.warn(
            'ellipse major axis size must be larger or equal to the minor one. however, the provided major axis is smaller than the minor axis. this may cause an unintentional change of ellipse orientation',
            UserWarning,
        )

    x0 = np.asarray(x0)
    R = rot2d(alpha)
    M = R @ np.diag(np.reciprocal(np.square(major_minor), dtype=R.dtype)) @ R.T
    a, b, c = np.array([1, 2, 1]) * M[np.triu_indices_from(M)]
    d, e = -2 * M @ x0
    f = x0.T @ M @ x0 - 1

    return np.vstack((a, b, c, d, e, f))


def a2g(
    x0: np.ndarray, C33: np.ndarray, f: float
) -> tuple[np.ndarray, np.ndarray, float]:
    factor = x0.T @ C33 @ x0 - f

    evals, evecs = np.linalg.eigh(C33)
    # TODO sqrt argument may be negative
    val = factor * np.reciprocal(evals)
    major_minor = np.sqrt(val)

    # TODO viz generates division by zero warning
    # alpha = np.arctan(np.divide(*evecs[::-1, 0]))
    alpha = np.arctan2(*evecs[::-1, 0]) % np.pi

    return x0, major_minor, alpha


def bracket(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    return np.linalg.det(np.stack((A[:, 0], B[:, 1], C[:, 2]), axis=1))


def cofactor(A: np.ndarray) -> np.ndarray:
    n, m = A.shape

    C = np.empty_like(A)

    # TODO should be able to achieve this by creating a stack of of indicies
    # that correspond to the minors, e.g. an index array (I) of shape
    # (n, n, n - 1, n-1). Then something equivalent to np.linalg.det(A[I]) will
    # be the cofactor, but the details are a bit tricky
    for i in range(n):
        for j in range(m):
            minors = np.delete(np.delete(A, i, axis=0), j, axis=1)
            sign = 1 if (i + j) % 2 == 0 else -1
            C[i, j] = sign * np.linalg.det(minors)

    return C


def adjugate(C: npt.ArrayLike) -> np.ndarray:
    det = np.linalg.det(C)

    if det == 0:
        return np.transpose(cofactor(C))

    return np.linalg.inv(C).T * det


def skew_symmetric(C: Sequence[float]) -> np.ndarray:
    a, b, c = C
    return np.array([[0, c, -b], [-c, 0, a], [b, -a, 0]])


def surface_normal(Q: np.ndarray, r: float = 1) -> tuple[np.ndarray, float]:
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


def projected_center(Q: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Provides the projected center of the circle in the camera coordinate
    system.
    """
    return np.linalg.solve(Q, n)


def estimate_pose(Q: np.ndarray, r: float, alpha: float):
    r"""Estimates the 5 :term:`DoF` camera pose with respect to the supporting
    plane of a circle projection :cite:`Chen2004`.

    Parameters
    ----------
    Q : numpy.ndarray
        :math:`3\times3` symmetric matrix that defines the oblique cone
        given by  the rays passing through the center of the camera and
        the circle projection.
    r : float
        The radius of the projected circle.
    alpha : float
        The orientation of the projected circle, in radians.

    Returns
    -------
    tuple
        Eight possible solutions :math:`(R_i,\vec t_i,\vec n_i,\vec
        c,s_{1,i},s_{2,i},s_{3,i},m_i)`, :math:`i=1,\dotsc,8` that
        describe the pose of the camera observing the circle projection
        and its supporting plane.

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
    s1, s2, s3 = np.array(list(itertools.product(*itertools.repeat([+1, -1], 3)))).T
    alpha = alpha * np.ones_like(s1)

    den = np.sqrt(-lambda1 * lambda3)
    z0 = s3 * lambda2 * r / den

    right = np.array([[s2 * h], [np.zeros_like(s1)], [-s1 * g]])

    # Batch multiplication + move num axis from the end to the front
    # Basically,
    # n = np.einsum('ij,jlk->kil', evecs, right)
    n = evecs @ np.moveaxis(right, -1, 0)

    # Expand z0 dimensions to enable correct broadcasting for the multiplication with a scalar.
    left = np.array([[lambda3 / lambda2], [0], [lambda1 / lambda2]])[..., np.newaxis]

    c = z0[..., np.newaxis, np.newaxis] * evecs @ np.moveaxis(left * right, -1, 0)

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    R = evecs @ np.moveaxis(
        np.array(
            [
                [g * cos_a, s1 * g * sin_a, s2 * h],
                [sin_a, -s1 * cos_a, np.zeros_like(s1)],
                [s1 * s2 * h * cos_a, s2 * h * sin_a, -s1 * g],
            ]
        ),
        -1,
        0,
    )

    factor = np.sqrt((lambda1 - lambda2) * (lambda2 - lambda3)) / lambda2
    t = (
        np.array(
            [[-s2 * factor * cos_a], [-s1 * s2 * factor * sin_a], [np.ones_like(s1)]]
        )
        * z0
    )
    t = np.moveaxis(t, -1, 0)
    mask = np.ravel((n[:, -1, :] > 0) & (c[:, -1, :] < 0))

    return R, t, n, c, s1, s2, s3, mask


class Conic:
    """Initializes the conic using the given coefficients.

    Parameters
    ----------
    args : array_like, optional
        Coefficients of the quadratic curve.
    """

    def __init__(self, *args: float | np.ndarray) -> None:
        if not (
            len(args) == 0 or (len(args) == 1 and np.size(*args) == 6) or len(args) == 6
        ):
            raise ValueError(
                f'unexpected number of arguments; expected 0, 1 or 6 arguments but got {len(args)}'
            )

        self.coeffs_: np.ndarray = np.ravel(args)

    @property
    def __C33(self) -> np.ndarray:
        a, b, c, d, e, f = self.coeffs_
        half_b = b / 2
        return np.block([[a, half_b], [half_b, c]])

    def __center(self, C33: np.ndarray) -> np.ndarray:
        d, e = self.coeffs_[-3:-1]
        return -np.linalg.solve(C33, np.vstack((d, e)) / 2)

    @property
    def center(self):
        """Returns the midpoint of a central conic :cite:`Ayoub1993`.

        Returns
        -------
        numpy.ndarray
            2-D coordinate of the conic center.
        """
        return self.__center(self.__C33)

    @property
    def homogeneous(self) -> np.ndarray:
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

        return np.array([[A, B, D], [B, C, E], [D, E, F]])

    def intersect(self, other: Conic, atol: float = 1e-4) -> np.ndarray:
        r"""Computes the intersections of `self` with another conic.

        The method implements the algorithm introduced in
        :cite:`RichterGebert2011`.

        Parameters
        ----------
        other : Conic
            The conic for which the intersections are to be computed.
        atol : float, optional
            The absolute tolerance to consider an intersection a duplicate of
            another one, and to consider an intersection conic degenerate.

        Returns
        -------
        numpy.ndarray
            A matrix of homogeneous 2-D points stored in column vectors
            of a :math:`3\times N` matrix consisting of :math:`0\leq
            N\leq 4` columns.


        .. plot:: ../examples/intersections.py
        """

        A = self.homogeneous
        B = other.homogeneous

        alpha = np.linalg.det(A)
        beta = bracket(A, A, B) + bracket(A, B, A) + bracket(B, A, A)
        gamma = bracket(A, B, B) + bracket(B, A, B) + bracket(B, B, A)
        delta = np.linalg.det(B)

        poly = np.array([alpha, beta, gamma, delta], dtype=complex)
        La = np.roots(poly)

        Ps = [Conic.__intersect(la, A, B, atol=atol) for la in La]
        Ps.append(np.empty_like(poly, shape=(3, 0)))
        PP = np.column_stack(Ps)

        # Use points that consists of real values only
        mask = ~np.any(~np.isclose(np.imag(PP), 0), axis=0)
        PP = PP[..., mask]

        return projectively_unique(np.real(PP), atol=atol)

    @staticmethod
    def __intersect(
        la: float, A: np.ndarray, B: np.ndarray, *, atol: float = 1e-9
    ) -> np.ndarray:
        # Set mu arbitrarily to 1 and compute the degenerate conic using the
        # pencil of conics
        C = la * A + B

        assert np.isclose(
            np.linalg.det(C), 0, atol=atol
        ), 'determinant of degenerate conic must be zero'

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

        bb = np.sqrt(bb2)

        p = BB[:, i] / bb

        M_p = skew_symmetric(p)
        CC = C + M_p

        i, j = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)

        if CC[i, j] == 0:
            return np.empty_like(CC, shape=(3, 0))

        # Lines constituting the degenerate conic
        g = CC[i, :]
        h = CC[:, j]

        P1 = Conic.__intersect_line(A, g)
        P2 = Conic.__intersect_line(A, h)

        return np.column_stack((P1, P2))

    @staticmethod
    def __intersect_line(A: np.ndarray, l: np.ndarray) -> np.ndarray:
        if np.all(np.equal(l, 0)):
            return np.empty_like(A, shape=(3, 0))

        A = np.asarray(A, dtype=complex)

        M_l = skew_symmetric(l)
        # B = np.linalg.multi_dot([M_l.T, A, M_l])
        # M_l.T @ A @ M_l
        B = np.einsum('ji,jk,kl->il', M_l, A, M_l)

        for i in range(3):
            if l[i] == 0:
                continue

            # If the corresponding coefficient is zero take a different 2x2
            # (minor) matrix
            D = np.linalg.det(np.delete(np.delete(B, i, axis=0), i, axis=1))
            t = l[i]
            break

        alpha = np.sqrt(-D) / t
        C = B + alpha * M_l
        i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)

        P1 = C[i, :]
        P2 = C[:, j]

        return np.column_stack((P1, P2))

    @staticmethod
    def __factors() -> np.ndarray:
        return np.array([1, 2, 1, 2, 2, 1])

    def intersect_line(self, l: npt.ArrayLike) -> np.ndarray:
        r"""Computes the intersections of `self` with a homogeneous line.

        A line in homogeneous coordinates is given by :math:`\vec l = (\vec
        n^\top, d)^\top \in \mathbb{R}^3` where :math:`\vec n` is the normal to
        the line and :math:`d` its distance to the origin.

        Parameters
        ----------
        l : array_like (3, )
            The homogeneous line to intersect the conic with.

        Returns
        -------
        numpy.ndarray
            A matrix of homogeneous 2-D points stored in column vectors
            of a :math:`3\times N` matrix consisting of :math:`0\leq
            N\leq 2` columns.


        .. plot:: ../examples/line_intersections.py
        """

        inter = Conic.__intersect_line(self.homogeneous, l)
        mask = np.all(np.isclose(np.imag(inter), 0), axis=0)
        return projectively_unique(np.real(inter[..., mask]))

    @staticmethod
    def from_ellipse(
        x0: npt.ArrayLike, major_minor: npt.ArrayLike, alpha: float
    ) -> Conic:
        """Constructs a conic section from ellipse parameters.

        Parameters
        ----------
        x0 : array_like
            The 2-D center of the ellipse.
        major_minor : array_like
            The size of the half axes.
        angle : float
            The orientation of the ellipse in radians.

        Returns
        -------
        conics.Conic
            The ellipse conic.
        """
        return Conic(g2a(x0, major_minor, alpha))

    @staticmethod
    def from_circle(x0: npt.ArrayLike, r: float) -> Conic:
        """Constructs a conic from geometric representation of a circle.

        Parameters
        ----------
        x0 : array_like
            The 2-D center of the circle.
        r : float
            The circle radius.

        Returns
        -------
        conics.Conic
            The circle conic.
        """

        return Conic.from_homogeneous(_make_circle(x0, r))

    def to_ellipse(self):
        R"""Returns the geometric representation of the ellipse conic.

        Returns
        -------
        tuple
            A tuple containing the ellipse center :math:`\vec
            x_c\in\mathbb{R}^2`, the length of the semi-major and semi-
            minor axes :math:`(a,b)\in\mathbb{R}_{>0}^2`, and the
            ellipse orientation :math:`-\pi\leq\alpha<\pi`.
        """

        C33 = self.__C33
        x0 = self.__center(C33)
        return a2g(x0, C33, self.coeffs_[-1])

    def to_circle(self):
        """Converts the conic to a circle.

        If the conic section does not represent a circle, an exception will be
        thrown.

        Returns
        -------
        x0 : numpy.ndarray
            The circle center
        r : float
            The circle radius.

        Raises
        ------
        ValueError
            Raised if the instance does not represent a circle.
        """
        C33 = self.__C33
        x0 = self.__center(C33)
        f = self.coeffs_[-1]

        tmp = C33 - np.diag(np.diagonal(C33))

        if np.any(~np.isclose(tmp, 0)):
            raise ValueError('conic is not a circle')

        det = np.linalg.det(C33)
        s = np.sqrt(det)
        # f = s(x0.T @ x0 - r^2)
        # f/s = x0.T @ x0 - r^2
        # f/s - x0.T @ x0 = -r^2
        # r^2 = x0.T @ x0 - f/s
        r2 = np.dot(x0.T, x0) - f / s

        assert r2 > 0
        r = np.sqrt(r2)

        return x0, r

    @staticmethod
    def from_homogeneous(Q: np.ndarray) -> Conic:
        r"""Constructs a conic section from its homogeneous :math:`3\times3`
        symmetric matrix representation.

        Returns
        -------
        conics.Conic
            New conic section

        Raises
        ------
        ValueError
            Raised if the `Q` is not symmetric.
        """

        if not np.isclose(Q - Q.T, np.zeros_like(Q)).all():
            raise ValueError('homogeneous conic matrix must be symmetric')

        coeffs = [Q[0, 0], Q[0, 1], Q[1, 1], Q[0, 2], Q[1, 2], Q[2, 2]]
        return Conic(np.multiply(coeffs, Conic.__factors()))

    def transform(
        self, R: np.ndarray, L: np.ndarray | None = None, *, invert: bool = True
    ) -> Conic:
        r"""Transforms the conic using a homography :math:`H\in\mathbb{R}^{3\times3}` as

        .. math::

            \tilde{C}' = (H\vec p)^\top \tilde{C}(H\vec p)
                       = \vec p^\top (H^\top \tilde{C} H) \vec p

        See :cite:t:`Hartley2004` for details.

        Parameters
        ----------
        R : numpy.ndarray
            The applied homography on the right hand-side of the
            original conic section (unless `L` is given)
        L : numpy.ndarray
            The applied homography on the left hand-side of the original
            conic section. If not given, the transform is computed as
            the transpose of `R`.
        invert : bool
            Indicates whether the inverse of the homography will be used
            to transform the conic. Set to `True` if you intend to
            transform the points on the conic (default). If `False`, the
            transformation is applied as is without inversion.

        Returns
        -------
        conics.Conic
            The transform conic section.
        """

        if invert:
            R = np.linalg.inv(R)

        # (Mx)^T*C*(Mx) => x^T (M^T*C*M) x
        L = L if L is not None else np.transpose(R)
        Q = L @ self.homogeneous @ R
        return Conic.from_homogeneous(Q)

    def translate(self, t: npt.ArrayLike) -> Conic:
        """Shifts the points on the conic by a 2-D translation vector `t`.

        Parameters
        ----------
        t : array_like
            2-D translation vector by which the points on the conic are
            shifted.

        Returns
        -------
        conics.Conic
            The shifted conic.
        """
        t = np.reshape(t, (2, 1))
        M = np.block([[np.eye(2), -t], [0, 0, 1]])
        return self.transform(M, invert=False)

    def scale(self, sx: foat, sy: float | None = None) -> Conic:
        R"""Scales the conic coordinates.

        Parameters
        ----------
        sx : float
            Scale factor along the horizontal axis:
        sy : float, None
            Scale factor along the vertical axis. If other than `None`,
            the scaling is non-uniform. Otherwise the same factor as for
            the horizontal axis is used.

        Returns
        -------
        conics.Conic
            Scaled conic.
        """

        s = np.stack((sx, sx if sy is None else sy)).ravel()
        d = np.stack((*np.reciprocal(s, dtype=self.coeffs_.dtype), 1))
        M = np.diag(d)

        return self.transform(M, invert=False)

    def rotate(self, angle: float) -> Conic:
        """Rotates the `points` on the conic in the counter-clockwise
        direction.

        Parameters
        ----------
        angle : float
            The counter-clockwise rotation angle, in radians.

        Returns
        -------
        conics.Conic
            The rotated conic section.
        """
        M = np.block([[rot2d(-angle), np.zeros((2, 1))], [0, 0, 1]])
        return self.transform(M, invert=False)

    def normalize(self, d: float = -1) -> Conic:
        r"""Normalizes the conic coefficients such that the determinant of the
        homogeneous symmetric matrix obtains the specified value `d`, i.e.,
        :math:`\det Q = d` where :math:`Q\in\mathbb{R}^{3\times3}` is the conic
        in the matrix form. The normalization scheme was proposed by
        :cite:t:`Kanatani1993`.

        To obtain a specific determinant value, one can exploit the determinant
        property :math:`\det(kC) = k^n \det C` where :math:`n` is the size of
        the square matrix. Here, we want to determine the factor :math:`k` that
        yields the desired determinant because

        .. math::

            k^n \det C=d \iff k^n=\frac{d}{\det C}

        Since :math:`n=3`, it follows that :math:`k=\sqrt[3]{\frac{d}{\det C}}`.

        Parameters
        ----------
        d : float
            The determinant value the matrix form the conic should
            obtain.

        Returns
        -------
        conics.Conic
            The normalized conic.
        """
        C = self.homogeneous
        k = np.cbrt(d / np.linalg.det(C))
        return Conic.from_homogeneous(k * C)

    def __repr__(self) -> str:
        return repr(self.coeffs_)

    def __sub__(self, other: Conic) -> Conic:
        return Conic(self.coeffs_ - other.coeffs_)

    def __plus__(self, other: Conic) -> Conic:
        return Conic(self.coeffs_ + other.coeffs_)

    def __call__(self, pts: npt.ArrayLike) -> np.ndarray:
        """Evaluates the conic on the points `pts`."""

        s = np.shape(pts)

        if s[0] == 2:
            x, y = np.asarray(pts)
            A, B, C, D, E, F = self.coeffs_
            return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F

        # np.einsum('ij...,jk...,ij...->i...', pts, C, pts)
        return np.einsum('ji...,jk...,ji...->i...', pts, self.homogeneous, pts)

    @staticmethod
    def from_parabola(center: npt.ArrayLike, p: float, alpha: float) -> Conic:
        R"""Constructs a conic section from the geometric representation of a
        parabola.

        The conversion uses the method by :cite:t:`Ahn2001`.

        Parameters
        ----------
        center : numpy.ndarray
            The 2-D coordinate of the parabola vertex.
        p : float
            The distance from the focus.
        alpha : float
            Parabola orientation (in radians).

        Returns
        -------
        conics.Conic
            New conic section.
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
        F = (x * s - y * c) ** 2 + 2 * p * (x * c + y * s)

        return Conic(np.array([A, B, C, D, E, F]) * Conic.__factors())

    def to_parabola(self) -> tuple[np.ndarray, float, float]:
        R"""Returns the geometric representation of the parabola given by the
        current conic.

        Returns
        -------
        tuple
            2-D coordinate :math:`\vec x_c\in\mathbb{R}^2` of the
            parabola vertex, the distance :math:`p>0` from the focus and
            the parabola orientation :math:`-\pi\leq\alpha<\pi`.
        """

        A, B, C, D, E, F = self.coeffs_ / Conic.__factors()

        alpha = np.arctan2(A, -B)

        ab = np.hypot(A, B)
        p = -(A * E - B * D) / ((A + C) * ab)

        X = np.array(
            [
                [A, B],
                [
                    (A * D + 2 * C * D - B * E) / (A + C),
                    (C * E + 2 * A * E - B * D) / (A + C),
                ],
            ]
        )
        b = -np.array([(A * D + B * E) / (A + C), F])
        vertex = np.linalg.solve(X, b)

        if p < 0:
            p *= -1
            alpha -= np.copysign(np.pi, alpha)

        return (vertex, p, alpha)

    def gradient(self, pts: npt.ArrayLike) -> np.ndarray:
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

        Parameters
        ----------
        pts : numpy.ndarray
            2-D coordinates where the gradient is evaluated.

        Returns
        -------
        numpy.ndarray
            The gradient vector with respect to each 2-D coordinate.
        """

        a, b, c, d, e, f = self.coeffs_
        x, y = pts

        dx = 2 * a * x + b * y + d
        dy = b * x + 2 * c * y + e

        return np.vstack((dx, dy))

    def constrain(
        self,
        pts: npt.ArrayLike,
        type: Literal['parabola'] = 'parabola',
        fix_angle: bool | float = False,
    ) -> Conic:
        R"""Conditions the conic to a specific type and specific properties.

        Parameters
        ----------
        pts : numpy.ndarray
            :math:`n` 2-D coordinates given by a :math:`2\times n`
            matrix where each coordinate is stored in a column. The
            conditioning is performed with respect to the specified
            coordinates.
        type : str
            Desired conic type. Possible choice is only ``parabola``.
        fix_angle : bool, float
            Specifies whether to fix the angle to the current
            configuration or use a specific value given by the argument
            in radians.

        Returns
        -------
        conics.Conic
            The constrained conic.
        """

        if type != 'parabola':
            raise ValueError(
                'constraining conic to anything else than a parabola is not supported yet'
            )

        return self.__constrain_parabola(pts, fix_angle)

    def __constrain_parabola(
        self,
        pts: np.ndarray,
        fix_angle: bool | float = False,
        w8: float = 1e6,
        w9: float = 1e6,
        w10: float = 1e6,
    ):
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

        def fun(a: np.ndarray, pts: np.ndarray) -> np.ndarray:
            x, y = pts
            A, B, C, D, E, F = a  # / Conic.__factors()

            f = (
                np.column_stack((x**2, 2 * x * y, y**2, 2 * x, 2 * y, np.ones_like(x)))
                * a[np.newaxis, ...]
            )
            f7 = np.sum(f, axis=1)

            residuals = np.stack((*f7, w8 * (B**2 - A * C), w9 * (A + C - 1)))

            if fix_angle:
                t = np.hypot(A, B)
                residuals = np.stack(
                    (*residuals, w10 * (A - c1 * t), w10 * (B + c2 * t))
                )

            return residuals

        def jac(a: np.ndarray, pts: np.ndarray) -> np.ndarray:
            x, y = pts
            A, B, C, D, E, F = a  # / Conic.__factors()

            top = np.column_stack(
                (x**2, 2 * x * y, y**2, 2 * x, 2 * y, np.ones_like(x))
            )
            bl = np.array([[-w8 * C, 2 * w8 * B, -w8 * A], [w9, 0, w9]])
            br = np.zeros_like(bl)

            J = np.block([[top], [bl, br]])

            if fix_angle:
                bl = np.array(
                    [[w10 * c2**2, w10 * c1 * c2, 0], [w10 * c1 * c2, w10 * c1**2, 0]]
                )
                br = np.zeros_like(bl)

                J = np.block([[J], [bl, br]])

            return J

        r = least_squares(fun, x0, args=(pts,), jac=jac)

        coeffs = r.x * Conic.__factors()

        return Conic(coeffs)


if False:
    A1 = _make_circle([5, 5], 20)
    A2 = _make_circle([5.01, 5], 10)

    # icp(A1, A2)

if False:
    A1 = _make_circle([5, 5], 20)
    A2 = _make_circle([5.01, 5], 10)

    # icp(A1, A2)

if False:
    c = Conic(1, 2, 3, 4, 5, 6.0)

    C1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    C2 = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 1.0])

    intersections = Conic(C1).intersect(Conic(C2))

    C1 = np.array([1, 0, 0, 0, -1, 0], dtype=np.float32)
    C2 = np.array([1, 0, 0, -2, -1, 1], dtype=np.float32)

    intersections = Conic(C1).intersect(Conic(C2))
