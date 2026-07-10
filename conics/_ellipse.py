# conics - Python library for dealing with conics
#
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

from ._conic import Conic
from .geometry import rot2d
from scipy.optimize import least_squares
import numpy as np
import numpy.typing as npt


def _cost_contact_point(
    a: float, b: float, x: float, y: float, xi: float, yi: float
) -> np.ndarray[tuple[int]]:
    f1 = 0.5 * (a**2 * y**2 + b**2 * x**2 - a**2 * b**2)
    f2 = b**2 * x * (yi - y) - a**2 * y * (xi - x)

    return np.stack((f1, f2))


def _jac_contact_point(
    a: float, b: float, x: float, y: float, xi: float, yi: float
) -> np.ndarray[tuple[int, int]]:
    Q = np.array(
        [
            [b**2 * x, a**2 * y],
            [(a**2 - b**2) * y + b**2 * yi, (a**2 - b**2) * x - a**2 * xi],
        ]
    )

    return Q


class Ellipse:
    """Initializes the ellipse using its geometric representation.

    Parameters
    ----------
    center: array_like, (2, )
        The center of the ellipse.
    major_minor: array_like, (2, )
        The size of the semi-major and semi-minor axes.
    alpha: float
        The orientation angle in radians.
    """

    def __init__(
        self, center: npt.ArrayLike, major_minor: npt.ArrayLike, alpha: float
    ) -> None:
        self.center = np.ravel(center)
        self.major_minor = np.ravel(major_minor)
        self.alpha = alpha

    def contact(self, pts: npt.ArrayLike) -> np.ndarray:
        """Computes the orthogonal points on the ellipse given some 2-D points.

        Orthogonal (contact) points are points on the ellipse closest to those
        passed.

        Parameters
        ----------
        pts: array_like
            The 2-D points whose closest (orthogonal) points on the ellipse
            should be determined.

        Returns
        -------
        contact_pts: numpy.ndarray
            The orthogonal points on the ellipse.
        """
        pts = np.atleast_2d(pts)

        R = rot2d(self.alpha)
        # Rotate into the ellipse-local frame. Since `pts` stores points as
        # rows, the transform is applied on the right instead of
        # transposing `pts` and using R.T on the left.
        pts1 = (pts - self.center) @ R

        a, b = self.major_minor

        def fun(
            xy: np.ndarray[tuple[int]], xyi: np.ndarray[tuple[int]]
        ) -> np.ndarray[tuple[int]]:
            x, y = xy
            xi, yi = xyi
            return _cost_contact_point(a, b, x, y, xi, yi)

        def jac(
            xy: np.ndarray[tuple[int]], xyi: np.ndarray[tuple[int]]
        ) -> np.ndarray[tuple[int, int]]:
            x, y = xy
            xi, yi = xyi

            return _jac_contact_point(a, b, x, y, xi, yi)

        xi, yi = pts1.T

        t1, t2 = (pts1 * np.array([b, a])).T
        den = np.hypot(t1, t2)
        # den is 0 only when pts1 is at the origin (the ellipse center); fall
        # back to 0 there instead of NaN so the initial guess below stays
        # finite and relies solely on xk2.
        ratio = np.divide(
            np.prod(pts1, axis=1), den, out=np.zeros_like(den), where=den != 0
        )
        xk1 = pts1 * ratio[..., np.newaxis]

        mask = np.abs(xi) < a
        tmp = np.sqrt(a**2 - xi**2, out=np.zeros_like(xi), where=mask)

        xk21 = np.column_stack((xi, np.copysign(b / a * tmp, yi)))
        xk22 = np.column_stack(
            (np.copysign(a * np.ones_like(xi), xi), np.zeros_like(xi))
        )

        xk2 = np.where(mask[..., np.newaxis], xk21, xk22)

        x0 = np.mean(np.stack((xk1, xk2)), axis=0)
        x = np.empty_like(pts1)

        for i, (x0i, xyi) in enumerate(zip(x0, pts1)):
            r = least_squares(fun, np.ravel(x0i), args=(xyi,), jac=jac)
            x[i, :] = r.x

        return x @ R.T + self.center

    def refine(self, pts: npt.ArrayLike) -> Ellipse:
        """Refine the ellipse non-linearly by minimizing the orthogonal
        distances.

        The method uses the approach introduced by :cite:t:`Ahn2001`.

        The analytic Jacobian used by the underlying least-squares solve is
        cross-checked symbolically and numerically in
        ``scripts/derive_refine_jacobian.py``.

        Parameters
        ----------
        pts: array_like
            The 2-D points whose closest (orthogonal) points on the ellipse
            should be determined.

        Returns
        -------
        Ellipse:
            The refine ellipse.
        """
        pts = np.atleast_2d(pts)

        def fun(aa: np.ndarray[tuple[int]], pts: np.ndarray) -> np.ndarray[tuple[int]]:
            xc, yc, a, b, alpha = aa

            e = Ellipse([xc, yc], [a, b], alpha)
            xy = e.contact(pts)

            residuals = xy - pts

            # Stack residuals for x and y component after each other
            return np.ravel(residuals)

        def jac(aa: np.ndarray[tuple[int]], pts: np.ndarray):
            xc, yc, a, b, alpha = aa

            e = Ellipse([xc, yc], [a, b], alpha)
            xy = e.contact(pts)

            R = rot2d(alpha)
            c, s = R[:, 0]

            center = np.array([xc, yc])
            # Rotate into the local frame on the right (points are stored as
            # rows), then transpose the small result for the dense
            # elementwise math below.
            xy1 = ((xy - center) @ R).T
            pts1 = ((pts - center) @ R).T

            a2 = a**2
            b2 = b**2
            xyp = np.prod(xy1, axis=0)

            x, y = xy1
            xi, yi = pts1

            diff = pts1 - xy1
            dxi, dyi = diff
            x2 = x**2
            y2 = y**2

            xyxyips = x * xi - y * yi

            # The contact point (x, y) depends on the center only through the
            # local coordinates xi, yi of the input point, so its columns
            # need the same solve(Q, ...) chain rule treatment as a, b, and
            # alpha below. Unlike those, the center also appears directly in
            # the residual (xy1 + center - pts), which contributes an
            # identity term added back after the solve.
            zero = np.zeros_like(x)
            B1 = np.array([[zero], [-(a2 * y * c + b2 * x * s)]])
            B2 = np.array([[zero], [b2 * x * c - a2 * y * s]])
            B3 = np.array([[a * (b2 - y2)], [+2 * a * y * dxi]])
            B4 = np.array([[b * (a2 - x2)], [-2 * b * x * dyi]])
            B5 = np.array([[(a2 - b2) * xyp], [(a2 - b2) * (x2 - y2 - xyxyips)]])

            B = np.column_stack([B1, B2, B3, B4, B5])
            B = np.moveaxis(B, -1, 0)

            Q = _jac_contact_point(a, b, x, y, xi, yi)
            Q = np.moveaxis(Q, -1, 0)

            J = R @ np.linalg.solve(Q, B)
            # Add the identity from the center's direct contribution to the
            # residual to the (xc, yc) columns of every point's 2x5 block, i.e.
            # J[:, :, :2] += np.eye(2), but touching only the nonzero diagonal
            # entries instead of the whole 2x2 block.
            J[:, (0, 1), (0, 1)] += 1

            return J.reshape(-1, 5, order='C')

        x0 = np.stack((*self.center, *self.major_minor, self.alpha))
        r = least_squares(fun, x0, args=(pts,), jac=jac)

        return Ellipse(r.x[:2], r.x[2:4], r.x[-1])

    @property
    def area(self) -> float:
        """Compute the area of this ellipse."""
        return np.pi * np.prod(self.major_minor)

    def segment_area(self, line: npt.ArrayLike) -> float:
        """Computes the area of the region obtained as a result of intersecting the ellipse with a line.

        This ellipse's normalized homogeneous conic :math:`C` (see
        :meth:`to_conic`) evaluates to exactly :math:`-1` at its own center in
        homogeneous coordinates, :math:`\\vec c=(c_x,c_y,1)^\\top`, i.e.
        :math:`\\vec c^\\top C \\vec c=-1`, and is negative everywhere else
        inside the ellipse. Writing points in homogeneous coordinates as
        :math:`\\vec p=(x,y,1)^\\top` as well, this function computes the area
        of the set
        :math:`\\{ \\vec p : \\vec p^\\top \\vec l < 0 \\land \\vec p^\\top C \\vec p < 0 \\}`.

        If the line does not intersect or is tangent to the ellipse, then
        either zero or the full ellipse area is returned. If the line is
        degenerate, having no direction (:math:`\\vec l=(0,0,d)^\\top`), the
        sign of :math:`d` alone decides between the same two outcomes.

        The area is computed by mapping the ellipse to the unit circle
        through the affine transform that rotates into the ellipse-local
        frame and scales each axis by the corresponding semi-axis length.
        This transform has constant Jacobian determinant :math:`\\text{major}
        \\cdot\\text{minor}`, so any area computed in the transformed (unit
        circle) space scales back to the ellipse by that factor. The line
        maps to another line under this transform, at some perpendicular
        distance :math:`h` from the origin (in units of the unit circle's
        radius).

        For :math:`h<1` the cap it cuts off on the side not containing the
        origin can be rotated, without changing its area, so that the cutting
        line becomes the vertical line :math:`x=h`. The cap area then follows
        by integrating the circle's cross-sectional width over
        :math:`x\\in[h,1]`:

        .. math::
            \\int_h^1 2\\sqrt{1-x^2}\\,dx = \\arccos h-h\\sqrt{1-h^2}
            \\enspace,

        from which the requested half-plane area follows directly depending
        on which side of the line the origin (the ellipse center) falls on.

        :math:`\\arccos h` is evaluated as :math:`\\operatorname{atan2}(s, h)`
        with :math:`s=\\sqrt{1-h^2}`, and :math:`s` itself is computed from the
        factored product :math:`(1-h)(1+h)` rather than :math:`1-h^2`. Squaring
        :math:`h` first and then subtracting from 1 cancels leading digits once
        :math:`h` approaches 1 (a near-tangent line), and can even drive the
        argument of the square root slightly negative. The factored form
        halves that cancellation and stays nonnegative.

        The integral derivation above, the :math:`\\operatorname{atan2}`
        rewrite, and the affine scaling by :math:`\\text{major}\\cdot\\text{minor}`
        are all cross-checked symbolically and numerically in
        ``scripts/derive_segment_area.py``.

        Parameters
        ----------
        line : array_like (3,)
            A homogeneous line cutting the ellipse into the two regions this
            method chooses between.

        Returns
        -------
        area : float
            The area of the intersection of the interior of this ellipse and the
            negative half plane of line. This value is nonnegative and bounded
            by the total ellipse area.
        """
        line = np.asarray(line, "f8")
        center_distance = line[:2] @ self.center + line[-1]

        major_minor = self.major_minor
        R = rot2d(self.alpha)
        direction = line[:2] @ R
        A, B = direction * major_minor
        C = center_distance
        norm = np.hypot(A, B)
        total_area = self.area

        if norm == 0:
            return total_area if center_distance < 0 else 0.0

        # Perpendicular distance from the origin to the transformed line, in
        # units of the unit circle's radius.
        h = np.abs(C) / norm

        if h >= 1:
            return total_area if center_distance < 0 else 0.0

        # Area of the unit-circle cap not containing the origin. sqrt(1 - h**2)
        # is computed from (1 - h) * (1 + h) rather than 1 - h**2 to avoid
        # cancellation as h approaches 1 (a near-tangent line), and arccos(h)
        # is replaced by the equivalent atan2(s, h), which does not require
        # its argument to be clipped to [-1, 1].
        s = np.sqrt(np.maximum(0.0, (1 - h) * (1 + h)))
        cap_area = np.prod(major_minor) * (np.arctan2(s, h) - h * s)

        # The origin (the ellipse center) lies in {expr < 0} exactly when C
        # (the value of the transformed line at the origin) is negative.
        if C < 0:
            return total_area - cap_area

        return cap_area

    @staticmethod
    def from_conic(C: Conic) -> Ellipse:
        """Constructs an ellipse from the specified conic."""
        center, major_minor, alpha = C.to_ellipse()
        return Ellipse(center, major_minor, alpha)

    def to_conic(self) -> Conic:
        """Constructs a :class:`Conic` from the current :class:`Ellipse`
        instance.
        """
        return Conic.from_ellipse(self.center, self.major_minor, self.alpha)
