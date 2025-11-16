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

from ._conic import Conic
from ._utils import polygon_area
from .geometry import hnormalized
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
        center = self.center[..., np.newaxis]
        pts1 = R.T @ (pts - center)

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

        xi, yi = pts1

        t1, t2 = np.array([[b], [a]]) * pts1
        xk1 = pts1 * np.prod(pts1, axis=0) / np.hypot(t1, t2)

        mask = np.abs(xi) < a
        tmp = np.sqrt(a**2 - xi**2, out=np.zeros_like(xi), where=mask)

        xk21 = np.vstack((xi, np.copysign(b / a * tmp, yi)))
        xk22 = np.vstack((np.copysign(a * np.ones_like(xi), xi), np.zeros_like(xi)))

        xk2 = np.where(mask, xk21, xk22)

        x0 = np.mean(np.stack((xk1, xk2)), axis=0)
        x = np.empty_like(pts1)

        for i, (x0i, xyi) in enumerate(zip(x0.T, pts1.T)):
            r = least_squares(fun, np.ravel(x0i), args=(xyi,), jac=jac)
            x[:, i] = r.x

        return R @ x + center

    def refine(self, pts: npt.ArrayLike) -> Ellipse:
        """Refine the ellipse non-linearly by minimizing the orthogonal
        distances.

        The method uses the approach introduced by :cite:t:`Ahn2001`.

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
            return np.ravel(residuals, order='F')

        def jac(aa: np.ndarray[tuple[int]], pts: np.ndarray):
            xc, yc, a, b, alpha = aa

            e = Ellipse([xc, yc], [a, b], alpha)
            xy = e.contact(pts)

            R = rot2d(alpha)
            c, s = R[0]

            center = np.array([[xc], [yc]])
            xy1 = R.T @ (xy - center)
            pts1 = R.T @ (pts - center)

            a2 = a**2
            b2 = b**2
            xyp = np.prod(xy1, axis=0)

            x, y = xy1
            xi, yi = pts1

            diff = pts1 - xy1
            dxi, dyi = diff
            xyxyip = xy1 * pts1
            x2 = x**2
            y2 = y**2

            # TODO Use numpy.einsum to possibly minimize the round-off error
            xyxyips = np.sum(xyxyip * np.array([[1], [-1]]), axis=0)

            B1 = np.array([[b2 * x * c - a2 * y * s], [b2 * dyi * c + a2 * dxi * s]])
            B2 = np.array([[b2 * x * s + a2 * y * c], [b2 * dyi * s - a2 * dxi * c]])
            B3 = np.array([[a * (b2 - y2)], [+2 * a * y * dxi]])
            B4 = np.array([[b * (a2 - x2)], [-2 * b * x * dyi]])
            B5 = np.array([[(a2 - b2) * xyp], [(a2 - b2) * (x2 - y2 - xyxyips)]])

            B = np.column_stack([B1, B2, B3, B4, B5])
            B = np.moveaxis(B, -1, 0)

            Q = _jac_contact_point(a, b, x, y, xi, yi)
            Q = np.moveaxis(Q, -1, 0)

            J = R @ np.linalg.solve(Q, B)

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

        If the ellipse is defined by a normalized homogeneous conic :math:`C`, such
        that the center :math:`\\vec c` is negative, (i.e.
        :math:`\\vec c^\\top C \\vec c < 0`), then this function computes the area
        of the spaced defined by
        :math:`{ \\vec p : \\vec p^\\top l < 0 \\land \\vec p^\\top C \\vec p < 0 }`.

        If the line does not intersect or tangents the ellipse, then either zero
        or the full ellipse area will be returned.

        Parameters
        ----------
        line : array_like (3,)
            An array of representing a homogeneous line to define the conic section area.

        Returns
        -------
        area : float
            The area of the intersection of the interior of this eclipse and the
            negative half plane of line. This value is nonnegative and bounded
            by the total ellipse area.
        """
        # NOTE while the library currently uses column-major order, this
        # function internally uses row-major order
        line = np.asarray(line, "f8")
        conic = self.to_conic()
        center_distance = line[:2] @ self.center + line[-1]
        intersections = conic.intersect_line(line)

        if intersections.shape[1] == 2:
            point_from, point_to = hnormalized(intersections).T

            # create a polygon of the points in area, and compute the area in
            # the triangle between center and both points
            poly_shape = np.stack((point_from, point_to, self.center))
            poly_area = polygon_area(poly_shape)

            # note compute the interior angle from both points
            cx, cy = np.moveaxis(poly_shape[:2, :] - self.center[None], -1, 0)
            interior_angles = np.atan2(cy, cx) - self.alpha

            # using the interior angles, compute the section area from the major
            # axis to each point
            major, minor = self.major_minor
            norm_y = major * np.sin(interior_angles)
            norm_x = minor * np.cos(interior_angles)
            norm_area = (major * minor) / 2
            area_from, area_to = np.moveaxis(
                np.atan2(norm_y, norm_x) * norm_area, -1, 0
            )

            # now subtract to get just the sector area, then subtract the
            # polygon area, and finally obtain the remainder with respect to the
            # total area for when these are negative (indicating a reverse
            # direction) of the intersection points
            sector_area = area_to - area_from
            total_area = self.area
            area = (sector_area - poly_area) % total_area

            # if the center is negative, but the area is less than half, need to
            # invert
            if (center_distance < 0) == (sector_area * 2 < total_area):
                return total_area - area

            return area
        elif center_distance > 0:
            return 0.0

        return self.area

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
