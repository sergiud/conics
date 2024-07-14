
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
from .geometry import rot2d
from scipy.optimize import least_squares
import numpy as np


def _cost_contact_point(a, b, x, y, xi, yi):
    f1 = 0.5 * (a**2 * y**2 + b**2 * x**2 - a**2 * b**2)
    f2 = b**2 * x * (yi - y) - a**2 * y * (xi - x)

    return np.stack((f1, f2))


def _jac_contact_point(a, b, x, y, xi, yi):
    Q = np.array([[b**2 * x, a**2 * y],
                  [(a**2 - b**2) * y + b**2 * yi, (a**2 - b**2) * x - a**2 * xi]])

    return Q


class Ellipse:
    def __init__(self, center, major_minor, alpha):
        self.center = np.ravel(center)
        self.major_minor = np.ravel(major_minor)
        self.alpha = alpha

    def contact(self, pts):
        pts = np.atleast_2d(pts)

        R = rot2d(self.alpha)
        center = self.center[..., np.newaxis]
        pts1 = R.T @ (pts - center)

        a, b = self.major_minor

        def fun(xy, xyi):
            x, y = xy
            xi, yi = xyi
            return _cost_contact_point(a, b, x, y, xi, yi)

        def jac(xy, xyi):
            x, y = xy
            xi, yi = xyi

            return _jac_contact_point(a, b, x, y, xi, yi)

        xi, yi = pts1

        t1, t2 = np.array([[b], [a]]) * pts1
        xk1 = pts1 * np.prod(pts1, axis=0) / np.hypot(t1, t2)

        mask = np.abs(xi) < a
        tmp = np.sqrt(a**2 - xi**2, out=np.zeros_like(xi), where=mask)

        xk21 = np.vstack((xi, np.copysign(b / a * tmp, yi)))
        xk22 = np.vstack(
            (np.copysign(
                a * np.ones_like(xi),
                xi),
                np.zeros_like(xi)))

        xk2 = np.where(mask, xk21, xk22)

        x0 = np.mean(np.stack((xk1, xk2)), axis=0)
        x = np.empty_like(pts1)

        for i, (x0i, xyi) in enumerate(zip(x0.T, pts1.T)):
            r = least_squares(fun, np.ravel(x0i), args=(xyi, ), jac=jac)
            x[:, i] = r.x

        return R @ x + center

    def refine(self, pts):
        pts = np.atleast_2d(pts)

        def fun(aa, pts):
            xc, yc, a, b, alpha = aa

            e = Ellipse([xc, yc], [a, b], alpha)
            xy = e.contact(pts)

            residuals = xy - pts

            # Stack residuals for x and y component after each other
            return np.ravel(residuals, order='F')

        def jac(aa, pts):
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

            xyxyips = np.sum(xyxyip * np.array([[1], [-1]]), axis=0)

            B1 = np.array([[b2 * x * c - a2 * y * s],
                           [b2 * dyi * c + a2 * dxi * s]])
            B2 = np.array([[b2 * x * s + a2 * y * c],
                           [b2 * dyi * s - a2 * dxi * c]])
            B3 = np.array([[a * (b2 - y2)],
                           [+2 * a * y * dxi]])
            B4 = np.array([[b * (a2 - x2)],
                           [-2 * b * x * dyi]])
            B5 = np.array([[(a2 - b2) * xyp],
                           [(a2 - b2) * (x2 - y2 - xyxyips)]])

            B = np.column_stack([B1, B2, B3, B4, B5])
            B = np.moveaxis(B, -1, 0)

            Q = _jac_contact_point(a, b, x, y, xi, yi)
            Q = np.moveaxis(Q, -1, 0)

            J = R @ np.linalg.solve(Q, B)

            return J.reshape(-1, 5, order='C')

        x0 = np.stack((*self.center, *self.major_minor, self.alpha))
        r = least_squares(fun, x0, args=(pts, ), jac=jac)
        print(r)

        return Ellipse(r.x[:2], r.x[2:4], r.x[-1])

    @staticmethod
    def from_conic(C):
        center, major_minor, alpha = C.to_ellipse()
        return Ellipse(center, major_minor, alpha)

    def to_conic(self):
        return Conic.from_ellipse(self.center, self.major_minor, self.alpha)
