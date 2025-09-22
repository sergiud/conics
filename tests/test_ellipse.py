#
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
#

from conics import Conic
from conics import Ellipse
from conics.fitting import fit_dlt
from conics.fitting import fit_nievergelt
import numpy as np


def test_ellipse_fitting():
    pts = np.array([[1, 2, 5, 7, 9, 3, 6, 8], [7, 6, 8, 7, 5, 7, 2, 4]])

    C = fit_dlt(pts)
    C = fit_nievergelt(pts, type='ellipse', scale=True)
    e = Ellipse.from_conic(C)

    values1 = C(pts)
    sse1 = np.inner(values1, values1)

    e = Ellipse([4.84, 4.979], [3.391, 3.391], 0)

    e1 = e.refine(pts)

    C2 = e1.to_conic()

    values2 = C2(pts)
    sse2 = np.inner(values2, values2)

    assert sse2 <= sse1
    # print(e.center, e.major_minor, e.alpha)

    e1 = e.refine(pts)
    print(e1.center, e1.major_minor, e1.alpha)


def test_circle():
    c = Conic.from_circle([1, 2], 3)
    C = 4 * c.homogeneous

    c1 = Conic.from_homogeneous(C)

    x0, r = c1.to_circle()
    np.testing.assert_array_almost_equal(x0, [[1], [2]])
    np.testing.assert_almost_equal(r, 3)


def test_circle_not_circle():
    c = Conic.from_ellipse([1, 2], [4, 3], 0.1)
    C = 4 * c.homogeneous
    c1 = Conic.from_homogeneous(C)

    with np.testing.assert_raises(ValueError):
        c1.to_circle()


def test_ellipse_segment_area() -> None:
    ellipse = Ellipse([0, 0], [2, 1], 0.0)
    # cuts the ellipse in half, so half the area
    assert np.isclose(ellipse.segment_area([1, 1, 0]), np.pi)
    # less of the ellipse is in the negative, so should be in (0, pi)
    assert np.isclose(ellipse.segment_area([1, 1, 1]), 1.4, atol=0.1)
    # inversion of that inverts
    assert np.isclose(ellipse.segment_area([-1, -1, -1]), 4.9, atol=0.1)
    # outside of ellipse on positive size, so no area
    assert np.isclose(ellipse.segment_area([1, 1, 3]), 0.0)
    # more of the ellipse is in the negative, so should be in (pi, 2 pi)
    assert np.isclose(ellipse.segment_area([1, 1, -1]), 4.9, atol=0.1)
    # inverting line inverts area
    assert np.isclose(ellipse.segment_area([-1, -1, 1]), 1.4, atol=0.1)
    # outside of ellipse on negative side, so full area
    assert np.isclose(ellipse.segment_area([1, 1, -3]), 2 * np.pi)
    # inversion of that, so no area
    assert np.isclose(ellipse.segment_area([-1, -1, 3]), 0.0)

    # same as above, but flipped angle
    assert np.isclose(ellipse.segment_area([1, -1, 0]), np.pi)
    # less of the ellipse is in the negative, so should be in (0, pi)
    assert np.isclose(ellipse.segment_area([1, -1, 1]), 1.4, atol=0.1)
    # inversion of that inverts
    assert np.isclose(ellipse.segment_area([-1, 1, -1]), 4.9, atol=0.1)
    # outside of ellipse on positive size, so no area
    assert np.isclose(ellipse.segment_area([1, -1, 3]), 0.0)
    # more of the ellipse is in the negative, so should be in (pi, 2 pi)
    assert np.isclose(ellipse.segment_area([1, -1, -1]), 4.9, atol=0.1)
    # inverting line inverts area
    assert np.isclose(ellipse.segment_area([-1, 1, 1]), 1.4, atol=0.1)
    # outside of ellipse on negative side, so full area
    assert np.isclose(ellipse.segment_area([1, -1, -3]), 2 * np.pi)
    # inversion of that, so no area
    assert np.isclose(ellipse.segment_area([-1, 1, 3]), 0.0)


def test_circle_segment_area() -> None:
    circle = Ellipse([0, 0], [1, 1], 0.0)
    # tangent, ouside of circle, so should be zero
    assert np.isclose(circle.segment_area([1, 0, 1]), 0)
    # top corner of circle, so quarter circle area (pi/4) less that triangle area (1/2)
    assert np.isclose(circle.segment_area([-1, -1, 1]), np.pi / 4 - 1 / 2)
    # inverse of that
    assert np.isclose(circle.segment_area([1, 1, -1]), np.pi * 3 / 4 + 1 / 2)
