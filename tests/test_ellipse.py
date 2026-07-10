#
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
#

from conics import Conic
from conics import Ellipse
from conics.fitting import fit_dlt
from conics.fitting import fit_nievergelt
from conics.geometry import rot2d
from scipy import integrate
import numpy as np
import pytest
import warnings


def test_ellipse_fitting():
    x = [1, 2, 5, 7, 9, 3, 6, 8]
    y = [7, 6, 8, 7, 5, 7, 2, 4]
    pts = np.column_stack((x, y))

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


def test_ellipse_contact_at_center():
    # A query point exactly at the ellipse center makes one of the two
    # candidate initial guesses divide by zero (0/0), which used to turn
    # into NaN and crash least_squares instead of falling back to the
    # other, perfectly finite candidate.
    e = Ellipse([0, 0], [2, 1], 0)

    contact = e.contact([[0, 0]])

    np.testing.assert_array_almost_equal(contact, [[0, 1]])


def test_ellipse_refine_converges():
    # The analytic Jacobian unpacked a rotation matrix row as
    # (cos(alpha), -sin(alpha)) into (c, s), instead of (cos(alpha),
    # sin(alpha)). The resulting sign error, together with a missing direct
    # contribution of the center to the residual, only affected the center
    # columns, so refine() made no progress moving the center towards
    # points it was not already centered on.
    true_center = np.array([0.3, -0.2])
    true_major_minor = np.array([2.0, 1.0])
    true_alpha = 0.4

    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    R = rot2d(true_alpha)
    local = np.column_stack(
        (true_major_minor[0] * np.cos(t), true_major_minor[1] * np.sin(t))
    )
    pts = local @ R.T + true_center

    guess = Ellipse([0.5, -0.1], [2.2, 1.1], 0.5)
    fitted = guess.refine(pts)

    np.testing.assert_array_almost_equal(fitted.center, true_center)
    np.testing.assert_array_almost_equal(fitted.major_minor, true_major_minor)
    np.testing.assert_almost_equal(fitted.alpha, true_alpha)


def test_circle():
    c = Conic.from_circle([1, 2], 3)
    C = 4 * c.homogeneous

    c1 = Conic.from_homogeneous(C)

    x0, r = c1.to_circle()
    np.testing.assert_array_almost_equal(x0, [[1], [2]])
    np.testing.assert_almost_equal(r, 3)


def test_circle_negative_scale():
    # A homogeneous conic matrix is only defined up to scale, so negating it
    # still represents the same circle. Recovering the radius from
    # sqrt(det(C33)) discards the sign of that scale factor and used to
    # return the wrong radius whenever the scale was negative.
    c = Conic.from_circle([1, 2], 3)
    c1 = Conic.from_homogeneous(-c.homogeneous)

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


def test_ellipse_segment_area_tangent() -> None:
    # A line exactly tangent to the ellipse (h == 1) is a boundary case of
    # the h >= 1 branch. test_circle_segment_area exercises this only for a
    # circle, where major == minor never puts the major * minor axis scaling
    # to the test, so it is repeated here for a non-circular ellipse.
    ellipse = Ellipse([0, 0], [2, 1], 0.0)
    # tangent at the major-axis vertex (2, 0), center on the negative side
    assert np.isclose(ellipse.segment_area([1, 0, -2]), 2 * np.pi)
    # tangent at the major-axis vertex (-2, 0), center on the positive side
    assert np.isclose(ellipse.segment_area([1, 0, 2]), 0.0)
    # tangent at the minor-axis vertex (0, 1), center on the negative side
    assert np.isclose(ellipse.segment_area([0, 1, -1]), 2 * np.pi)
    # tangent at the minor-axis vertex (0, -1), center on the positive side
    assert np.isclose(ellipse.segment_area([0, 1, 1]), 0.0)


def test_ellipse_segment_area_degenerate_line() -> None:
    # line[:2] == (0, 0) makes the transformed line direction vanish
    # regardless of rotation or axis scaling, so norm == 0 and the sign of
    # center_distance alone decides between the full area and zero.
    ellipse = Ellipse([0, 0], [2, 1], 0.3)
    assert np.isclose(ellipse.segment_area([0, 0, -1]), ellipse.area)
    assert np.isclose(ellipse.segment_area([0, 0, 1]), 0.0)


def test_circle_segment_area() -> None:
    circle = Ellipse([0, 0], [1, 1], 0.0)
    # tangent, ouside of circle, so should be zero
    assert np.isclose(circle.segment_area([1, 0, 1]), 0)
    # top corner of circle, so quarter circle area (pi/4) less that triangle area (1/2)
    assert np.isclose(circle.segment_area([-1, -1, 1]), np.pi / 4 - 1 / 2)
    # inverse of that
    assert np.isclose(circle.segment_area([1, 1, -1]), np.pi * 3 / 4 + 1 / 2)


def _quadrature_segment_area(ellipse, line):
    """Independently integrates the half plane area cut from an ellipse.

    Used as a ground truth for segment_area() that does not share any code
    with its implementation.
    """
    major, minor = ellipse.major_minor
    R = rot2d(ellipse.alpha)

    def indicator(y, x):
        p = np.array([x, y]) @ R.T + ellipse.center
        return 1.0 if (p @ line[:2] + line[2]) < 0 else 0.0

    def y_lo(x):
        return -minor * np.sqrt(np.maximum(0.0, 1 - (x / major) ** 2))

    def y_hi(x):
        return minor * np.sqrt(np.maximum(0.0, 1 - (x / major) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', integrate.IntegrationWarning)
        value, _ = integrate.dblquad(indicator, -major, major, y_lo, y_hi)

    return value


@pytest.mark.parametrize(
    'line',
    [
        np.array([1.0, 0.0, -1.8]),
        np.array([np.cos(2.5), np.sin(2.5), 0.9]),
    ],
)
def test_ellipse_segment_area_asymmetric_cut(line) -> None:
    # A line offset that is not symmetric about the ellipse center used to
    # make segment_area return the area of the wrong side (its
    # region-selection condition was inverted). The reference is computed by
    # direct numerical integration, independently of segment_area's own
    # implementation.
    ellipse = Ellipse([0, 0], [2, 1], 0.0)

    area = ellipse.segment_area(line)
    expected = _quadrature_segment_area(ellipse, line)

    np.testing.assert_allclose(area, expected, atol=1e-3)


def _unit_circle_cap_quadrature(h: float) -> float:
    """Independently integrates the area of the unit circle cap {x > h}.

    Used as a ground truth that does not share segment_area()'s own
    arccos/arctan2 evaluation, so it stays a valid reference even for h close
    to 1.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', integrate.IntegrationWarning)
        value, _ = integrate.quad(
            lambda x: 2 * np.sqrt(np.maximum(0.0, 1 - x**2)),
            h,
            1.0,
            epsabs=1e-14,
            epsrel=1e-14,
            limit=200,
        )

    return value


@pytest.mark.parametrize('h', [1 - 1e-6, 1 - 1e-8])
def test_ellipse_segment_area_near_tangent_cut(h) -> None:
    # A near-tangent cut drives h = |C| / norm close to 1. Computing
    # arccos(h) - h * sqrt(1 - h**2) directly loses precision there because
    # 1 - h**2 cancels leading digits (h**2 rounds close to 1 before the
    # subtraction). At h = 1 - 1e-8 that formula is off by about 2%. The
    # reference is computed independently by 1-D quadrature.
    circle = Ellipse([0, 0], [1, 1], 0.0)
    # {-x + h < 0} == {x > h}, the cap not containing the center, so
    # segment_area returns the cap area directly rather than
    # total_area - cap_area, keeping the tiny cap value from being swallowed
    # by cancellation against the much larger total area.
    line = np.array([-1.0, 0.0, h])

    area = circle.segment_area(line)
    expected = _unit_circle_cap_quadrature(h)

    np.testing.assert_allclose(area, expected, rtol=1e-3)
