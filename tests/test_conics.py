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


"""Tests for `conics` package."""

from conics import Conic
from conics import estimate_pose
from conics import projected_center
from conics import surface_normal
from conics._conic import _make_circle
from conics._conic import concentric_conics_vanishing_line
from conics.geometry import hnormalized
import numpy as np
import pytest


def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), +np.cos(theta)]])


def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, +np.cos(theta)]])


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), +np.cos(theta), 0],
                     [0, 0, 1]])


def test_foo():
    r = 1
    C = _make_circle([1, 0], r)
    c0 = Conic.from_homogeneous(C)

    angle = 1 * -np.pi / 7
    R0 = Rx(np.pi / 6) @ Ry(1 * np.pi / 3) @ Rz(angle)
    tc = c0.transform(R0.T, R0, invert=False)
    # tc = c0.transform(R0)

    CC = tc.homogeneous
    # CC = R0.T @ C @ R0

    R, t, n, c, s1, s2, s3, mask = estimate_pose(CC, r, -angle / 2)

    # diffs = R[mask] - R0.T
    # fn = np.linalg.norm(diffs, 'fro', axis=(1, 2))

    # inv_c = tc.transform(R[mask][0])


def test_construction():
    c1 = Conic(1, 2, 3, 4, 5, 6)  # noqa: F841
    c2 = Conic([1, 2, 3, 4, 5, 6])  # noqa: F841

    np.testing.assert_raises(ValueError, Conic, 1)
    np.testing.assert_raises(ValueError, Conic, 1, 2)


def test_unit_circle():
    c = Conic.from_ellipse((0, 0), (1, 1), 0)
    x0, major_minor, alpha = c.to_ellipse()

    np.testing.assert_equal(np.ravel(x0), (0, 0))
    np.testing.assert_equal(np.ravel(major_minor), (1, 1))
    assert alpha == 0


def test_circle():
    c = Conic.from_ellipse((0, 0), (5, 5), 0)
    x0, major_minor, alpha = c.to_ellipse()

    np.testing.assert_equal(np.ravel(x0), (0, 0))
    np.testing.assert_equal(np.ravel(major_minor), (5, 5))
    assert alpha == 0


def test_ellipse_center():
    x0_true = (1, 2)
    c1 = Conic.from_ellipse(x0_true, (6, 5), np.pi / 5)

    x0, major_minor, alpha = c1.to_ellipse()

    np.testing.assert_equal(x0_true, x0.ravel())


def test_conic_scale_indeterminacy():
    c1 = Conic.from_ellipse((1, 2), (6, 5), np.pi / 5)
    x0, major_minor, alpha = c1.to_ellipse()

    c2 = Conic.from_homogeneous(c1.homogeneous * np.random.random())

    x00, major_minor0, alpha0 = c2.to_ellipse()

    np.testing.assert_almost_equal(x0, x00)
    np.testing.assert_almost_equal(major_minor, major_minor0)
    np.testing.assert_almost_equal(alpha, alpha0)


def test_shifted_circle():
    c = Conic.from_ellipse((1, 2), (5, 5), 0)
    x0, major_minor, alpha = c.to_ellipse()

    np.testing.assert_equal(np.ravel(x0), (1, 2))
    np.testing.assert_equal(np.ravel(major_minor), (5, 5))

    assert alpha == 0


def test_homogeneous_circle():
    x0 = np.array([[0], [0]])
    r = 100
    # f = 1

    C = _make_circle(x0, r)
    c = Conic.from_homogeneous(C)

    x00, major_minor0, alpha = c.to_ellipse()

    np.testing.assert_equal(x0, x00)
    np.testing.assert_equal(major_minor0, r)


def test_homogeneous_to_inhomogeneous():
    c1 = Conic.from_ellipse((1, 2), (4, 3), np.pi / 2)
    C = c1.homogeneous
    c2 = Conic.from_homogeneous(C)

    np.testing.assert_equal(c1.coeffs_, c2.coeffs_)


def test_transform_translate():
    x0 = np.array([1, 2])
    major_minor = np.array([4, 3])
    angle = np.pi / 3
    c1 = Conic.from_ellipse(x0, major_minor, angle)

    t = np.array([2, 3])

    c2 = c1.translate(t)

    np.testing.assert_allclose(c2.center.ravel(), x0 + t)

    x00, major_minor0, angle0 = c2.to_ellipse()

    np.testing.assert_allclose(x00.ravel(), x0 + t)
    np.testing.assert_allclose(major_minor0.ravel(), major_minor)
    np.testing.assert_allclose(angle0, angle)


def test_transform_rotate_and_translate():
    x0 = np.array([1, 2])
    major_minor = np.array([4, 3])
    angle = np.pi / 3
    c1 = Conic.from_ellipse(x0, major_minor, angle)

    by = np.pi / 10
    c2 = c1.translate(-x0).rotate(by).translate(x0)

    x00, major_minor0, angle0 = c2.to_ellipse()

    np.testing.assert_allclose(x00.ravel(), x0)
    np.testing.assert_allclose(major_minor0.ravel(), major_minor)
    np.testing.assert_allclose(angle + by, angle0)


def test_conic_normalizaiton():
    x0 = np.array([1, 2])
    major_minor = np.array([4, 3])
    angle = np.pi / 3
    c1 = Conic.from_ellipse(x0, major_minor, angle)

    c2 = c1.normalize(d=-1)
    det = np.linalg.det(c2.homogeneous)

    np.testing.assert_almost_equal(det, -1)


def test_circle_determinant():
    C = _make_circle([1, 0], r=1)
    np.testing.assert_almost_equal(np.linalg.det(C), -1)


def test_circle_surface_normal():
    x0 = np.array([[0], [0]])
    r = 100
    # f = 1
    C = _make_circle(x0, r)

    n, d = surface_normal(C)

    np.testing.assert_equal(n, [0, 0, 1])
    np.testing.assert_equal(d, 1)

    # Projection center

    c = projected_center(C, n)
    print('C', c)


def test_pose_estimation():
    x0 = np.array([[0], [0]])
    r = 100
    # f = 1
    C = _make_circle(x0, r)

    R, t, n, c, s1, s2, s3, mask = estimate_pose(C, 1, 0)

    RR = R[mask]
    tt = t[mask]
    nn = n[mask]

    np.testing.assert_equal(np.eye(3), RR[0])
    np.testing.assert_equal(np.eye(3), RR[1])

    np.testing.assert_equal([0, 0, -1 / r], np.ravel(tt[0]))
    np.testing.assert_equal([0, 0, -1 / r], np.ravel(tt[1]))

    np.testing.assert_equal([0, 0, 1], np.ravel(nn[0]))
    np.testing.assert_equal([0, 0, 1], np.ravel(nn[1]))

    unit_z = np.array([[0], [0], [1]])

    normals = RR @ unit_z

    np.testing.assert_equal(nn, normals)


def test_concentric_circles():
    x0 = [5, 4]
    A1 = _make_circle([5, 4], 10)
    A2 = _make_circle([5, 4], 20)

    x00, _ = concentric_conics_vanishing_line(A2, A1)

    np.testing.assert_almost_equal(x0, hnormalized(x00))


def test_single_circle_intersection():
    c1 = Conic.from_circle([0, 0], 1)
    c2 = Conic.from_circle([2, 0], 1)

    inter = c1.intersect(c2)
    den = inter[-1, ...]
    inter = np.real(inter[..., np.isreal(den)])
    # Remove duplicate columns
    inter = np.unique(inter, axis=1)

    hinter = hnormalized(inter)
    hinter = np.unique(hinter, axis=1)

    assert hinter.shape[1] == 1

    np.testing.assert_array_almost_equal(hinter, [[1], [0]])

    c = c1 - c2

    np.testing.assert_array_almost_equal(c(inter), 0)
    np.testing.assert_array_almost_equal(c(hinter), 0)


def test_conic_from_homogeneous_non_symmetric():
    Q = np.arange(9).reshape(3, 3)

    with pytest.raises(ValueError):
        c = Conic.from_homogeneous(Q)  # noqa: F841


def test_circle_scale():
    c1 = Conic.from_circle([1, 2], 5)
    c2 = c1.scale(2)

    center, major_minor, alpha = c2.to_ellipse()

    np.testing.assert_array_almost_equal(center.ravel(), [2, 4])
    np.testing.assert_array_almost_equal(major_minor.ravel(), [10, 10])


def test_circle_scale_non_uniform():
    c1 = Conic.from_circle([1, 2], 5)
    c2 = c1.scale(2, 4)

    center, major_minor, alpha = c2.to_ellipse()

    np.testing.assert_array_almost_equal(center.ravel(), [2, 8])
    np.testing.assert_array_almost_equal(major_minor.ravel(), [20, 10])


def test_complex_circle_intersection():
    c1 = Conic.from_circle([0, 0], 1)
    c2 = Conic.from_circle([3, 0], 1)

    c = c1 - c2

    inter = c1.intersect(c2)

    np.testing.assert_almost_equal(c(inter), 0)
