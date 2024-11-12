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
from conics import Parabola
from conics.fitting import fit_nievergelt
import numpy as np


def test_up_parabola_contact_points():
    x = np.arange(-4, 4 + 1)
    y = x**2 + 5

    pts = np.vstack((x, y))

    C = fit_nievergelt(pts, type='parabola', scale=True)
    p = Parabola.from_conic(C)

    contact_pts = p.contact(pts)

    np.testing.assert_array_almost_equal(pts, contact_pts)


def test_parabola1():
    x = [-1, 2, 5, 10, -4]
    y = [1, -2, 3, -4, -3]

    pts = np.vstack((x, y))

    p1 = Parabola([-4.707, -1.269], 0.512, 0.088)

    contact_pts = p1.contact(pts)
    C = p1.to_conic()
    vals = C(contact_pts)

    np.testing.assert_array_almost_equal(np.sum(vals), 0)

    p2 = p1.refine(pts)

    np.testing.assert_approx_equal(p2.p, 0.38164, significant=5)
    np.testing.assert_approx_equal(p2.alpha, 0.08523, significant=5)
    np.testing.assert_array_almost_equal(p2.vertex, [-6.73135, -1.30266],
                                         decimal=3)


def test_parabola_conversion1():
    c = Conic.from_parabola([1, 2], 1, np.pi / 2)
    vertex, p, alpha = c.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [1, 2])
    np.testing.assert_approx_equal(p, 1)
    np.testing.assert_approx_equal(alpha, np.pi / 2)


def test_parabola_conversion2():
    c = Conic.from_parabola([1, 2], 1, -np.pi / 2)
    vertex, p, alpha = c.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [1, 2])
    np.testing.assert_approx_equal(p, 1)
    np.testing.assert_approx_equal(alpha, -np.pi / 2)


def test_parabola_conversion3():
    c = Conic.from_parabola([1, 2], -1, -np.pi / 2)
    vertex, p, alpha = c.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [1, 2])
    np.testing.assert_approx_equal(p, 1)
    np.testing.assert_approx_equal(alpha, np.pi / 2)
