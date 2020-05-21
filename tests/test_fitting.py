
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

from conics.fitting import fit_nievergelt
import numpy as np


def test_nievergelt_up_parabola():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = +x**2

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, np.zeros((2, )))
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, np.pi / 2)


def test_nievergelt_up_parabola_shifted():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = +x**2 + 5

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [0, 5])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, np.pi / 2)


def test_nievergelt_up_parabola_shifted1():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = +x**2 + 10 * x + 5

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [-5, -20])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, np.pi / 2)


def test_nievergelt_down_parabola():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = -x**2

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, np.zeros((2, )))
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, -np.pi / 2)


def test_nievergelt_down_parabola_shifted():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = -x**2 + 5

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [0, 5])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, -np.pi / 2)


def test_nievergelt_down_parabola_shifted1():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = -x**2 + 10 * x + 5

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [5, 30])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, -np.pi / 2)


def test_nievergelt_down_parabola_shifted1():
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    y = -x**2 + 10 * x + 5

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [5, 30])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, -np.pi / 2)


def test_nievergelt_spaeth_parabola1():
    x = [-6.6, -2.8, -0.2, 0.4, 1.2, 1.4]
    y = [8.8, 5.4, 3.6, 7.8, 3.4, 4.8]

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [1.2, 3.4])
    np.testing.assert_approx_equal(p, 0.5)
    np.testing.assert_approx_equal(alpha, np.deg2rad(126.869897))


def test_nievergelt_spaeth_parabola2():
    x = [-7, -3, 0, 0, 1, 1]
    y = [9, 5, 4, 8, 3, 5]

    C = fit_nievergelt(np.row_stack((x, y)), type='parabola')

    vertex, p, alpha = C.to_parabola()

    np.testing.assert_array_almost_equal(vertex, [0.667771, 3.227661])
    np.testing.assert_approx_equal(p, 0.52615439)
    np.testing.assert_approx_equal(alpha, np.deg2rad(124.306068))
