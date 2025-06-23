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
#

from conics import Conic
from conics.geometry import hnormalized
from conics.geometry import homogeneous
from conics.geometry import line_intersection
from conics.geometry import line_through
from conics.geometry import projectively_unique
import numpy as np


def test_intersecting_lines():
    l1 = line_through([0, 0], [1, 1])
    l2 = line_through([0, 1], [1, 0])

    p = line_intersection(l1, l2)

    np.testing.assert_(not np.isclose(p[-1], 0))

    np.testing.assert_array_equal(hnormalized(p), 0.5)
    np.testing.assert_array_equal(homogeneous([0.5, 0.5]), np.atleast_2d(p / p[-1]).T)


def test_projectively_unique_empty():
    a = projectively_unique(np.empty((3, 0)))
    assert a.shape == (3, 0)


def test_projectively_unique_all():
    a = projectively_unique([[0, 1], [0, 1], [0, 1]])
    np.testing.assert_array_equal(a, [[0], [0], [0]])


def test_projectively_unique():
    a = projectively_unique([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(a, [[1], [1], [1]])


def test_projectively_unique_four_intersections():
    e1 = Conic.from_ellipse([0, 0], [2, 1], np.pi / 4)
    e2 = Conic.from_ellipse([0, 0], [2, 1], np.pi * 3 / 4)

    inter = hnormalized(e1.intersect(e2))
    assert inter.shape == (2, 4)


def test_projectively_unique_one_intersection():
    c1 = Conic.from_circle([0, 0], 1)
    c2 = c1.translate([2 * 1, 0])

    inter = hnormalized(c1.intersect(c2))
    assert inter.shape == (2, 1)


def test_projectively_unique_two_intersections():
    c1 = Conic.from_circle([0, 0], 1)
    c2 = Conic.from_circle([0.5, 0], 1)

    inter = hnormalized(c1.intersect(c2))
    assert inter.shape == (2, 2)
