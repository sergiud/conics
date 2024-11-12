
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

from conics.geometry import hnormalized
from conics.geometry import homogeneous
from conics.geometry import line_intersection
from conics.geometry import line_through
import numpy as np


def test_intersecting_lines():
    l1 = line_through([0, 0], [1, 1])
    l2 = line_through([0, 1], [1, 0])

    p = line_intersection(l1, l2)

    np.testing.assert_(not np.isclose(p[-1], 0))

    np.testing.assert_array_equal(hnormalized(p), 0.5)
    np.testing.assert_array_equal(homogeneous(
        [0.5, 0.5]), np.atleast_2d(p / p[-1]).T)
