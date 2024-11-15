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
