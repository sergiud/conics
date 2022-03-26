
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

import numpy as np
from conics import Ellipse
from conics.fitting import fit_dlt
from conics.fitting import fit_nievergelt

def test_ellipse_fitting():
    pts = np.array([ [1, 2, 5, 7, 9, 3, 6, 8], [7, 6, 8, 7, 5, 7, 2, 4]])

    C = fit_dlt(pts)
    C = fit_nievergelt(pts, type='ellipse', scale=True)
    e = Ellipse.from_conic(C)
    e = Ellipse([4.84, 4.979], [3.391, 3.391], 0)

    #print(e.center, e.major_minor, e.alpha)

    e1 = e.refine(pts)
    print(e1.center, e1.major_minor, e1.alpha)