
# conics - Python library for dealing with conics
#
# Copyright 2019 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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


def hnormalized(p):
    return p[:-1] / p[-1]


def homogeneous(p):
    p = np.atleast_2d(p).T
    return np.vstack((p, np.ones_like(p[0])))


def line_through(a, b):
    start = np.stack((*np.ravel(a), 1))
    end = np.stack((*np.ravel(b), 1))

    return np.cross(start, end)


def line_intersection(l1, l2):
    return np.cross(l1, l2)


def rot2d(alpha):
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s], [s, c]])
