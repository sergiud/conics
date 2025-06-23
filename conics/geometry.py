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

import numpy as np
import numpy.typing as npt


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


def projectively_unique(p: npt.ArrayLike, atol: float = 1e-4) -> np.ndarray:
    n = np.size(p, axis=-1)

    # Pairwise cross-product between each intersection for filtering out
    # multiplicative multiples
    c = np.empty_like(p, shape=(n, n, 3))

    for i in np.arange(n):
        for j in np.arange(n):
            c[i, j] = np.cross(np.take(p, i, axis=-1), np.take(p, j, axis=-1))

    # Compute the norm of the cross-products which is zero for a pair of
    # intersections that are equivalent up to scale
    d = np.linalg.norm(c, axis=-1)
    m = np.isclose(d, 0, atol=atol)
    # The indices of the strictly upper-triangular part of the adjacency
    # matrix
    i, j = np.triu_indices(n, k=1)

    duplicate = np.zeros(n, dtype=bool)

    for k in np.arange(n):
        if duplicate[k]:
            continue

        # Current row of the strictly upper-triangular matrix
        r = m[k][j[k == i]]
        # Determine duplicate intersections by evaluating the norm of the
        # pairwise cross-product with respect to the current intersection
        l = np.nonzero(r)
        # Mark other intersections as duplicates to avoid repeated
        # processing. Account for the offset of the diagonal element.
        duplicate[k + l + 1] = True

    return np.take(p, np.nonzero(~duplicate)[0], axis=-1)
