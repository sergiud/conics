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


def hnormalized(p: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return p[:-1] / p[-1]


def homogeneous(p: npt.ArrayLike) -> npt.NDArray[np.floating]:
    p = np.atleast_2d(p).T
    return np.vstack((p, np.ones_like(p[0])))


def line_through(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.floating]:
    start = np.append(a, 1)
    end = np.append(b, 1)

    return np.cross(start, end)


def line_intersection(l1: npt.ArrayLike, l2: npt.ArrayLike) -> npt.NDArray[np.floating]:
    return np.cross(l1, l2)


def rot2d(alpha: float) -> npt.NDArray[np.float64]:
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def projectively_unique(p: npt.ArrayLike, atol: float = 1e-4) -> np.ndarray:
    p = np.asarray(p)
    n = p.shape[-1]
    # The indices of the strictly upper-triangular part of the adjacency
    # matrix
    i, j = np.triu_indices(n, k=1)

    # Pairwise cross-product between each intersection for filtering out
    # multiplicative multiples
    c = np.cross(p.T[i], p.T[j])

    # Compute the norm of the cross-products which is zero for a pair of
    # intersections that are equivalent up to scale
    d = np.linalg.norm(c, axis=-1)
    m = np.isclose(d, 0, atol=atol)

    # mark points as duplicate if cross product with an earlier point has 0 norm
    duplicate = np.bincount(j, m) > 0
    return p[:, ~duplicate]
