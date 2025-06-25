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

"""
Fitting and non-linear refinement of a parabola
===============================================

This example shows a linearly fitted parabola and its non-linear refined
variant.
"""

from conics import Parabola
from conics.fitting import fit_nievergelt
import matplotlib.pyplot as plt
import numpy as np

# x = [-7, -3, 0, 0, 1, 1]
# y = [9, 5, 4, 8, 3, 5]
x = [-1, 2, 5, 10, -4]
y = [1, -2, 3, -4, -3]

pts = np.vstack((x, y))

C = fit_nievergelt(pts, type='parabola', scale=False)

X, Y = np.meshgrid(
    np.linspace(np.min(x) - 3, np.max(x) + 1),
    np.linspace(-1 + np.min(y), np.max(y) + 1),
)
Z = C([X, Y])

p = Parabola.from_conic(C)

p_refined = p.refine(pts)
C_refined = p_refined.to_conic()

contact_pts = p_refined.contact(pts)

Z_refined = C_refined([X, Y])

plt.figure()
plt.axis('equal')

cs = plt.contour(X, Y, Z, colors='blue', levels=[0])
artists1, _ = cs.legend_elements()

cs_refined = plt.contour(X, Y, Z_refined, colors='red', levels=[0])
artists2, _ = cs_refined.legend_elements()

plt.scatter(x, y, label='observations')

for xy in np.dstack((contact_pts.T, pts.T)):
    plt.plot(*xy, '--', c='gray')

plt.scatter(*contact_pts, label='orthogonal contact points')

a, l = plt.gca().get_legend_handles_labels()

plt.legend(artists1 + artists2 + a, ['fitted parabola', 'refined parabola'] + l)
plt.show()
