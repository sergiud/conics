# conics - Python library for dealing with conics
#
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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
Parabola to quadratic Bézier curve conversion
=============================================

A parabola can be represented exactly by a quadratic (second-degree) Bézier
curve. This example fits a parabola to a set of observations and converts the
result into the corresponding Bézier control points.
"""

from conics import Parabola
from conics.fitting import fit_nievergelt
from conics.fitting import parabola_to_bezier
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

x = [-7, -3, 0, 0, 1, 1]
y = [9, 5, 4, 8, 3, 5]

pts = np.column_stack((x, y))

C = fit_nievergelt(pts, type='parabola', scale=True)
pb = Parabola.from_conic(C)

control_points = parabola_to_bezier(pb, *pts[[0, -3]])
s1, inter, s2 = control_points

X, Y = np.meshgrid(
    np.linspace(np.min(x) - 1, np.max(x) + 1),
    np.linspace(-1 + np.min(y), np.max(y) + 1),
)
Z = C(np.dstack([X, Y]))

fig = plt.figure()

plt.contour(X, Y, Z, levels=0)

plt.scatter(*pts.T, label='observations')

path = mpatches.Path(
    control_points, [mpatches.Path.MOVETO, mpatches.Path.CURVE3, mpatches.Path.CURVE3]
)
pp = mpatches.PathPatch(
    path, fill=False, linestyle='--', edgecolor='blue', lw=3, label='Bezier curve'
)

plt.gca().add_patch(pp)

plt.plot(*control_points.T, '--', c='gray')
plt.scatter(*control_points.T, label='control points')

for i, xy in enumerate(control_points):
    plt.annotate('$p_{}$'.format(i), xy)

plt.legend()

plt.show()
