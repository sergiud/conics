
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

from conics import Parabola
from conics.fitting import fit_nievergelt
from conics.fitting import parabola_to_bezier
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

x = [-7, -3, 0, 0, 1, 1]
y = [9, 5, 4, 8, 3, 5]

pts = np.vstack((x, y))

C = fit_nievergelt(pts, type='parabola', scale=True)
pb = Parabola.from_conic(C)

control_points = parabola_to_bezier(pb, *pts[:, [0, -3]].T)
s1, inter, s2 = control_points.T

X, Y = np.meshgrid(np.linspace(np.min(x) - 1, np.max(x) + 1),
                   np.linspace(-1 + np.min(y), np.max(y) + 1))
Z = C(np.vstack((X.ravel(), Y.ravel())))

fig = plt.figure()

plt.contour(X, Y, Z.reshape(X.shape), levels=0)

plt.scatter(*pts, label='observations')

path = mpatches.Path(control_points.T, [mpatches.Path.MOVETO,
                                        mpatches.Path.CURVE3, mpatches.Path.CURVE3])
pp = mpatches.PathPatch(
    path,
    fill=False,
    linestyle='--',
    edgecolor='blue',
    lw=3, label='Bezier curve')

plt.gca().add_patch(pp)

plt.plot(*control_points, '--', c='gray')
plt.scatter(*control_points, label='control points')

for i, xy in enumerate(control_points.T):
    plt.annotate('$p_{}$'.format(i), xy)

plt.legend()

plt.show()
