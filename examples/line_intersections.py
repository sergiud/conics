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

from conics import Conic
from conics.geometry import hnormalized
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


def as_ellipse(c, **kwargs):
    x0, major_minor, angle = c.to_ellipse()
    return Ellipse(x0, *major_minor.ravel() * 2, angle=np.rad2deg(angle), **kwargs)


c1 = Conic.from_circle([0.5, -1], 1)
c2 = c1.translate([0.5, 0]).scale(1.1)

e1 = Conic.from_ellipse([0, 0], [2, 1], np.pi / 4)
e2 = Conic.from_ellipse([0, 0], [2, 1], np.pi * 3 / 4)

c3 = Conic.from_circle([0, 0], 1)
c4 = c3.translate([2 * 1, 0])

alpha = np.deg2rad(-45)
l = np.array([np.cos(alpha), np.sin(alpha), 0])

x = np.linspace(-2, 2)
y = np.linspace(-2, 2)

X, Y = np.meshgrid(x, y)
Z_l = np.dot(np.dstack([X, Y]), l[:-1]) + l[-1]

inter11 = hnormalized(c1.intersect_line(l))
inter12 = hnormalized(c2.intersect_line(l))

inter21 = hnormalized(c3.intersect_line(l))
inter22 = hnormalized(c4.intersect_line(l))

inter31 = hnormalized(e1.intersect_line(l))
inter32 = hnormalized(e2.intersect_line(l))

fig = plt.figure()
ax1, ax2, ax3 = fig.subplots(1, 3)

ax1.set_title('Circles-line')
ax1.add_patch(as_ellipse(c1, facecolor='none', edgecolor='red'))
ax1.add_patch(as_ellipse(c2, facecolor='none', edgecolor='blue'))
ax1.contour(X, Y, Z_l, levels=[0])
ax1.scatter(*inter11)
ax1.scatter(*inter12)

ax2.set_title('Circles-line')
ax2.add_patch(as_ellipse(c3, facecolor='none', edgecolor='red'))
ax2.add_patch(as_ellipse(c4, facecolor='none', edgecolor='blue'))
ax2.contour(X, Y, Z_l, levels=[0])
ax2.scatter(*inter21)
#ax2.scatter(*inter22.T)

ax3.set_title('Ellipses-line')
ax3.add_patch(as_ellipse(e1, facecolor='none', edgecolor='red'))
ax3.add_patch(as_ellipse(e2, facecolor='none', edgecolor='blue'))
ax3.contour(X, Y, Z_l, levels=[0])
ax3.scatter(*inter31)
ax3.scatter(*inter32)

ax1.axis('equal')
ax2.axis('equal')

plt.show()

