
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

from conics import Conic
from conics.geometry import hnormalized
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


def as_ellipse(c, **kwargs):
    x0, major_minor, angle = c.to_ellipse()
    return Ellipse(x0, *major_minor.ravel() * 2, angle=np.rad2deg(angle), **kwargs)


c1 = Conic.from_circle([0, 0], 1)
c2 = Conic.from_circle([0.5, 0], 1)

e1 = Conic.from_ellipse([0, 0], [2, 1], np.pi / 4)
e2 = Conic.from_ellipse([0, 0], [2, 1], np.pi * 3 / 4)

c3 = Conic.from_circle([0, 0], 1)
c4 = c3.translate([2 * 1, 0])

inter1 = hnormalized(c1.intersect(c2))
inter2 = hnormalized(e1.intersect(e2))
inter3 = hnormalized(c3.intersect(c4))

fig = plt.figure()
ax1, ax2, ax3 = fig.subplots(1, 3)

ax1.set_title('Circles')
ax1.add_patch(as_ellipse(c1, facecolor='none', edgecolor='red'))
ax1.add_patch(as_ellipse(c2, facecolor='none', edgecolor='blue'))
ax1.scatter(inter1[0], inter1[1])

ax2.set_title('Circles')
ax2.add_patch(as_ellipse(c3, facecolor='none', edgecolor='red'))
ax2.add_patch(as_ellipse(c4, facecolor='none', edgecolor='blue'))
ax2.scatter(inter3[0], inter3[1])

ax3.set_title('Ellipses')
ax3.add_patch(as_ellipse(e1, facecolor='none', edgecolor='red'))
ax3.add_patch(as_ellipse(e2, facecolor='none', edgecolor='blue'))
ax3.scatter(inter2[0], inter2[1])

ax1.axis('equal')
ax2.axis('equal')

plt.show()
