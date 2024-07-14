
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

from conics import Ellipse
from conics.fitting import fit_dlt
from conics.fitting import fit_nievergelt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

pts = np.asarray([[.35, 1.2], [1.5, 1.2], [-2.3, 4.2], [-1, -1.2], [0, -2], [-1,
                                                                             -3]]).T

pts = np.array([[1, 2, 5, 7, 9, 3, 6, 8], [7, 6, 8, 7, 5, 7, 2, 4]])
# pts = np.array([ [1, 2, 5, 7, 9, 6, 3, 8], [7, 6, 8, 7, 5, 7, 2, 4]])

C = fit_dlt(pts)
C = fit_nievergelt(pts, type='ellipse', scale=True)
e = Ellipse.from_conic(C)
# e = Ellipse([5, 4], [3, 2], np.pi/4)

e1 = e.refine(pts)
contact_pts = e.contact(pts)

width, height = 2 * np.asarray(e.major_minor)
ee = mpatches.Ellipse(e.center, width, height, np.rad2deg(e.alpha), edgecolor='red',
                      facecolor='none', lw=2)

width1, height1 = 2 * np.asarray(e1.major_minor)
ee1 = mpatches.Ellipse(e1.center, width1, height1, np.rad2deg(e1.alpha),
                       edgecolor='blue', facecolor='none', lw=2)

plt.figure()
plt.axis('equal')
plt.gca().add_patch(ee)
plt.gca().add_patch(ee1)
plt.scatter(*pts)
plt.scatter(*contact_pts)
plt.show()
