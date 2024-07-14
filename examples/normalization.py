
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

from conics.fitting import fit_nievergelt
import matplotlib.pyplot as plt
import numpy as np

y = [-2, -0.1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
x = [1, 0.5, 0.1, 0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.6]

pts = np.vstack((x, y))

C1 = fit_nievergelt(pts, type='parabola', scale=False)
C2 = fit_nievergelt(pts, type='parabola', scale=True)

X, Y = np.meshgrid(np.linspace(np.min(x) - 1, np.max(x) + 1),
                   np.linspace(-1 + np.min(y), np.max(y) + 1))
Z1 = C1(np.vstack((X.ravel(), Y.ravel())))
Z2 = C2(np.vstack((X.ravel(), Y.ravel())))

fig = plt.figure()
# fig.set_aspect('equal')

ax1, ax2 = fig.subplots(1, 2)

# ax1.set_aspect('equal', 'box')
# ax2.set_aspect('equal', 'box')

cs = ax1.contour(X, Y, Z1.reshape(X.shape), levels=0)
# cs.collections[1].set_label('fitted parabola (unscaled)')

cs_refined = ax2.contour(X, Y, Z2.reshape(X.shape), colors='red', levels=0)
# cs_refined.collections[1].set_label('fitted parabola (scaled)')

ax1.scatter(x, y, label='observations')
ax2.scatter(x, y)

ax1.set_title('without scaling')
ax2.set_title('with scaling')

fig.legend(loc='upper center')
plt.show()
