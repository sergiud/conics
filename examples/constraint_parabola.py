
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

from conics.fitting import fit_dlt
from conics.fitting import fit_nievergelt
import matplotlib.pyplot as plt
import numpy as np

x = [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5]
y = [0, +0.5, 1, 0.5, 1, -0.5, -1, -0.5]


pts = np.vstack((x, y))
C1 = fit_dlt(pts)

C2 = C1.constrain(pts)

C3 = fit_nievergelt(pts)

X, Y = np.meshgrid(np.linspace(np.min(x) - 1, np.max(x) + 1),
                   np.linspace(-1 + np.min(y), np.max(y) + 1))
Z1 = C1(np.vstack((X.ravel(), Y.ravel())))
Z2 = C2(np.vstack((X.ravel(), Y.ravel())))
Z3 = C3(np.vstack((X.ravel(), Y.ravel())))

plt.figure()
plt.contour(X, Y, Z1.reshape(X.shape), levels=0)
plt.contour(X, Y, Z2.reshape(X.shape), levels=0)
plt.contour(X, Y, Z3.reshape(X.shape), levels=0)
plt.scatter(*pts)
plt.show()
