
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

from ._nievergelt import fit_nievergelt
from .conic import Conic


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    x = [-7, -3, 0, 0, 1, 1]
    y = [9, 5, 4, 8, 3, 5]

    x = [-6.6, -2.8, -0.2, 0.4, 1.2, 1.4]
    y = [8.8, 5.4, 3.6, 7.8, 3.4, 4.8]

    y = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    x = [3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 5]

    x = [-1.1, 0, 0, 0, -0.5]
    y = [0, 1, 2, 3, 4]

    #x = np.array([-4, -2, -1, 0, 1, 2, 4])
    # y = 2*x**2+1*x+5  # +0.25*x**2+0
    #y = np.array([-4, -2, -1, 0, 1, 2, 4])
    # x = -2*y**2-0.5#+0.25*x**2+0
    #x = [1, 0, 3, 4, 5]
    #y = [3, -1, 10, 50, 100]

    #x = [-34.75, -22, -15.5, -8.0, -4, -1.0, 1.5, 4.5, 9.25, 17, 23.5, 36, 64.5]
    #y = [20.25, 17, 15.0, 13.5, 13, 12.5, 12.5, 13.0, 14.00, 16, 18.0, 21, 29.5]

    pts = np.row_stack((x, y)).astype(np.float)
    #pts[1] *= 1e-2
    #C = fit_nievergelt(pts, type='hyperbola')
    C = fit_nievergelt(pts, type='parabola', scale=True)

    print(C)

    plt.figure()
    plt.axis('equal')

    vertex, p, alpha = C.to_parabola()

    x, y = pts

    X, Y = np.meshgrid(np.linspace(np.min(x)-1, np.max(x)+1),
                       np.linspace(-1+np.min(y), np.max(y)+1))
    Z = C(np.row_stack((X.ravel(), Y.ravel())))

    plt.contour(X, Y, Z.reshape(X.shape), levels=0)
    plt.scatter(x, y)
    plt.scatter(*vertex)
    print(vertex)

    plt.show()
