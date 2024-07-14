
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

from ._conic import Conic
import numpy as np


def _build_scatter_matrices(pts):
    x, y = pts

    x2 = x**2
    xy = x * y
    y2 = y**2

    x2m = np.mean(x2)
    xym = np.mean(xy)
    y2m = np.mean(y2)

    x2 -= x2m
    xy -= xym
    y2 -= y2m

    D1 = pts.T
    D2 = np.column_stack((x2, xy, y2))
    S22 = D2.T @ D2
    S21 = D2.T @ D1
    S11 = D1.T @ D1

    return S11, S21, S22, x2m, xym, y2m


def _build_reduced_scatter_matrix(S11, S21, S22):
    S11invS21T = np.linalg.solve(S11, S21.T)
    M = S22 - S21 @ S11invS21T

    return M, S11invS21T


def _compute_parabola(M):
    evals, evecs = np.linalg.eigh(M)

    evecs = evecs[..., ::-1]
    evals = evals[::-1]

    e11, e12, e13, e21, e22, e23, e31, e32, e33 = evecs.ravel(order='F')

    gamma1 = e22**2 - 4 * e21 * e23  # s^2
    gamma2 = -4 * e13 * e21 + 2 * e12 * e22 - 4 * e11 * e23  # st
    gamma3 = e12**2 - 4 * e11 * e13  # t^2
    gamma4 = -4 * e23 * e31 + 2 * e22 * e32 - 4 * e21 * e33  # s
    gamma5 = -4 * e13 * e31 + 2 * e12 * e32 - 4 * e11 * e33  # t
    gamma6 = e32**2 - 4 * e31 * e33  # constant term

    alpha1, alpha2 = evals[:2]**2
    alpha3 = alpha1 * alpha2

    k1 = 4 * gamma3 * gamma6 - gamma5**2
    k2 = gamma2 * gamma6 - 0.5 * gamma4 * gamma5
    k3 = 0.5 * gamma2 * gamma5 - gamma3 * gamma4
    k4 = 4 * gamma6 * gamma1 - gamma4**2
    k5 = 4 * gamma1 * gamma3 - gamma2**2
    k6 = gamma2 * gamma4 - 2 * gamma1 * gamma5
    k7 = -4 * (gamma1 * alpha1 + alpha2 * gamma3)
    k8 = gamma1 * k1 - gamma2 * k2 + gamma4 * k3

    K0 = 16 * gamma6 * alpha3**2
    K1 = -8 * alpha3 * (k1 * alpha2 + k4 * alpha1)
    K2 = 4 * ((2 * gamma2 * k2 + 4 * k8) * alpha3 + gamma1 *
              k4 * alpha1**2 + gamma3 * K1 * alpha2**2)
    K3 = 2 * k7 * k8
    K4 = k5 * k8

    r = np.roots([K4, K3, K2, K1, K0])

    u = k5 * r**2 + k7 * r + 4 * alpha3
    r_by_u = np.divide(r, u, out=u, where=u != 0)

    s = 2 * r_by_u * (k3 * r + alpha1 * gamma4)
    t = r_by_u * (k6 * r + 2 * alpha2 * gamma5)

    ust = np.vstack((np.ones_like(s), s, t))

    theta = evecs[..., ::-1] @ ust
    a, b, c = theta
    error = b**2 - 4 * a * c

    k = np.argmin(np.abs(error))

    return np.real(theta[..., k])


def _partial_backsubstitute(S1121):
    z_eh = np.block([[np.eye(3)], [-S1121]])
    return z_eh


def _backsubstitute(S1121, z2, x2m, xym, y2m):
    z_eh = _partial_backsubstitute(S1121)

    B = np.block([[z_eh], [-x2m, -xym, -y2m]])
    K = B @ z2

    return K


def _denormalize(z, scale_inv, mean):
    K = Conic(z)

    TT = scale_inv * np.eye(2)
    T = np.block([[TT, -scale_inv * mean[..., np.newaxis]],
                  [0, 0, 1]])

    return K.transform(T, invert=False)


def _diff2_mu_sqr_error(mu, Q0, Q1, Q2, R0, R1, R2):
    return 2 * (2 * Q2 * mu + Q1) / (R2 * mu**2 + R1 * mu + R0)**2 - 4 * (Q2 *
                                                                          mu**2 + Q1 * mu + Q0) * (2 * R2 * mu + R1) / (R2 * mu**2 + R1 * mu + R0)**3


def _correct_bias(pts, S11, S21, S22, z_e, z_h, x2m, xym, y2m, k_e, k_h):
    x, y = pts

    zeros = np.zeros_like(x)
    ones = np.ones_like(x)

    Dx = np.column_stack((2 * x, y, zeros, ones, zeros))
    Dy = np.column_stack((zeros, x, 2 * y, zeros, ones))

    Sxy = Dx.T @ Dx + Dy.T @ Dy
    S = np.block([[S22, S21],
                  [S21.T, S11]])

    sigma1 = z_e.T @ S @ z_e
    sigma2 = z_e.T @ S @ z_h
    sigma3 = z_h.T @ S @ z_h
    sigma4 = z_e.T @ Sxy @ z_e
    sigma5 = z_e.T @ Sxy @ z_h
    sigma6 = z_h.T @ Sxy @ z_h

    Q2 = sigma2 * sigma4 - sigma2 * sigma6 - sigma3 * sigma4 + \
        sigma3 * sigma5 - sigma1 * sigma5 + sigma1 * sigma6
    Q1 = sigma3 * sigma4 + 2 * sigma1 * sigma5 - \
        sigma1 * sigma6 - 2 * sigma2 * sigma4
    Q0 = sigma2 * sigma4 - sigma1 * sigma5

    mu12 = np.roots([Q2, Q1, Q0])

    # Evaluate second order derivative to minimize for mu.
    R2 = sigma4 - 2 * sigma5 + sigma6
    R1 = 2 * sigma5 - 2 * sigma4
    R0 = sigma4

    d_e2_d_mu = _diff2_mu_sqr_error(mu12, Q0, Q1, Q2, R0, R1, R2)
    mask = d_e2_d_mu > 0

    # Second derivative test
    if np.any(mask):
        mu = mu12[mask][0]
    else:
        mu = 0  # No minimum

    # mu12_2 = np.roots([k_e + k_h, -2 * k_e, k_e])

    # Backsubstitute mu into the conic pencil.
    z_mu = (1 - mu) * z_e + mu * z_h

    factor = np.column_stack((x2m, xym, y2m, 0, 0))
    z0 = -factor @ z_mu
    z = np.stack((*z_mu, *z0))

    return z


def _compute_ellipse(M):
    C = np.diag([-2, 1, -2])[..., ::-1]

    evals, evecs = np.linalg.eig(np.linalg.solve(C, M))
    k = np.diagonal(evecs.T @ C @ evecs)

    u = np.argmin(k)

    idxs = np.arange(k.size)
    mask = (idxs != u) & (k > 0)
    idxs = idxs[mask]

    v = idxs[np.argmin(np.abs(k[idxs]))]

    z2_e, z2_h = evecs[..., [u, v]].T

    k_e, k_h = k[[u, v]]

    return z2_e, z2_h, k_e, k_h


def fit_harker(pts, type):
    mean = np.mean(pts, axis=1)
    centered = pts - mean[..., np.newaxis]

    scale = np.sqrt(np.mean(centered**2))
    scale_inv = 1 / scale
    normalized = centered * scale_inv

    S11, S21, S22, x2m, xym, y2m = _build_scatter_matrices(
        normalized.astype(scale.dtype))

    M, S11invS21T = _build_reduced_scatter_matrix(S11, S21, S22)

    if type == 'ellipse' or type == 'hyperbola':
        z2_e, z2_h, k_e, k_h = _compute_ellipse(M)

        if type == 'ellipse':
            Btop = _partial_backsubstitute(S11invS21T)
            z_e, z_h = (Btop @ np.column_stack((z2_e, z2_h))).T
            z = _correct_bias(normalized, S11, S21, S22, z_e, z_h, x2m, xym,
                              y2m, k_e, k_h)
        else:  # hyperbola
            z = _backsubstitute(S11invS21T, z2_h, x2m, xym, y2m)
    elif type == 'parabola':
        z2 = _compute_parabola(M)
        z = _backsubstitute(S11invS21T, z2, x2m, xym, y2m)
    else:
        raise ValueError('unsupported conic type {}'.format(type))

    C = _denormalize(z, scale_inv, mean)

    return C


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = [-7, -3, 0, 0, 1, 1]
    y = [9, 5, 4, 8, 3, 5]

    t = np.linspace(-np.pi, np.pi)
    x = np.cos(t) + np.random.normal(scale=0.05, size=t.size)
    y = np.sin(t) + np.random.normal(scale=0.05, size=t.size)

    # y = np.linspace(-5, 4, num=250)
    # x = -y**2 + 2 * y - 5 + np.random.normal(scale=0.25, size=y.size)

    # x = [-6.6, -2.8, -0.2, 0.4, 1.2, 1.4]
    # y = [8.8, 5.4, 3.6, 7.8, 3.4, 4.8]

    # pts = rot2d(np.pi / 4 * 0) @ np.vstack((x, y))
    pts = np.vstack((x, y))

    C = fit_harker(pts, type='ellipse')

    print(C(pts))

    X, Y = np.meshgrid(np.linspace(np.min(pts[0]) - 1, np.max(pts[0]) + 1),
                       np.linspace(-1 + np.min(pts[1]), np.max(pts[1]) + 1))
    Z = C(np.vstack((X.ravel(), Y.ravel())))

    plt.figure()
    plt.contour(X, Y, Z.reshape(X.shape), levels=0)
    plt.scatter(*pts)
    plt.show()
