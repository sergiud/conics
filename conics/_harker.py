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

from ._conic import Conic
from ._parabola import Parabola
from ._conic import rot2d
import numpy as np
import scipy.linalg


def _build_scatter_matrices(pts):
    x, y = pts.T

    x2 = x**2
    xy = x * y
    y2 = y**2

    x2m = np.mean(x2)
    xym = np.mean(xy)
    y2m = np.mean(y2)

    x2 -= x2m
    xy -= xym
    y2 -= y2m

    D1 = pts
    D2 = np.column_stack((x2, xy, y2))
    S22 = D2.T @ D2
    S21 = D2.T @ D1
    S11 = D1.T @ D1

    return S11, S21, S22, x2m, xym, y2m


def _build_reduced_scatter_matrix(S11, S21, S22):
    print(S11, S21)
    S11invS21T = np.linalg.solve(S11, S21.T)
    #S11invS21T, _, _, _ = np.linalg.lstsq(S11, S21.T, rcond=None)
    M = S22 - S21 @ S11invS21T

    return M, S11invS21T


def _compute_parabola(M, beta=1e5):
    # FIXME Minimal set of points is 4 for arbitrarily oriented parabolas
    evals, evecs = np.linalg.eigh(M)
    print(evals)

    evecs = evecs[..., ::-1]
    evals = evals[::-1]

    e11, e12, e13, e21, e22, e23, e31, e32, e33 = evecs.ravel(order='F')

    gamma1 = e22**2 - 4 * e21 * e23  # s^2
    gamma2 = -4 * e13 * e21 + 2 * e12 * e22 - 4 * e11 * e23  # st
    gamma3 = e12**2 - 4 * e11 * e13  # t^2
    gamma4 = -4 * e23 * e31 + 2 * e22 * e32 - 4 * e21 * e33  # s
    gamma5 = -4 * e13 * e31 + 2 * e12 * e32 - 4 * e11 * e33  # t
    gamma6 = e32**2 - 4 * e31 * e33  # constant term

    alpha1, alpha2 = evals[:2] ** 2 + beta
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
    K2 = 4 * (
        (2 * gamma2 * k2 + 4 * k8) * alpha3
        + gamma1 * k4 * alpha1**2
        + gamma3 * k1 * alpha2**2
    )
    K3 = 2 * k7 * k8
    K4 = k5 * k8

    r = np.roots([K4, K3, K2, K1, K0])
    print(r)

    tol = 1e-8  # relative to root magnitude; tune as needed
    is_real = np.abs(r.imag) <= tol * np.maximum(np.abs(r.real), 1.0)
    real_roots = r[is_real].real

    if real_roots.size == 0:
        raise ValueError('no real Lagrange multiplier found; degenerate fit')

    mu = real_roots[np.argmin(np.abs(real_roots))]

    r = mu

    u = k5 * r**2 + k7 * r + 4 * alpha3
    print(u)
    r_by_u = r / u#np.divide(r, u, out=u, where=u != 0)

    s = 2 * r_by_u * (k3 * r + alpha1 * gamma4)
    t = r_by_u * (k6 * r + 2 * alpha2 * gamma5)

    ust = np.vstack((np.ones_like(s), s, t))

    theta = evecs[..., ::-1] @ ust
    a, b, c = theta
    error = b**2 - 4 * a * c
    # FIXME Return error for scoring
    print(error)

    k = np.argmin(np.abs(error))

    theta_k = theta[..., k]
    return theta_k#np.real_if_close(theta_k)


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
    T = np.block([[TT, -scale_inv * mean[..., np.newaxis]], [0, 0, 1]])

    return K.transform(T, invert=False)


def _diff2_mu_sqr_error(mu, Q0, Q1, Q2, R0, R1, R2):
    return (
        2 * (2 * Q2 * mu + Q1) / (R2 * mu**2 + R1 * mu + R0) ** 2
        - 4
        * (Q2 * mu**2 + Q1 * mu + Q0)
        * (2 * R2 * mu + R1)
        / (R2 * mu**2 + R1 * mu + R0) ** 3
    )


def _correct_bias(pts, S11, S21, S22, z_e, z_h, x2m, xym, y2m, k_e, k_h):
    x, y = pts.T

    zeros = np.zeros_like(x)
    ones = np.ones_like(x)

    Dx = np.column_stack((2 * x, y, zeros, ones, zeros))
    Dy = np.column_stack((zeros, x, 2 * y, zeros, ones))

    Sxy = Dx.T @ Dx + Dy.T @ Dy
    S = np.block([[S22, S21], [S21.T, S11]])

    sigma1 = z_e.T @ S @ z_e
    sigma2 = z_e.T @ S @ z_h
    sigma3 = z_h.T @ S @ z_h
    sigma4 = z_e.T @ Sxy @ z_e
    sigma5 = z_e.T @ Sxy @ z_h
    sigma6 = z_h.T @ Sxy @ z_h

    Q2 = (
        sigma2 * sigma4
        - sigma2 * sigma6
        - sigma3 * sigma4
        + sigma3 * sigma5
        - sigma1 * sigma5
        + sigma1 * sigma6
    )
    Q1 = sigma3 * sigma4 + 2 * sigma1 * sigma5 - sigma1 * sigma6 - 2 * sigma2 * sigma4
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

    evals, evecs = scipy.linalg.eig(M, C)
    # The generalized eigenproblem is guaranteed to have real eigenpairs;
    # scipy.linalg.eig still returns a complex dtype whenever the pencil is
    # not symmetric-definite, which otherwise poisons all downstream
    # computations (e.g. np.arctan2 rejects complex input even when the
    # imaginary part is 0).
    evals = evals.real
    evecs = evecs.real
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
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    scale = np.sqrt(np.mean(centered**2))
    scale_inv = 1 / scale
    normalized = centered * scale_inv

    S11, S21, S22, x2m, xym, y2m = _build_scatter_matrices(
        normalized.astype(scale.dtype)
    )

    M, S11invS21T = _build_reduced_scatter_matrix(S11, S21, S22)

    if type == 'ellipse' or type == 'hyperbola':
        z2_e, z2_h, k_e, k_h = _compute_ellipse(M)

        if type == 'ellipse':
            Btop = _partial_backsubstitute(S11invS21T)
            z_e, z_h = (Btop @ np.column_stack((z2_e, z2_h))).T
            z = _correct_bias(
                normalized, S11, S21, S22, z_e, z_h, x2m, xym, y2m, k_e, k_h
            )
        else:  # hyperbola
            z = _backsubstitute(S11invS21T, z2_h, x2m, xym, y2m)
    elif type == 'parabola':
        z2 = _compute_parabola(M)
        z = _backsubstitute(S11invS21T, z2, x2m, xym, y2m)
    else:
        raise ValueError('unsupported conic type {}'.format(type))

    C = _denormalize(z, scale_inv, mean)

    return C

def _solve_homogeneous_quadratic(A, B, C):
    """Solve A*s^2 + B*s*t + C*t^2 = 0 for (s, t) up to scale.

    Returns a list of up to two (s, t) tuples. Complex results indicate no
    real solution exists (e.g. no real parabola passes through the 4 points).
    """
    if A != 0:
        disc = B**2 - 4 * A * C
        sqrt_disc = np.sqrt(complex(disc))
        r1, r2 = (-B + np.array([sqrt_disc, -sqrt_disc])) / (2 * A)
        return [(r1, 1.0), (r2, 1.0)]
    elif B != 0:
        # A == 0: t*(B*s + C*t) = 0 -> t=0 (pure e_a), or s = -(C/B)*t
        return [(1.0, 0.0), (-C / B, 1.0)]
    else:
        # A == B == 0: only t == 0 works unless C is also 0 (fully degenerate)
        return [(1.0, 0.0)]


def _compute_parabola_minimal(M):
    """Closed-form parabola(s) through an exact 4-point fit.

    M has (generically) a 2-D near-null space for 4 points; every conic in
    that pencil fits all 4 points exactly, so this finds where the pencil
    intersects the parabola constraint b^2 - 4ac = 0, rather than minimizing
    an error that is identically zero everywhere (which is why routing this
    case through the general quartic-based solver fails).
    """
    evals, evecs = np.linalg.eigh(M)  # ascending eigenvalues
    e_a = evecs[:, 0]  # smallest eigenvalue
    e_b = evecs[:, 1]  # second-smallest eigenvalue

    ea1, ea2, ea3 = e_a
    eb1, eb2, eb3 = e_b

    A = ea2**2 - 4 * ea1 * ea3
    B = 2 * ea2 * eb2 - 4 * (ea1 * eb3 + eb1 * ea3)
    C = eb2**2 - 4 * eb1 * eb3

    solutions = _solve_homogeneous_quadratic(A, B, C)

    z2_candidates = []
    for s, t in solutions:
        if np.iscomplex(s) or np.iscomplex(t):
            continue  # no real parabola for this branch
        z2_candidates.append(np.real(s) * e_a + np.real(t) * e_b)

    return z2_candidates


def fit_harker_minimal4(pts, type='parabola'):
    """Minimal-sample closed-form fit for RANSAC hypothesis generation.

    Returns a list of Conic candidates (up to 2 for a parabola from 4
    points), since a 4-point sample generally admits two solutions and
    disambiguation is left to MSAC/RANSAC scoring against the full point set.
    """
    if type != 'parabola':
        raise NotImplementedError('minimal solver currently only supports parabola')

    if len(pts) != 4:
        raise ValueError('minimal parabola fit requires exactly 4 points')

    mean = np.mean(pts, axis=0)
    centered = pts - mean
    scale = np.sqrt(np.mean(centered**2))
    scale_inv = 1 / scale
    normalized = centered * scale_inv

    S11, S21, S22, x2m, xym, y2m = _build_scatter_matrices(
        normalized.astype(scale.dtype)
    )
    M, S11invS21T = _build_reduced_scatter_matrix(S11, S21, S22)

    z2_candidates = _compute_parabola_minimal(M)

    conics = []
    for z2 in z2_candidates:
        z = _backsubstitute(S11invS21T, z2, x2m, xym, y2m)
        print('z', z)
        C = _denormalize(z, scale_inv, mean)
        print(C)
        conics.append(C)

    return conics


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = [-7, -3, 0, 0, 1, 1]
    y = [9, 5, 4, 8, 3, 5]

    t = np.linspace(-np.pi, np.pi)
    x = np.cos(t) + np.random.normal(scale=0.05, size=t.size)
    y = np.sin(t) + np.random.normal(scale=0.05, size=t.size)

    y = np.linspace(-5, 4, num=8)
    x = -y**2 + 2*y - 5   # exact, no noise term
    #pts = np.column_stack((x, y))

    y = np.linspace(-5, 4, num=4)
    x = -y**2 - 2 * y - 5 + np.random.normal(scale=0.5, size=y.size)

    # x = [-6.6, -2.8, -0.2, 0.4, 1.2, 1.4]
    # y = [8.8, 5.4, 3.6, 7.8, 3.4, 4.8]

    x = np.linspace(0, 7, num=5)
    y = 1e-2 * (x - 3.5)**2 + 1 + np.random.normal(scale=5e-4, size=x.size) # gentle, monotonic-curvature "almost linear" parabola
    pts = np.column_stack((x, y)) @ rot2d(-np.pi/3).T
    #pts = np.column_stack((x, y))

    #C1, C2 = fit_harker_minimal4(pts, type='parabola')
    C1 = fit_harker(pts, type='parabola')
    print(C1)
    C2 = C1

    P = Parabola(*C1.to_parabola())
    R = rot2d(P.alpha)
    #y = np.linspace(-1, 1)
    #x = y**2/(2*P.p)
    p = P.p
    vertex = P.vertex

    #pts1 = np.column_stack((x, y)) @ R.T + P.vertex

    pts_local_data = (pts - vertex) @ R          # transform real data into local frame
    y_min, y_max = pts_local_data[:, 1].min(), pts_local_data[:, 1].max()
    margin = 0.05 * (y_max - y_min)
    y_local = np.linspace(y_min - margin, y_max + margin, 500)
    x_local = y_local**2 / (2 * p)

    pts1 = np.column_stack((x_local, y_local)) @ R.T + vertex

    X, Y = np.meshgrid(
        np.linspace(np.min(pts[:, 0]) - 1, np.max(pts[:, 0]) + 1),
        np.linspace(-1 + np.min(pts[:, 1]), np.max(pts[:, 1]) + 1),
    )
    Z1 = C1(np.column_stack((X.ravel(), Y.ravel())))
    Z2 = C2(np.column_stack((X.ravel(), Y.ravel())))

    plt.figure()
    #plt.contour(X, Y, Z1.reshape(X.shape), levels=0)
    #plt.contour(X, Y, Z2.reshape(X.shape), levels=0)
    plt.plot(*pts1.T)
    plt.scatter(*pts.T)
    plt.show()
