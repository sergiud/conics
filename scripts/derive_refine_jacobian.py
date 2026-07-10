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
"""Symbolic derivation and cross-check of Ellipse.refine()'s analytic Jacobian.

This script is not part of the test suite (it depends on sympy, which is not
a project dependency) and is not imported by the library. It exists so the
Jacobian used internally by Ellipse.refine() can be re-derived from first
principles, independently of conics/_ellipse.py, and checked against the
Jacobian the shipped code actually computes. Run it directly:

    python scripts/derive_refine_jacobian.py
"""

from conics import Ellipse
from conics.geometry import rot2d
from scipy.optimize import least_squares as _real_least_squares
from unittest.mock import patch
import conics._ellipse as _ellipse_mod
import numpy as np
import sympy as sp


def derive_residual_jacobian() -> (
    tuple[list[sp.Symbol], sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Matrix]
):
    """Derives d(residual)/d(xc, yc, a, b, alpha) via implicit differentiation.

    ``refine()`` fits (xc, yc, a, b, alpha) by minimizing, over all input
    points, the residual between each point and its contact point (the
    closest point on the ellipse). The contact point's local coordinates
    (x, y) are defined implicitly as the solution of

        f1(x, y, a, b) = a**2*y**2 + b**2*x**2 - a**2*b**2 = 0
        f2(x, y, a, b, xi, yi) = b**2*x*(yi - y) - a**2*y*(xi - x) = 0

    where (xi, yi) are the input point's local coordinates, which themselves
    depend on (xc, yc, alpha). Differentiating f1 = f2 = 0 with respect to
    each fit parameter p gives a linear system for d(x, y)/dp, since (x, y)
    cannot be solved for in closed form. The world-space residual is then

        residual(p) = (x, y) @ R(alpha).T + (xc, yc) - (px, py)

    whose derivative combines that implicit d(x, y)/dp with the residual's
    own direct dependence on p (through R and, for the center, the added
    (xc, yc) term).

    The center columns of the analytic Jacobian must include both
    contributions, and the rotation submatrix must use
    (cos(alpha), sin(alpha)), not (cos(alpha), -sin(alpha)).
    """
    xc, yc, a, b, alpha, px, py, x, y = sp.symbols(
        'xc yc a b alpha px py x y', real=True
    )

    c, s = sp.cos(alpha), sp.sin(alpha)

    # R matches Ellipse.contact()'s rot2d(alpha). pts1 = (pts - center) @ R
    # for row vectors is local = R.T * (pt - center) for column vectors, and
    # xy @ R.T + center is world = R * xy + center.
    R = sp.Matrix([[c, -s], [s, c]])
    center = sp.Matrix([xc, yc])
    pt = sp.Matrix([px, py])
    xy = sp.Matrix([x, y])

    xi, yi = R.T * (pt - center)

    f = sp.Matrix(
        [
            sp.Rational(1, 2) * (a**2 * y**2 + b**2 * x**2 - a**2 * b**2),
            b**2 * x * (yi - y) - a**2 * y * (xi - x),
        ]
    )

    params = [xc, yc, a, b, alpha]

    # Q = d(f)/d(x, y), used to solve for d(x, y)/dp by implicit
    # differentiation. This is exactly _jac_contact_point()'s return value.
    Q = f.jacobian([x, y])

    # One linear solve with all five parameter columns as simultaneous
    # right-hand sides, instead of solving Q separately for each parameter.
    dxy_dp = Q.solve(-f.jacobian(params))

    residual = R * xy + center - pt

    # Total derivative of residual(p): the explicit dependence on p (holding
    # x, y fixed) plus the implicit dependence carried through (x, y)(p).
    J = residual.jacobian(params) + residual.jacobian([x, y]) * dxy_dp

    return params, px, py, x, y, J


def _capture_analytic_jacobian(e: Ellipse, pts: np.ndarray) -> np.ndarray:
    """Captures the Jacobian refine() actually hands to least_squares.

    least_squares() is patched to evaluate the real `fun`/`jac` closures at
    the ellipse's current parameters and return immediately, instead of
    optimizing, so this exercises the exact shipped Jacobian rather than a
    reimplementation of it.

    refine() and contact() both call the (same, module-level) least_squares,
    contact() once per input point to solve for that point's local contact
    coordinates, refine() once overall for the 5 ellipse parameters. Only
    the latter call should be short-circuited: contact() has to actually
    run to produce the contact points refine()'s Jacobian depends on, so
    calls with a 2-parameter x0 (a single contact point) are passed through
    to the real least_squares, keyed on x0's length rather than on calling
    order since contact()'s calls happen while refine()'s call is being
    handled.
    """
    captured = {}

    def fake_least_squares(fun, x0, args=(), jac=None, **kwargs):
        if len(x0) != 5:
            return _real_least_squares(fun, x0, args=args, jac=jac, **kwargs)

        captured['J'] = jac(x0, *args)

        class Result:
            pass

        r = Result()
        r.x = x0
        return r

    with patch.object(_ellipse_mod, 'least_squares', fake_least_squares):
        e.refine(pts)

    return captured['J']


def cross_check_against_implementation(trials: int = 200, seed: int = 0) -> None:
    """Compares the sympy-derived Jacobian to the one refine() actually uses.

    A random ellipse and a fixed number of random points are drawn per
    trial to evaluate the analytic Jacobian captured directly from
    Ellipse.refine(). Since Ellipse itself models one ellipse at a time,
    that part stays a per-trial Python loop, but the independently derived
    symbolic Jacobian is evaluated for every (ellipse, point) pair in a
    single vectorized numpy call over the collected samples, rather than
    lambdified and invoked once per point.
    """
    rng = np.random.default_rng(seed)
    points_per_trial = 3

    params, px, py, x_sym, y_sym, J_sym = derive_residual_jacobian()
    xc, yc, a, b, alpha = params

    # Lambdify each of the 10 Jacobian entries separately so that, given
    # array arguments, numpy broadcasts every entry over all samples at
    # once instead of sympy looping internally over a Matrix of scalars.
    J_fn = sp.lambdify(
        (xc, yc, a, b, alpha, px, py, x_sym, y_sym), list(J_sym), 'numpy'
    )

    xc_vals, yc_vals, a_vals, b_vals, alpha_vals = [], [], [], [], []
    px_vals, py_vals, x_vals, y_vals = [], [], [], []
    J_impl_blocks = []

    for _ in range(trials):
        center = rng.uniform(-3, 3, 2)
        major, minor = rng.uniform(0.3, 3, 2)
        alpha_val = rng.uniform(-np.pi, np.pi)
        e = Ellipse(center, [major, minor], alpha_val)

        pts = e.center + rng.uniform(-4, 4, (points_per_trial, 2))

        J_impl_blocks.append(
            _capture_analytic_jacobian(e, pts).reshape(points_per_trial, 2, 5)
        )

        R = rot2d(alpha_val)
        xy1 = (e.contact(pts) - e.center) @ R

        xc_vals.append(np.full(points_per_trial, e.center[0]))
        yc_vals.append(np.full(points_per_trial, e.center[1]))
        a_vals.append(np.full(points_per_trial, major))
        b_vals.append(np.full(points_per_trial, minor))
        alpha_vals.append(np.full(points_per_trial, alpha_val))
        px_vals.append(pts[:, 0])
        py_vals.append(pts[:, 1])
        x_vals.append(xy1[:, 0])
        y_vals.append(xy1[:, 1])

    J_impl = np.concatenate(J_impl_blocks, axis=0)

    J_flat = J_fn(
        np.concatenate(xc_vals),
        np.concatenate(yc_vals),
        np.concatenate(a_vals),
        np.concatenate(b_vals),
        np.concatenate(alpha_vals),
        np.concatenate(px_vals),
        np.concatenate(py_vals),
        np.concatenate(x_vals),
        np.concatenate(y_vals),
    )
    J_expected = np.stack(J_flat, axis=-1).reshape(-1, 2, 5)

    # Contact points near the ellipse's evolute cusps make Q (see
    # _jac_contact_point) nearly singular, so both the implementation and
    # the independent derivation solve the same ill-conditioned system and
    # can legitimately land on Jacobian entries of very different magnitude
    # between samples. A relative tolerance accounts for that scale swing,
    # unlike an absolute-only one.
    rel = np.abs(J_impl - J_expected) / (1.0 + np.abs(J_expected))
    worst = np.max(rel)

    assert worst < 1e-6, f'worst relative difference was {worst}'
    print(
        f'cross-check over {trials * points_per_trial} random '
        f'ellipse/point samples: worst relative difference = {worst:.3e}'
    )


def main():
    derive_residual_jacobian()
    print(
        'Derived the residual Jacobian symbolically via implicit '
        'differentiation of f1 = f2 = 0.'
    )

    cross_check_against_implementation()


if __name__ == '__main__':
    main()
