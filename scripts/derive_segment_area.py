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
"""Symbolic derivation and cross-check of Ellipse.segment_area().

This script is not part of the test suite (it depends on sympy, which is not
a project dependency) and is not imported by the library. It exists so the
closed-form formula used by Ellipse.segment_area() can be re-derived from
first principles and checked independently of the implementation. Run it
directly:

    python scripts/derive_segment_area.py
"""

from conics import Ellipse
import numpy as np
import sympy as sp


def derive_unit_circle_cap_area() -> sp.Expr:
    """Derives the area of the unit-circle cap cut off by the line x = h.

    By symmetry any chord at perpendicular distance h from the center of the
    unit circle can be rotated into the vertical line x = h without changing
    the cap area, so it suffices to integrate the circle's cross-sectional
    width over x in [h, 1].
    """
    x, h = sp.symbols('x h', real=True)

    width = 2 * sp.sqrt(1 - x**2)
    cap_area = sp.integrate(width, (x, h, 1))
    cap_area = sp.simplify(cap_area)

    closed_form = sp.acos(h) - h * sp.sqrt(1 - h**2)

    # simplify() alone does not apply the acos(h) + asin(h) == pi/2 identity
    # that relates sympy's asin-based antiderivative to the acos-based
    # closed form used by the implementation, so rewrite in terms of asin
    # first.
    diff = sp.simplify((cap_area - closed_form).rewrite(sp.asin))
    assert diff == 0, f'integral does not match the closed form, diff={diff}'

    return closed_form


def verify_atan2_rewrite() -> None:
    """Confirms acos(h) == atan2(sqrt(1 - h**2), h) on h in (-1, 1).

    Ellipse.segment_area() evaluates arccos via this identity because
    sqrt(1 - h**2) is already needed for the second term of the cap-area
    formula, and atan2 does not require its argument to be clipped to
    [-1, 1] the way acos does.

    Proof: let t = acos(h), so t is in [0, pi] and h = cos(t). Since sin is
    nonnegative on [0, pi], sqrt(1 - h**2) = sin(t). atan2(sin(t), cos(t))
    equals t for any t in (-pi, pi], and [0, pi] lies within that range, so
    atan2(sqrt(1 - h**2), h) = atan2(sin(t), cos(t)) = t = acos(h).

    sympy cannot simplify atan2 symbolically here, so the identity is
    checked numerically instead, which is sufficient since both sides are
    real-analytic on (-1, 1) and the samples span the domain densely.
    """
    for h in np.linspace(-1, 1, 2001, endpoint=False)[1:]:
        lhs = np.arccos(h)
        rhs = np.arctan2(np.sqrt(1 - h**2), h)
        assert np.isclose(lhs, rhs, atol=1e-12), (h, lhs, rhs)


def derive_ellipse_cap_area() -> None:
    """Derives the ellipse cap-area formula via the unit-circle substitution.

    Substituting x = a*u, y = b*v maps the ellipse x**2/a**2 + y**2/b**2 <= 1
    onto the unit disk u**2 + v**2 <= 1 with Jacobian determinant a*b, and
    maps the line A*x + B*y + C = 0 onto (A*a)*u + (B*b)*v + C = 0. So the
    ellipse cap area is a*b times the unit-circle cap area at perpendicular
    distance h = |C| / sqrt((A*a)**2 + (B*b)**2), matching
    Ellipse.segment_area()'s norm and h.
    """
    a, b, A, B, C = sp.symbols('a b A B C', positive=True)
    u, v = sp.symbols('u v', real=True)

    x, y = a * u, b * v
    line_in_uv = sp.expand(A * x + B * y + C)

    Au, Bu = sp.Poly(line_in_uv, u, v).coeff_monomial(u), sp.Poly(
        line_in_uv, u, v
    ).coeff_monomial(v)

    assert Au == A * a
    assert Bu == B * b


def cross_check_against_implementation(trials: int = 500, seed: int = 0) -> None:
    """Compares Ellipse.segment_area() to the derived closed form directly.

    This recomputes the closed form from A, B, C independently of
    conics/_ellipse.py (no shared code) for random ellipses and lines.
    """
    rng = np.random.default_rng(seed)
    h_sym = sp.symbols('h', real=True)
    closed_form = derive_unit_circle_cap_area()
    cap_area_fn = sp.lambdify(h_sym, closed_form, 'numpy')

    worst = 0.0

    for _ in range(trials):
        center = rng.uniform(-3, 3, 2)
        major, minor = rng.uniform(0.3, 3, 2)
        alpha = rng.uniform(-np.pi, np.pi)
        ellipse = Ellipse(center, [major, minor], alpha)

        line = np.array([*rng.uniform(-1, 1, 2), rng.uniform(-5, 5)])
        if np.hypot(*line[:2]) < 1e-9:
            continue

        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        Aq, Bq = (line[:2] @ R) * [major, minor]
        Cq = line[:2] @ ellipse.center + line[2]
        norm = np.hypot(Aq, Bq)
        h = abs(Cq) / norm

        total_area = np.pi * major * minor

        if h >= 1:
            expected = total_area if Cq < 0 else 0.0
        else:
            cap_area = major * minor * cap_area_fn(h)
            expected = total_area - cap_area if Cq < 0 else cap_area

        got = ellipse.segment_area(line)
        worst = max(worst, abs(got - expected))

    assert worst < 1e-9, f'worst absolute difference was {worst}'
    print(
        f'cross-check over {trials} random ellipses/lines: '
        f'worst abs difference = {worst:.3e}'
    )


def main():
    closed_form = derive_unit_circle_cap_area()
    print('Unit-circle cap area at distance h:', closed_form)

    verify_atan2_rewrite()
    print('acos(h) == atan2(sqrt(1 - h**2), h) on (-1, 1): confirmed')

    derive_ellipse_cap_area()
    print(
        'Ellipse cap area = major * minor * cap_area(h), '
        'h = |C| / hypot(A * major, B * minor): confirmed'
    )

    cross_check_against_implementation()


if __name__ == '__main__':
    main()
