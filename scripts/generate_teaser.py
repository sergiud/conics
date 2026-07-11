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
"""Generates the README teaser image in a light and a dark variant.

The four panels are a self-contained tour of the library aimed at readers who
are not familiar with conics: fitting and non-linearly refining an ellipse,
intersecting two conics, fitting and refining a parabola, and measuring the
area a line cuts off an ellipse. Both variants are saved as SVGs with a
transparent background under ``docs/static/`` so a ``<picture>`` element in
the README can pick the one matching the reader's OS theme. Run it directly
from the repository root:

    python scripts/generate_teaser.py
"""

from __future__ import annotations

from conics import Conic
from conics import Ellipse
from conics import Parabola
from conics.fitting import fit_nievergelt
from conics.geometry import hnormalized
from conics.geometry import rot2d
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Colors are taken from the categorical palette slots "blue", "aqua", "red",
# and the chrome/ink roles, each with a light- and dark-surface step so the
# two renderings stay readable on their respective background.
_THEMES = {
    'light': {
        'ink': '#0b0b0b',
        'ink_secondary': '#52514e',
        'muted': '#898781',
        'blue': '#2a78d6',
        'aqua': '#1baf7a',
        'red': '#e34948',
        'orange': '#eb6834',
    },
    'dark': {
        'ink': '#ffffff',
        'ink_secondary': '#c3c2b7',
        'muted': '#898781',
        'blue': '#3987e5',
        'aqua': '#199e70',
        'red': '#e66767',
        'orange': '#d95926',
    },
}


def _style_axes(ax, colors):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(ax.get_title(), color=colors['ink'], fontsize=12, pad=10)
    ax.set_aspect('equal', adjustable='datalim')


def _panel_ellipse(ax, colors, rng):
    true_ellipse = Ellipse([0.3, -0.2], [2.4, 1.3], 0.5)
    # Points cover most, but not all, of the ellipse's circumference: a
    # partially observed contour is where an algebraic fit and its
    # non-linear refinement noticeably part ways.
    theta = rng.uniform(-0.4, 2.5 * np.pi / 1.15, 11)
    circle_pts = np.column_stack((np.cos(theta), np.sin(theta)))
    pts = (
        circle_pts * true_ellipse.major_minor
    ) @ rot2d(true_ellipse.alpha).T + true_ellipse.center
    pts += rng.normal(scale=0.13, size=pts.shape)

    C = fit_nievergelt(pts, type='ellipse', scale=True)
    fitted = Ellipse.from_conic(C)
    refined = fitted.refine(pts)

    ax.scatter(
        *pts.T, s=28, color=colors['ink_secondary'], zorder=3, label='noisy points'
    )
    ax.add_patch(
        mpatches.Ellipse(
            fitted.center,
            *(2 * np.asarray(fitted.major_minor)),
            angle=np.rad2deg(fitted.alpha),
            edgecolor=colors['blue'],
            facecolor='none',
            lw=2,
            ls='--',
            label='algebraic fit',
        )
    )
    ax.add_patch(
        mpatches.Ellipse(
            refined.center,
            *(2 * np.asarray(refined.major_minor)),
            angle=np.rad2deg(refined.alpha),
            edgecolor=colors['red'],
            facecolor='none',
            lw=2.2,
            label='non-linear refinement',
        )
    )
    ax.set_title('Fit an ellipse to noisy points')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
        labelcolor=colors['ink_secondary'],
        ncols=1,
    )


def _as_mpl_ellipse(c: Conic, **kwargs):
    x0, major_minor, angle = c.to_ellipse()
    return mpatches.Ellipse(
        x0.ravel(), *(major_minor.ravel() * 2), angle=np.rad2deg(angle), **kwargs
    )


def _panel_intersection(ax, colors):
    c1 = Conic.from_circle([0, 0], 1)
    c2 = Conic.from_circle([0.9, 0.4], 1.1)

    inter = hnormalized(c1.intersect(c2))

    ax.add_patch(
        _as_mpl_ellipse(c1, facecolor='none', edgecolor=colors['blue'], lw=2.2)
    )
    ax.add_patch(
        _as_mpl_ellipse(c2, facecolor='none', edgecolor=colors['orange'], lw=2.2)
    )
    ax.scatter(
        *inter.T,
        s=60,
        color=colors['ink'],
        edgecolor=colors['ink'],
        zorder=3,
        label='intersections',
    )
    ax.set_xlim(-1.6, 2.5)
    ax.set_ylim(-1.6, 2.0)
    ax.set_title('Intersect two conics')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
        labelcolor=colors['ink_secondary'],
    )


def _panel_parabola(ax, colors):
    x = [-1, 2, 5, 10, -4]
    y = [1, -2, 3, -4, -3]
    pts = np.column_stack((x, y))

    C = fit_nievergelt(pts, type='parabola', scale=False)
    p = Parabola.from_conic(C)
    p_refined = p.refine(pts)
    C_refined = p_refined.to_conic()

    X, Y = np.meshgrid(
        np.linspace(np.min(x) - 3, np.max(x) + 1, 300),
        np.linspace(np.min(y) - 1, np.max(y) + 1, 300),
    )
    Z = C(np.dstack([X, Y]))
    Z_refined = C_refined(np.dstack([X, Y]))

    contact_pts = p_refined.contact(pts)

    ax.contour(X, Y, Z, colors=[colors['blue']], levels=[0], linewidths=2, linestyles='--')
    ax.contour(X, Y, Z_refined, colors=[colors['red']], levels=[0], linewidths=2.2)

    for xy in np.dstack((contact_pts, pts)):
        ax.plot(*xy, '--', c=colors['muted'], lw=1, zorder=1)

    ax.scatter(x, y, s=28, color=colors['ink_secondary'], zorder=3, label='observations')
    ax.scatter(
        *contact_pts.T,
        s=28,
        color=colors['aqua'],
        zorder=3,
        label='orthogonal contact points',
    )
    ax.set_title('Fit and refine a parabola')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
        labelcolor=colors['ink_secondary'],
    )


def _panel_segment_area(ax, colors):
    e = Ellipse([0, 0], [2.4, 1.4], 0.3)

    line = np.array([0.0, -1.0, 0.75])
    area = e.segment_area(line)
    pct = 100 * area / e.area

    theta = np.linspace(0, 2 * np.pi, 400)
    boundary = np.column_stack((np.cos(theta), np.sin(theta))) * e.major_minor
    boundary = boundary @ rot2d(e.alpha).T + e.center

    side = line[:2] @ boundary.T + line[-1]
    cap_pts = boundary[side < 0]

    # Crop the drawn line to the chord it forms inside the ellipse instead of
    # an arbitrary fixed span, which would protrude past the boundary by a
    # different amount on either side.
    crossings = hnormalized(e.to_conic().intersect_line(line))
    crossings = crossings[np.argsort(crossings[:, 0])]

    ax.add_patch(
        mpatches.Ellipse(
            e.center,
            *(2 * np.asarray(e.major_minor)),
            angle=np.rad2deg(e.alpha),
            edgecolor=colors['blue'],
            facecolor='none',
            lw=2.2,
        )
    )
    ax.fill(*cap_pts.T, color=colors['aqua'], alpha=0.45, zorder=2)
    ax.plot(*crossings.T, color=colors['orange'], lw=2, ls='--')

    ax.annotate(
        f'{pct:.0f}% of the\nellipse area',
        xy=(0, 1.05),
        xytext=(0, 1.75),
        ha='center',
        color=colors['ink'],
        fontsize=9,
        arrowprops=dict(arrowstyle='-', color=colors['muted'], lw=1),
    )
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.0, 2.4)
    ax.set_title('Measure the area a line cuts off')


def render(theme: str, out_path: Path) -> None:
    colors = _THEMES[theme]
    rng = np.random.default_rng(0)

    with plt.rc_context(
        {
            'font.family': 'sans-serif',
            'text.color': colors['ink'],
            'axes.labelcolor': colors['ink'],
            'axes.titlecolor': colors['ink'],
            'svg.fonttype': 'none',
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))

        _panel_ellipse(axes[0, 0], colors, rng)
        _panel_intersection(axes[0, 1], colors)
        _panel_parabola(axes[1, 0], colors)
        _panel_segment_area(axes[1, 1], colors)

        for ax in axes.ravel():
            _style_axes(ax, colors)

        fig.suptitle(
            'conics: construct, fit, intersect, and measure conic sections',
            color=colors['ink'],
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, transparent=True)
        plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / 'docs' / 'static'
    out_dir.mkdir(parents=True, exist_ok=True)

    for theme in ('light', 'dark'):
        render(theme, out_dir / f'teaser-{theme}.svg')


if __name__ == '__main__':
    main()
