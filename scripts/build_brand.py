"""
Generate the project's brand assets: logo, favicons and the social card.

    python scripts/build_brand.py

The mark is a compass whose needle points down into the left tail of a
distribution - the quantile the whole project is about.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

INK = "#0f172a"  # slate-900
RING = "#2563eb"  # blue-600
TAIL = "#db2777"  # pink-600

NEEDLE_ANGLE = 225  # degrees, pointing down-left toward the loss tail


def draw_compass(ax, ring_color: str, needle_light: str, lw: float = 2.6) -> None:
    """Draw the compass mark centred on (0, 0) with radius 1."""
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # outer ring
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=ring_color, lw=lw))

    # cardinal ticks
    for deg in (0, 90, 180, 270):
        a = math.radians(deg)
        ax.plot(
            [0.80 * math.cos(a), 0.95 * math.cos(a)],
            [0.80 * math.sin(a), 0.95 * math.sin(a)],
            color=ring_color,
            lw=lw * 0.55,
            solid_capstyle="round",
        )

    # needle: two triangles sharing a base through the centre
    a = math.radians(NEEDLE_ANGLE)
    u = np.array([math.cos(a), math.sin(a)])  # toward the tail
    v = np.array([-u[1], u[0]])  # perpendicular
    tip_lo = u * 0.74
    tip_hi = -u * 0.74
    base_a, base_b = v * 0.20, -v * 0.20

    # the half pointing into the tail carries the accent colour
    ax.add_patch(Polygon([tip_lo, base_a, base_b], closed=True, fc=TAIL, ec="none"))
    ax.add_patch(Polygon([tip_hi, base_a, base_b], closed=True, fc=needle_light, ec="none"))
    ax.add_patch(Circle((0, 0), 0.085, fc=ring_color, ec="none"))


def figure(size_px: int, bg: str | None) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(1, 1), dpi=size_px)
    if bg:
        fig.patch.set_facecolor(bg)
    else:
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.patch.set_alpha(0)
    return fig, ax


def save(fig: plt.Figure, name: str, dpi: int, transparent: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=dpi, transparent=transparent)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def build_icons() -> None:
    # site header sits on the indigo Material bar -> white mark, transparent bg
    fig, ax = figure(512, None)
    draw_compass(ax, ring_color="#ffffff", needle_light="#ffffff", lw=3.0)
    save(fig, "logo.png", 512, transparent=True)

    # favicon / touch icon: on a solid dark ground so it reads in any tab colour
    for size, name in [(64, "favicon.png"), (180, "apple-touch-icon.png")]:
        fig, ax = figure(size, INK)
        draw_compass(ax, ring_color="#ffffff", needle_light="#ffffff", lw=3.4)
        save(fig, name, size, transparent=False)

    # light-background version for the README
    fig, ax = figure(512, None)
    draw_compass(ax, ring_color=RING, needle_light="#94a3b8", lw=3.0)
    save(fig, "logo-dark.png", 512, transparent=True)


def build_social_card() -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(INK)

    ax_mark = fig.add_axes([0.07, 0.28, 0.26, 0.5])
    ax_mark.patch.set_alpha(0)
    draw_compass(ax_mark, ring_color="#ffffff", needle_light="#ffffff", lw=4.5)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.37, 0.63, "Quantile Compass", color="#ffffff", fontsize=44, fontweight="bold", va="center"
    )
    ax.text(0.37, 0.505, "Portfolio Value-at-Risk", color="#cbd5e1", fontsize=23, va="center")
    ax.text(
        0.37,
        0.405,
        "20 years  ·  4 crises  ·  EWMA volatility",
        color="#94a3b8",
        fontsize=15,
        family="monospace",
        va="center",
    )
    ax.annotate(
        "View the analysis  →",
        xy=(0.37, 0.265),
        color=RING,
        fontsize=17,
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.55", fc="none", ec=RING, lw=1.8),
    )
    save(fig, "og-card.png", 100, transparent=False)


def build_svg() -> None:
    """Hand-written SVG twin of the mark, used as the MkDocs header logo."""
    a = math.radians(NEEDLE_ANGLE)
    ux, uy = math.cos(a), -math.sin(a)  # SVG y axis points down
    vx, vy = -uy, ux
    cx = cy = 24.0
    r = 20.0

    def pt(sx: float, sy: float) -> str:
        return f"{cx + sx * r:.2f},{cy + sy * r:.2f}"

    tip_lo = pt(ux * 0.74, uy * 0.74)
    tip_hi = pt(-ux * 0.74, -uy * 0.74)
    base_a = pt(vx * 0.20, vy * 0.20)
    base_b = pt(-vx * 0.20, -vy * 0.20)

    ticks = []
    for deg in (0, 90, 180, 270):
        t = math.radians(deg)
        x1, y1 = cx + 0.80 * r * math.cos(t), cy - 0.80 * r * math.sin(t)
        x2, y2 = cx + 0.95 * r * math.cos(t), cy - 0.95 * r * math.sin(t)
        ticks.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none">
  <circle cx="{cx}" cy="{cy}" r="{r}" stroke="currentColor" stroke-width="2.6"/>
  {"".join(ticks)}
  <polygon points="{tip_lo} {base_a} {base_b}" fill="{TAIL}"/>
  <polygon points="{tip_hi} {base_a} {base_b}" fill="currentColor"/>
  <circle cx="{cx}" cy="{cy}" r="1.9" fill="currentColor"/>
</svg>
"""
    path = OUT / "logo.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    build_icons()
    build_svg()
    build_social_card()


if __name__ == "__main__":
    main()
