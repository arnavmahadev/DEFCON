"""Matplotlib pitch rendering and frame plotting (supports task 1.1 acceptance).

Draws a soccer pitch in the DEFCON coordinate system (meters, origin at center,
attacking toward +x) and overlays a single tracking frame so you can eyeball
that a frame reads as a plausible 11v11 shape with the ball near a player.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from defcon.config import Config, load_config


def draw_pitch(ax=None, cfg: Config | None = None):
    """Draw pitch lines on a matplotlib Axes (created if not supplied)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    cfg = cfg or load_config()
    L, W, G = cfg.pitch.length, cfg.pitch.width, cfg.pitch.goal_width
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))

    line = dict(color="white", linewidth=1.5, zorder=1)
    hx, hy = L / 2, W / 2

    # Green playing surface as an explicit patch (robust to axis being hidden).
    ax.add_patch(Rectangle((-hx - 3, -hy - 3), L + 6, W + 6, color="#2e7d32", zorder=0))

    # Outer boundary + halfway line + center circle/spot.
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], **line)
    ax.plot([0, 0], [-hy, hy], **line)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor="white", lw=1.5, zorder=1))
    ax.add_patch(Circle((0, 0), 0.3, color="white", zorder=1))

    # Penalty + 6-yard boxes and goals at each end.
    for sign in (-1, 1):
        gx = sign * hx
        # penalty box (16.5 m deep, 40.32 m wide)
        ax.plot(
            [gx, gx - sign * 16.5, gx - sign * 16.5, gx],
            [-20.16, -20.16, 20.16, 20.16], **line,
        )
        # 6-yard box (5.5 m deep, 18.32 m wide)
        ax.plot(
            [gx, gx - sign * 5.5, gx - sign * 5.5, gx],
            [-9.16, -9.16, 9.16, 9.16], **line,
        )
        # goal
        ax.plot([gx, gx], [-G / 2, G / 2], color="white", lw=3, zorder=1)
        # penalty spot
        ax.add_patch(Circle((gx - sign * 11.0, 0), 0.25, color="white", zorder=1))

    ax.set_xlim(-hx - 3, hx + 3)
    ax.set_ylim(-hy - 3, hy + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_frame(
    tracking: pd.DataFrame,
    frame: int,
    cfg: Config | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot one tracking frame: home (blue), away (red), ball (white)."""
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    ax = draw_pitch(ax=ax, cfg=cfg)
    fr = tracking[tracking["frame"] == frame]
    colors = {"home": "#1e88e5", "away": "#e53935"}
    for team, color in colors.items():
        pts = fr[fr["team"] == team]
        ax.scatter(pts["x"], pts["y"], c=color, s=180, edgecolors="white", zorder=3, label=team)
        for _, row in pts.iterrows():
            jersey = str(row["player_id"]).replace("Player", "")
            ax.text(row["x"], row["y"], jersey, color="white", ha="center", va="center",
                    fontsize=7, zorder=4)
    ball = fr[fr["team"] == "ball"]
    ax.scatter(ball["x"], ball["y"], c="white", s=70, edgecolors="black", zorder=5, label="ball")

    if title is None:
        period = int(fr["period"].iloc[0]) if len(fr) else "?"
        title = f"Frame {frame} (period {period})"
    ax.set_title(title, color="black")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


def plot_responsibility(state, weights, out_path, cfg=None, title=None):
    """Figure-7-style plot: defenders shaded by responsibility weight for a pass.

    ``state`` is a GraphState (oriented +x); ``weights`` is a length-``n_players``
    array with the responsibility mass on each *defending* player (0 elsewhere).
    Draws attackers (blue), the carrier, the intended-receiver option (ring), the
    pass line, and defenders colored/sized by responsibility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax = draw_pitch(cfg=cfg)

    w = np.asarray(weights, dtype=float)
    att = state.is_attacking == 1
    dfd = state.is_attacking == 0

    # Attackers.
    ax.scatter(state.px[att], state.py[att], c="#1e88e5", s=150, edgecolors="white", zorder=3)
    # Carrier highlighted.
    ax.scatter([state.px[state.carrier_idx]], [state.py[state.carrier_idx]],
               c="#0d47a1", s=230, edgecolors="white", linewidths=2, zorder=4, marker="*")

    # Defenders colored by responsibility.
    dfd_idx = np.flatnonzero(dfd)
    wmax = max(w[dfd_idx].max(), 1e-6)
    colors = matplotlib.colormaps["Reds"](0.25 + 0.75 * w[dfd_idx] / wmax)
    sizes = 120 + 800 * (w[dfd_idx] / wmax)
    ax.scatter(state.px[dfd_idx], state.py[dfd_idx], c=colors, s=sizes,
               edgecolors="#7f0000", linewidths=1.2, zorder=3)
    for i in dfd_idx:
        if w[i] > 0.03:
            ax.text(state.px[i], state.py[i] - 2.2, f"{w[i]:.2f}", color="#7f0000",
                    ha="center", va="top", fontsize=8, fontweight="bold", zorder=5)

    ax.set_title(title or "Defender responsibility (P recover | pass fails)", color="black")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_epv_surface(xs, ys, epv_grid, out_path, cfg=None, state=None, title=None):
    """Heatmap of state EPV as the ball-carrier is swept over the pitch.

    ``xs``/``ys`` are grid coordinates (meters, centered); ``epv_grid`` is
    (len(ys), len(xs)). If ``state`` is given, overlay the frozen players.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax=ax, cfg=cfg)
    mesh = ax.pcolormesh(xs, ys, epv_grid, cmap="magma", alpha=0.82, shading="auto", zorder=1.5)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("state EPV", color="black")

    if state is not None:
        att = state.is_attacking == 1
        dfd = state.is_attacking == 0
        ax.scatter(state.px[att], state.py[att], c="#1e88e5", s=70, edgecolors="white", zorder=3)
        ax.scatter(state.px[dfd], state.py[dfd], c="#e53935", s=70, edgecolors="white", zorder=3)

    ax.set_title(title or "Expected Possession Value surface (ball position)", color="black")
    ax.annotate("attacking →", xy=(cfg.pitch.length / 2 - 12, -cfg.pitch.width / 2 - 1.5),
                color="black", fontsize=10)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def save_frame_png(tracking: pd.DataFrame, frame: int, out_path: str | Path, cfg: Config | None = None):
    """Render a frame to a PNG file and return the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame(tracking, frame, cfg=cfg)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path
