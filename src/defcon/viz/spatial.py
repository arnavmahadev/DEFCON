"""Spatial credit heatmaps (task 8.2) and pairwise attacker-defender matrix (8.3).

Both consume a prepared long credit frame so the plotting stays decoupled from the
data plumbing (see scripts/analysis_figures.py for how the frame is built).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from defcon.config import Config, load_config
from defcon.credit.rules import CATEGORIES
from defcon.viz.pitch import draw_pitch

_CAT_CMAP = {"intercept": "Blues", "disturb": "GnBu", "deter": "YlOrBr", "concede": "Reds"}
_INK, _MUTED = "#1b1a17", "#6f6a5c"

__all__ = ["plot_credit_zones", "plot_pairwise"]


def plot_credit_zones(df: pd.DataFrame, out_path, cfg: Config | None = None,
                      bins=(12, 8), title: str | None = None):
    """Four pitch panels (Intercept/Disturb/Deter/Concede) of credit binned over zones.

    ``df`` needs columns ``x``, ``y`` (meters, oriented so the attack goes +x) and
    ``value``, ``category``. Attack direction is left→right in every panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    L, W = cfg.pitch.length, cfg.pitch.width
    xedges = np.linspace(-L / 2, L / 2, bins[0] + 1)
    yedges = np.linspace(-W / 2, W / 2, bins[1] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#faf8f2")
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        draw_pitch(ax=ax, cfg=cfg)
        sub = df[df["category"] == cat]
        H, _, _ = np.histogram2d(sub["x"], sub["y"], bins=[xedges, yedges],
                                 weights=sub["value"].abs())
        # smooth-ish display via pcolormesh on bin centers
        Xc, Yc = np.meshgrid(xedges, yedges)
        mesh = ax.pcolormesh(Xc, Yc, H.T, cmap=_CAT_CMAP[cat], alpha=0.85, zorder=1.5,
                             shading="flat")
        fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(f"{cat.capitalize()}  (Σ|credit| = {sub['value'].abs().sum():.1f})",
                     color=_INK, fontsize=13, loc="left")
        ax.annotate("attack →", xy=(L / 2 - 16, -W / 2 - 1.5), color=_MUTED, fontsize=9)

    fig.suptitle(title or "Where defensive credit is created, binned over pitch zones",
                 color=_INK, fontsize=16, x=0.02, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#faf8f2")
    plt.close()
    return out_path


def plot_pairwise(df: pd.DataFrame, out_path, cfg: Config | None = None,
                  top_k: int = 10, title: str | None = None):
    """Defender × attacker matrix: net credit each defender earned vs each attacker.

    ``df`` needs columns ``defender``, ``attacker``, ``value``. Blue = the defender
    gained value against that attacker; red = conceded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    cfg = cfg or load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = df.pivot_table(index="defender", columns="attacker", values="value",
                         aggfunc="sum", fill_value=0.0)
    # keep the most-involved rows/cols
    defenders = mat.abs().sum(axis=1).sort_values(ascending=False).head(top_k).index
    attackers = mat.abs().sum(axis=0).sort_values(ascending=False).head(top_k).index
    mat = mat.loc[defenders, attackers]

    vmax = float(np.abs(mat.to_numpy()).max()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(attackers), 1.0 + 0.7 * len(defenders)))
    fig.patch.set_facecolor("#faf8f2")
    im = ax.imshow(mat.to_numpy(), cmap="RdBu", norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="net credit (blue = defender gained)")

    ax.set_xticks(range(len(attackers)))
    ax.set_xticklabels([str(a).replace("Player", "P") for a in attackers], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(defenders)))
    ax.set_yticklabels([str(d).replace("Player", "P") for d in defenders], fontsize=9)
    ax.set_xlabel("attacker (ball-carrier)", color=_MUTED)
    ax.set_ylabel("defender (credited)", color=_MUTED)
    ax.set_title(title or "Pairwise credit: defender vs. attacker", color=_INK, fontsize=13, loc="left")
    for i in range(len(defenders)):
        for j in range(len(attackers)):
            v = mat.iloc[i, j]
            if abs(v) > 0.15 * vmax:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 * vmax else _INK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#faf8f2")
    plt.close()
    return out_path
