"""Figure-6-style stacked bar of per-player defensive credit (task 6.1).

Colors follow the validated data-viz palette: the three positive contributions are
categorical identities (blue / aqua / yellow), and the penalty is the reserved
'critical' status color. Net is a diamond marker in ink. Marks carry a 2px surface
gap; a legend is always present (relief for the sub-3:1 fills).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from defcon.config import Config, load_config

# palette (light surface)
_COL = {
    "intercept": "#2a78d6",  # categorical slot 1 (blue)
    "disturb": "#1baf7a",    # slot 2 (aqua)
    "deter": "#eda100",      # slot 3 (yellow)
    "concede": "#d03b3b",    # status: critical (penalty)
}
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"

__all__ = ["plot_credit_breakdown"]


def plot_credit_breakdown(table, out_path, cfg: Config | None = None, top_n: int = 14,
                          suffix: str = "", title: str | None = None):
    """Stacked bar of Intercept/Disturb/Deter (up) and Concede (down), with Net.

    ``suffix`` selects the columns: "" for totals, "_p90" for per-90 values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    cfg = cfg or load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = table.sort_values(f"net{suffix}", ascending=False).head(top_n).reset_index(drop=True)
    labels = [str(p).replace("Player", "P") for p in df["player"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(min(1.0 + 0.72 * len(df), 12), 5.4))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    bar_kw = dict(width=0.72, edgecolor=_SURFACE, linewidth=1.6, zorder=3)
    # positive stack
    bottom = np.zeros(len(df))
    for cat in ("intercept", "disturb", "deter"):
        vals = df[f"{cat}{suffix}"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=_COL[cat], label=cat.capitalize(), **bar_kw)
        bottom += vals
    # concede (negative, drawn downward from 0)
    ax.bar(x, df[f"concede{suffix}"].to_numpy(), color=_COL["concede"], label="Concede", **bar_kw)
    # net marker
    ax.scatter(x, df[f"net{suffix}"].to_numpy(), marker="D", s=42, color=_INK,
               zorder=5, label="Net")

    ax.axhline(0, color="#c3c2b7", linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=_INK)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.set_major_locator(MultipleLocator(_nice_step(df, suffix)))
    ax.grid(axis="y", color=_GRID, linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    ylab = "credit per 90 min" if suffix == "_p90" else "defensive credit"
    ax.set_ylabel(ylab, color=_MUTED, fontsize=10)
    ax.set_title(title or "Defensive credit by player (Intercept / Disturb / Deter / Concede)",
                 color=_INK, fontsize=13, pad=12, loc="left")
    ax.legend(ncol=5, frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 1.13), labelcolor=_INK)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=_SURFACE)
    plt.close()
    return out_path


def _nice_step(df, suffix):
    span = df[f"net{suffix}"].abs().max() or 1.0
    for step in (0.5, 1, 2, 2.5, 5, 10, 20, 50):
        if span / step <= 6:
            return step
    return 100
