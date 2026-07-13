"""Interactive per-player defensive-credit timeline (task 8.1 / paper Fig 10).

Cumulative Net credit per player over match time, with hover showing the action
that caused each step. Themed to match the project's cream/serif identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

# validated categorical palette (data-viz skill)
_PLAYER_HUES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834", "#008300"]
_CREAM, _PAPER, _INK, _MUTED, _GRID = "#f4f1ea", "#faf8f2", "#1b1a17", "#6f6a5c", "#e5e0d3"

__all__ = ["build_timeline"]


def _match_minute(actions: pd.DataFrame) -> pd.Series:
    """Continuous match clock in minutes.

    Some providers restart the clock each half; others (Metrica) keep it running.
    Only offset the 2nd half if its timestamps actually reset below the 1st-half max.
    """
    minute = actions["time_s"] / 60.0
    p1 = minute[actions["period"] == 1]
    p2 = minute[actions["period"] == 2]
    p1_end = p1.max() if len(p1) else 0.0
    resets = len(p2) > 0 and p2.min() < p1_end
    offset = np.where((actions["period"] == 2) & resets, p1_end, 0.0)
    return minute + offset


def build_timeline(credits: pd.DataFrame, actions: pd.DataFrame, cfg: Config | None = None,
                   top_n: int = 6, title: str | None = None):
    """Return a self-contained Plotly Figure of cumulative Net credit per player."""
    import plotly.graph_objects as go

    cfg = cfg or load_config()
    # per (player, action) net credit + category breakdown
    per = (credits.groupby(["player", "action_id"])
           .agg(net=("value", "sum")).reset_index())
    meta = actions[["action_id", "time_s", "period", "type", "team"]].copy()
    meta["minute"] = _match_minute(actions)
    per = per.merge(meta, on="action_id", how="left").dropna(subset=["minute"])

    totals = per.groupby("player")["net"].sum().sort_values(ascending=False)
    players = list(totals.head(top_n).index)

    fig = go.Figure()
    for i, pl in enumerate(players):
        sub = per[per.player == pl].sort_values("minute")
        cum = sub["net"].cumsum()
        hue = _PLAYER_HUES[i % len(_PLAYER_HUES)]
        fig.add_trace(go.Scatter(
            x=sub["minute"], y=cum, mode="lines+markers",
            name=str(pl).replace("Player", "P"),
            line=dict(color=hue, width=2), marker=dict(size=5, color=hue),
            customdata=np.stack([sub["action_id"], sub["type"], sub["net"]], axis=-1),
            hovertemplate=("<b>%{fullData.name}</b>  min %{x:.1f}<br>"
                           "action %{customdata[0]} · %{customdata[1]}<br>"
                           "Δ credit %{customdata[2]:+.3f} · cumulative %{y:.2f}<extra></extra>"),
        ))

    fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(
        title=dict(text=title or "Cumulative defensive credit over the match",
                   font=dict(family="Georgia, 'Palatino Linotype', serif", size=22, color=_INK), x=0.02),
        paper_bgcolor=_CREAM, plot_bgcolor=_PAPER,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=_INK, size=13),
        xaxis=dict(title="match minute", gridcolor=_GRID, zeroline=False,
                   linecolor="#d9d3c3", ticksuffix="′"),
        yaxis=dict(title="cumulative Net credit", gridcolor=_GRID, zeroline=False, linecolor="#d9d3c3"),
        legend=dict(title="player", bgcolor="rgba(0,0,0,0)", orientation="v"),
        hovermode="closest", margin=dict(l=70, r=30, t=60, b=60),
    )
    return fig
