"""Aggregation of defensive credits (Phase 6).

- 6.1 per-match: sum credits by player × category, normalize per 90 minutes.
- 6.2 season roll-up: combine matches, sum minutes, filter by minutes threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config
from defcon.credit.engine import aggregate_by_category
from defcon.credit.rules import CATEGORIES, Credit

__all__ = ["minutes_played", "aggregate_per90", "season_rollup", "credits_to_items"]


def minutes_played(tracking: pd.DataFrame, cfg: Config | None = None) -> dict[str, float]:
    """Minutes each player is on the pitch = frames present / frame_rate / 60."""
    cfg = cfg or load_config()
    players = tracking[tracking["team"].isin(["home", "away"])]
    frames = players.groupby("player_id").size()
    return (frames / cfg.tracking.frame_rate / 60.0).to_dict()


def credits_to_items(credits: pd.DataFrame) -> list[Credit]:
    """Convert a long credit DataFrame (from CreditEngine) into Credit items."""
    return [Credit(r.player, float(r.value), r.category) for r in credits.itertuples()]


def aggregate_per90(
    items: list[Credit],
    minutes: dict[str, float],
    player_team: dict | None = None,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    """Per-player category totals plus per-90 columns (task 6.1)."""
    table = aggregate_by_category(items, player_team)
    table["minutes"] = table["player"].map(minutes).fillna(0.0)
    table = table[table["minutes"] >= min_minutes].reset_index(drop=True)
    denom = table["minutes"].replace(0, np.nan)
    for c in (*CATEGORIES, "net"):
        table[f"{c}_p90"] = table[c] * 90.0 / denom
    return table.sort_values("net_p90", ascending=False).reset_index(drop=True)


def season_rollup(
    matches: list[tuple[list[Credit], dict[str, float]]],
    player_team: dict | None = None,
    min_minutes: float = 900.0,
) -> pd.DataFrame:
    """Combine per-match (credits, minutes) into a season table (task 6.2).

    Credits are summed across matches; minutes are summed; per-90 uses the total.
    Players below ``min_minutes`` are filtered (paper uses 900).
    """
    all_items: list[Credit] = []
    total_minutes: dict[str, float] = {}
    for items, minutes in matches:
        all_items.extend(items)
        for p, m in minutes.items():
            total_minutes[p] = total_minutes.get(p, 0.0) + m
    return aggregate_per90(all_items, total_minutes, player_team, min_minutes)
