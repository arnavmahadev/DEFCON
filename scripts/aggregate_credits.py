#!/usr/bin/env python
"""Aggregate defensive credits per player, per 90, and across matches (Phase 6).

Runs the credit engine on Games 1 & 2, normalizes per 90 minutes, reproduces a
Figure-6 stacked bar, and rolls up a two-match season table.

Usage: python scripts/aggregate_credits.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from defcon import load_config
from defcon.credit.aggregate import (
    aggregate_per90,
    credits_to_items,
    minutes_played,
    season_rollup,
)
from defcon.credit.engine import CreditEngine
from defcon.data.pipeline import process_metrica_match
from defcon.epv.epv import EPVEngine
from defcon.viz.credit import plot_credit_breakdown

warnings.filterwarnings("ignore")
pd.set_option("display.width", 130)


def main() -> None:
    cfg = load_config()
    gd = cfg.path("tracking_dir") / "metrica"
    epv = EPVEngine.from_checkpoints(cfg)
    engine = CreditEngine(epv, cfg)

    matches, player_team = [], {}
    per_match_tables = {}
    for game in (1, 2):
        actions, tracking = process_metrica_match(gd, game, cfg)
        credits = engine.process_match(actions, tracking)
        items = credits_to_items(credits)
        mins = minutes_played(tracking, cfg)
        pt = actions.dropna(subset=["player"]).groupby("player")["team"].agg(
            lambda s: s.mode().iloc[0]).to_dict()
        player_team.update({f"g{game}:{k}": v for k, v in pt.items()})
        # Namespace players per game so cross-game same jersey isn't merged wrongly.
        game_items = [type(i)(f"g{game}:{i.player}", i.value, i.category) for i in items]
        game_mins = {f"g{game}:{k}": v for k, v in mins.items()}
        matches.append((game_items, game_mins))

        t90 = aggregate_per90(items, mins, pt, min_minutes=20)
        per_match_tables[game] = t90
        print(f"[agg] game {game}: {len(t90)} players | "
              f"top net/90 = {t90.iloc[0]['player']} {t90.iloc[0]['net_p90']:.2f}")

    # ---- Figure-6 for game 1 ----
    fig = plot_credit_breakdown(per_match_tables[1], "docs/img/figure6_credit.png", cfg,
                                top_n=14, suffix="",
                                title="Defensive credit by player — Metrica Game 1")
    print(f"[agg] saved Figure-6 -> {fig}")

    # ---- season roll-up (2 matches) ----
    # 900-min threshold is unreachable with 2 games; use a demonstrative 60-min floor.
    season = season_rollup(matches, player_team, min_minutes=60)
    cols = ["player", "team", "minutes", "intercept", "disturb", "deter", "concede", "net", "net_p90"]
    show = season[cols].copy()
    for c in ["intercept", "disturb", "deter", "concede", "net", "net_p90"]:
        show[c] = show[c].round(2)
    show["minutes"] = show["minutes"].round(0)
    print("\n=== two-match season table (net credit per 90, ≥60 min) — top 12 ===")
    print(show.head(12).to_string(index=False))

    out = cfg.path("data_processed") / "season_credits.parquet"
    season.to_parquet(out, index=False)
    print(f"\n[agg] saved season table -> {out}")


if __name__ == "__main__":
    main()
