#!/usr/bin/env python
"""Spatial credit heatmaps (8.2) and the pairwise attacker-defender matrix (8.3).

Joins per-action credits to action locations / acting players, orients every action
so the attack goes +x, and renders the two analysis figures.

Usage: python scripts/analysis_figures.py --game 1
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from defcon import load_config
from defcon.credit.engine import CreditEngine
from defcon.data.metrica import infer_playing_direction
from defcon.data.pipeline import process_metrica_match
from defcon.epv.epv import EPVEngine
from defcon.viz.spatial import plot_credit_zones, plot_pairwise

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    gd = cfg.path("tracking_dir") / "metrica"
    actions, tracking = process_metrica_match(gd, args.game, cfg)
    directions = infer_playing_direction(tracking)

    credits = pd.read_parquet(cfg.path("data_processed") / f"credits_game{args.game}.parquet")

    # per-action location + acting player (the "attacker"), oriented so attack -> +x
    act = actions[["action_id", "team", "player", "period", "start_x", "start_y"]].rename(
        columns={"player": "attacker"})
    dvec = act.apply(lambda r: directions.get((r["team"], int(r["period"])), 1), axis=1)
    act["x"] = act["start_x"] * dvec
    act["y"] = act["start_y"] * dvec

    df = credits.merge(act[["action_id", "attacker", "x", "y"]], on="action_id", how="left").dropna(
        subset=["x", "y"])
    df = df.rename(columns={"player": "defender"})

    # 8.2 spatial zones
    z = plot_credit_zones(df, "docs/img/credit_zones.png", cfg,
                          title=f"Where defensive credit is created — Game {args.game}")
    print(f"[8.2] saved {z}")

    # 8.3 pairwise matrix (defender x attacker)
    p = plot_pairwise(df[["defender", "attacker", "value"]], "docs/img/pairwise_matrix.png", cfg,
                      top_k=10, title=f"Pairwise credit — defender vs. attacker (Game {args.game})")
    print(f"[8.3] saved {p}")
    n_pairs = df.groupby(["defender", "attacker"]).ngroups
    print(f"[info] {len(df)} credit events over {n_pairs} defender-attacker pairs")


if __name__ == "__main__":
    main()
