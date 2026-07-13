#!/usr/bin/env python
"""End-to-end defensive credit assignment on a Metrica match (Phase 5).

raw data -> component models -> EPV -> team defensive value -> credit rules
-> per-player Intercept / Disturb / Deter / Concede.

Usage: python scripts/compute_credits.py --game 1
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from defcon import load_config
from defcon.credit.engine import CreditEngine, aggregate_by_category
from defcon.credit.rules import Credit
from defcon.data.pipeline import process_metrica_match
from defcon.epv.epv import EPVEngine

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    gd = cfg.path("tracking_dir") / "metrica"
    actions, tracking = process_metrica_match(gd, args.game, cfg)
    epv = EPVEngine.from_checkpoints(cfg)
    engine = CreditEngine(epv, cfg)

    print(f"[credit] processing game {args.game} ({len(actions)} on-ball actions) ...")
    credits = engine.process_match(actions, tracking)
    print(f"[credit] emitted {len(credits)} credit records")

    print("\n=== scenario distribution (5.2 router) ===")
    print(credits.drop_duplicates("action_id")["case"].value_counts().to_string())

    print("\n=== total credit by category (5.7) ===")
    print(credits.groupby("category")["value"].agg(["count", "sum"]).round(3).to_string())

    # player -> team for labeling
    player_team = (
        actions.dropna(subset=["player"]).groupby("player")["team"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    items = [Credit(r.player, r.value, r.category) for r in credits.itertuples()]
    table = aggregate_by_category(items, player_team)

    print("\n=== per-player defensive credit (top 12 by net) ===")
    cols = ["player", "team", "intercept", "disturb", "deter", "concede", "net"]
    show = table[cols].copy()
    for c in ["intercept", "disturb", "deter", "concede", "net"]:
        show[c] = show[c].round(3)
    print(show.head(12).to_string(index=False))
    print("\n=== bottom 5 by net (most conceded) ===")
    print(show.tail(5).to_string(index=False))

    out = cfg.path("data_processed") / f"credits_game{args.game}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    credits.to_parquet(out, index=False)
    print(f"\n[credit] saved -> {out}")


if __name__ == "__main__":
    main()
