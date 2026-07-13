#!/usr/bin/env python
"""Run the Phase 1 pipeline on a Metrica game and print a summary.

Usage:
    python scripts/process_match.py --game 1
    python scripts/process_match.py --game 1 --force   # ignore cache
"""

from __future__ import annotations

import argparse

from defcon import load_config
from defcon.data.pipeline import process_metrica_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=int, default=1)
    parser.add_argument("--game-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Recompute, ignore cache.")
    args = parser.parse_args()

    cfg = load_config()
    game_dir = args.game_dir or (cfg.path("tracking_dir") / "metrica")

    actions, tracking = process_metrica_match(game_dir, args.game, cfg, force=args.force)

    print(f"[pipeline] tracking rows: {len(tracking):,}")
    print(f"[pipeline] on-ball actions: {len(actions):,}")
    passes = actions[actions.type == "pass"]
    shots = actions[actions.type == "shot"]
    print(f"[pipeline]   passes: {len(passes)} (completion {passes.outcome.eq('success').mean():.3f})")
    print(f"[pipeline]   shots:  {len(shots)} (goals {shots.outcome.eq('success').sum()})")
    print(f"[pipeline] scores_next rate:   {actions.scores_next.mean():.4f}")
    print(f"[pipeline] concedes_next rate: {actions.concedes_next.mean():.4f}")
    print(f"[pipeline] possessions: {actions.possession_id.nunique()}")
    failed = passes[passes.outcome == "fail"]
    print(f"[pipeline] failed passes with inferred receiver: "
          f"{failed.inferred_receiver.notna().sum()} / {len(failed)}")
    print(f"[pipeline] median sync_dist: {actions.sync_dist.median():.2f} m")


if __name__ == "__main__":
    main()
