#!/usr/bin/env python
"""Train the UxG expected-goal model on Wyscout shots (task 3.6 / Milestone M1).

Splits train/test by match (no shots from the same match on both sides) to avoid
leakage, fits the logistic-regression model, reports AUC / Brier / log-loss, and
saves the fitted model + a metrics JSON.

Usage:
    python scripts/train_uxg.py
    python scripts/train_uxg.py --competitions Italy Spain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from defcon import load_config
from defcon.data.wyscout import load_shots
from defcon.models.uxg import UxGModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", nargs="+", default=None, help="Subset of competitions.")
    parser.add_argument("--out", default=None, help="Model output path (.joblib).")
    args = parser.parse_args()

    cfg = load_config()
    wyscout_dir = cfg.path("wyscout_dir")
    print(f"[uxg] loading shots from {wyscout_dir} ...")
    shots = load_shots(
        wyscout_dir,
        competitions=args.competitions,
        pitch_length=cfg.pitch.length,
        pitch_width=cfg.pitch.width,
    )
    print(f"[uxg] {len(shots):,} shots | goal rate {shots['is_goal'].mean():.3f} "
          f"| headers {shots['is_header'].mean():.3f} | set-pieces {shots['is_set_piece'].mean():.3f}")

    # Train/test split by match to prevent leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.uxg.test_fraction, random_state=cfg.uxg.seed)
    train_idx, test_idx = next(splitter.split(shots, groups=shots["match_id"]))
    train, test = shots.iloc[train_idx], shots.iloc[test_idx]
    print(f"[uxg] train {len(train):,} shots / test {len(test):,} shots "
          f"({train['match_id'].nunique()} / {test['match_id'].nunique()} matches)")

    model = UxGModel(cfg).fit(train)
    metrics = model.evaluate(test)
    print("\n[uxg] test metrics:")
    for k, v in metrics.as_dict().items():
        print(f"  {k:>10}: {v:.4f}" if isinstance(v, float) else f"  {k:>10}: {v}")

    print("\n[uxg] standardized coefficients:")
    for k, v in model.coefficients.items():
        print(f"  {k:>13}: {v:+.3f}")

    # A couple of sanity spot-checks.
    print("\n[uxg] spot checks (P goal):")
    checks = [
        ("penalty (11m center, set-piece)", cfg.pitch.length - 11, cfg.pitch.width / 2, False, True),
        ("6m box, center", cfg.pitch.length - 6, cfg.pitch.width / 2, False, False),
        ("edge of box, center", cfg.pitch.length - 16.5, cfg.pitch.width / 2, False, False),
        ("25m out, center", cfg.pitch.length - 25, cfg.pitch.width / 2, False, False),
        ("tight angle (6m, wide)", cfg.pitch.length - 6, cfg.pitch.width / 2 + 18, False, False),
        ("header 8m center", cfg.pitch.length - 8, cfg.pitch.width / 2, True, False),
    ]
    for label, x, y, header, sp in checks:
        print(f"  {label:<34}: {model.score_location(x, y, is_header=header, is_set_piece=sp):.3f}")

    out = Path(args.out) if args.out else cfg.path("checkpoints") / "uxg.joblib"
    model.save(out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics.as_dict(), indent=2))
    print(f"\n[uxg] saved model -> {out}")
    print(f"[uxg] saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
