#!/usr/bin/env python
"""Consolidate Table 3 (component metrics) and ranking Tables 5-7 (tasks 7.1, 7.4)."""

from __future__ import annotations

import warnings

import pandas as pd

from defcon import load_config
from defcon.eval.tables import build_table3, top_players

warnings.filterwarnings("ignore")
pd.set_option("display.width", 130)


def main() -> None:
    cfg = load_config()
    ckpt = cfg.path("checkpoints")

    print("=== Table 3 — component metrics (7.1) ===")
    t3 = build_table3(ckpt)
    for c in ("auc", "brier", "f1", "acc", "mrr", "ce"):
        if c in t3:
            t3[c] = t3[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    print(t3.fillna("—").to_string(index=False))
    print("\nnote: on 2 public matches GCN/boosting match-or-beat GAT on b1/a1 — the paper's "
          "GAT edge needs far more data. GIN collapses. b2/c2 data-limited.")

    season_path = cfg.path("data_processed") / "season_credits.parquet"
    if season_path.exists():
        season = pd.read_parquet(season_path)
        for by, title in [("net_p90", "Table 5 — top defenders by NET/90"),
                          ("intercept_p90", "Table 6 — top by INTERCEPT/90"),
                          ("concede_p90", "Table 7 — most CONCEDED/90 (ascending)")]:
            asc = by == "concede_p90"
            tbl = top_players(season, by=by, n=8, ascending=asc)
            for c in [col for col in tbl.columns if col.endswith("_p90") or col == "minutes"]:
                tbl[c] = tbl[c].round(2)
            print(f"\n=== {title} (7.4) ===")
            print(tbl.to_string(index=False))
        print("\nnote: players are ANONYMIZED (Metrica sample) — no Transfermarkt market-value "
              "column is possible; the headline market-value study (7.3) needs real identities.")
    else:
        print("\n[skip] season_credits.parquet not found — run scripts/aggregate_credits.py first.")


if __name__ == "__main__":
    main()
