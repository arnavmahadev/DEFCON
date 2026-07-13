"""Table 3 (component metrics) and ranking tables 5-7 (tasks 7.1, 7.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = ["build_table3", "top_players"]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def build_table3(checkpoints: Path) -> pd.DataFrame:
    """Assemble a Table-3-style component-metrics summary from saved metric JSONs.

    Reads the individual model metrics plus the backbone baselines (if present).
    Missing values are shown as NaN rather than fabricated.
    """
    rows = []

    uxg = _load(checkpoints / "uxg.metrics.json")
    if uxg:
        rows.append({"model": "c3 UxG", "backbone": "LogReg", "metric_kind": "bin",
                     "auc": uxg.get("auc"), "brier": uxg.get("brier")})

    comp = _load(checkpoints / "components_metrics.json") or {}
    b1 = _load(checkpoints / "pass_success_gat.metrics.json")
    baselines = _load(checkpoints / "baselines.json") or {}

    # b1 across backbones (from baselines.json), else the single GAT run.
    if "b1" in baselines:
        for bk, m in baselines["b1"].items():
            rows.append({"model": "b1 pass", "backbone": bk, "metric_kind": "bin",
                         "auc": m.get("auc"), "brier": m.get("brier"), "f1": m.get("f1")})
    elif b1 and "gat" in b1:
        rows.append({"model": "b1 pass", "backbone": "GAT", "metric_kind": "bin",
                     "auc": b1["gat"].get("auc"), "brier": b1["gat"].get("brier")})

    # a1 across backbones.
    if "a1" in baselines:
        for bk, m in baselines["a1"].items():
            rows.append({"model": "a1 select", "backbone": bk, "metric_kind": "sel",
                         "acc": m.get("accuracy"), "mrr": m.get("mrr"), "ce": m.get("ce")})
    elif "a1" in comp:
        g = comp["a1"].get("gat", {})
        rows.append({"model": "a1 select", "backbone": "GAT", "metric_kind": "sel",
                     "acc": g.get("accuracy"), "mrr": g.get("mrr"), "ce": g.get("ce")})

    for tag, name in (("c1", "c1 score"), ("c2", "c2 concede")):
        m = comp.get(tag)
        if m:
            rows.append({"model": name, "backbone": "GAT", "metric_kind": "bin",
                         "auc": m.get("auc"), "brier": m.get("brier"), "f1": m.get("f1")})

    d1 = comp.get("d1", {}).get("gat") if "d1" in comp else None
    if d1:
        rows.append({"model": "d1 resp.", "backbone": "GAT", "metric_kind": "sel",
                     "acc": d1.get("accuracy"), "mrr": d1.get("mrr"), "ce": d1.get("ce")})

    return pd.DataFrame(rows)


def top_players(season: pd.DataFrame, by: str = "net_p90", n: int = 10,
                ascending: bool = False) -> pd.DataFrame:
    """Top-N players by a credit column (Tables 5-7). Market value column if present."""
    cols = ["player"]
    if "team" in season:
        cols.append("team")
    if "position" in season:
        cols.append("position")
    cols += [c for c in ("intercept_p90", "concede_p90", "net_p90", "minutes") if c in season]
    if "market_value" in season:
        cols.append("market_value")
    return season.sort_values(by, ascending=ascending).head(n)[cols].reset_index(drop=True)
