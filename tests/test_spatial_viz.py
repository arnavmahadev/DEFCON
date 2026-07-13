"""Smoke tests for the analysis figures (8.2 zones, 8.3 pairwise)."""

import numpy as np
import pandas as pd

from defcon import load_config
from defcon.viz.spatial import plot_credit_zones, plot_pairwise


def test_credit_zones_runs(tmp_path):
    cfg = load_config()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(-50, 50, 200), "y": rng.uniform(-30, 30, 200),
        "value": rng.uniform(-0.2, 0.5, 200),
        "category": rng.choice(["intercept", "disturb", "deter", "concede"], 200),
    })
    out = plot_credit_zones(df, tmp_path / "zones.png", cfg)
    assert out.exists() and out.stat().st_size > 0


def test_pairwise_runs(tmp_path):
    cfg = load_config()
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "defender": rng.choice([f"D{i}" for i in range(6)], 150),
        "attacker": rng.choice([f"A{i}" for i in range(6)], 150),
        "value": rng.uniform(-0.3, 0.5, 150),
    })
    out = plot_pairwise(df, tmp_path / "pw.png", cfg, top_k=5)
    assert out.exists() and out.stat().st_size > 0
