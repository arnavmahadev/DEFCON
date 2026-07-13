"""Tests for per-90 aggregation and season roll-up (Phase 6)."""

import numpy as np
import pandas as pd
import pytest

from defcon import load_config
from defcon.credit.aggregate import aggregate_per90, minutes_played, season_rollup
from defcon.credit.rules import Credit


def test_minutes_played():
    cfg = load_config()  # 25 fps
    # Player A in 2500 frames -> 2500/25/60 = 1.667 min ; ball rows ignored.
    rows = []
    for f in range(2500):
        rows.append({"team": "home", "player_id": "A", "frame": f})
    for f in range(1250):
        rows.append({"team": "away", "player_id": "B", "frame": f})
    rows.append({"team": "ball", "player_id": "ball", "frame": 0})
    trk = pd.DataFrame(rows)
    mins = minutes_played(trk, cfg)
    assert mins["A"] == pytest.approx(2500 / 25 / 60)
    assert mins["B"] == pytest.approx(1250 / 25 / 60)
    assert "ball" not in mins


def test_aggregate_per90_scales_by_minutes():
    items = [Credit("A", 0.5, "intercept"), Credit("A", -0.1, "concede"),
             Credit("B", 0.2, "deter")]
    minutes = {"A": 45.0, "B": 90.0}  # A played a half
    table = aggregate_per90(items, minutes, {"A": "home", "B": "home"})
    a = table[table.player == "A"].iloc[0]
    assert a["net"] == pytest.approx(0.4)
    assert a["net_p90"] == pytest.approx(0.4 * 90 / 45)   # doubled (half the minutes)
    b = table[table.player == "B"].iloc[0]
    assert b["net_p90"] == pytest.approx(0.2)             # full match -> unchanged


def test_aggregate_per90_min_minutes_filter():
    items = [Credit("A", 0.5, "intercept"), Credit("B", 0.3, "deter")]
    minutes = {"A": 95.0, "B": 10.0}
    table = aggregate_per90(items, minutes, min_minutes=60)
    assert set(table["player"]) == {"A"}  # B filtered out


def test_season_rollup_sums_matches():
    m1 = ([Credit("A", 0.4, "intercept")], {"A": 90.0})
    m2 = ([Credit("A", 0.2, "intercept"), Credit("A", -0.1, "concede")], {"A": 90.0})
    season = season_rollup([m1, m2], min_minutes=100)
    a = season[season.player == "A"].iloc[0]
    assert a["minutes"] == pytest.approx(180.0)
    assert a["intercept"] == pytest.approx(0.6)
    assert a["net"] == pytest.approx(0.5)
    assert a["net_p90"] == pytest.approx(0.5 * 90 / 180)
