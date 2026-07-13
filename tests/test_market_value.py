"""Tests for the market-value correlation machinery (task 7.3)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defcon.data.pff import load_pff_metadata, load_pff_rosters, pff_game_paths, pff_identity_table
from defcon.eval.market_value import add_log_value, attach_identities, correlate_value

PFF_DATA = Path(__file__).parent / "data" / "pff"


def _table():
    vals = np.array([1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 4e7, 8e7], float)
    lv = np.log(vals)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(10)],
        "market_value": vals,
        # perfectly positive with log value, and perfectly negative:
        "net_p90": 3.0 * lv,
        "intercept_p90": -2.0 * lv + 5,
        "position": ["CB", "CB", "CB", "CB", "CB", "MF", "MF", "MF", "FW", "FW"],
    })


def test_add_log_value():
    t = add_log_value(_table())
    assert "log_value" in t
    assert t["log_value"].iloc[0] == pytest.approx(np.log(1e5))


def test_correlation_signs_and_magnitude():
    res = correlate_value(_table())
    overall = res[res.group == "overall"].iloc[0]
    # net perfectly positive, intercept perfectly negative with log value.
    assert overall["r_net_p90"] == pytest.approx(1.0, abs=1e-6)
    assert overall["r_intercept_p90"] == pytest.approx(-1.0, abs=1e-6)


def test_position_groups_present():
    res = correlate_value(_table())
    assert set(res["group"]) == {"overall", "CB", "MF", "FW"}
    assert res[res.group == "overall"]["n"].iloc[0] == 10


def test_handles_degenerate_group():
    # A constant metric -> r is NaN, not a crash.
    t = _table()
    t["net_p90"] = 1.0
    res = correlate_value(t)
    assert np.isnan(res[res.group == "overall"]["r_net_p90"].iloc[0])


def test_attach_identities_joins_pff_players():
    """The credit->identity->value bridge, run against a real PFF-format roster."""
    paths = pff_game_paths(PFF_DATA, 9001)
    md = load_pff_metadata(paths["metadata"])
    identities = pff_identity_table(load_pff_rosters(paths["rosters"], md))

    # a per-player credit table keyed by PFF player_id
    two = identities.head(2)
    id_a, name_a = two["player_id"].iloc[0], two["name"].iloc[0]
    id_b, name_b = two["player_id"].iloc[1], two["name"].iloc[1]
    credits = pd.DataFrame({"player": [id_a, id_b], "net_p90": [0.4, 0.1]})

    values = {name_a: 50_000_000, name_b: 180_000_000}
    joined = attach_identities(credits, identities, values)

    assert set(joined["name"]) == {name_a, name_b}
    assert "position" in joined and "market_value" in joined
    assert joined.loc[joined["name"] == name_b, "market_value"].iloc[0] == 180_000_000


def test_transfermarkt_join_by_name_and_asof(tmp_path):
    """attach_transfermarkt: accent-insensitive name join + as-of valuation pick."""
    from defcon.eval.market_value import attach_transfermarkt

    players = pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Jules Koundé", "Lionel Messi", "John Doe"],
        "country_of_citizenship": ["France", "Argentina", "England"],
        "sub_position": ["Centre-Back", "Second Striker", "Left-Back"],
    })
    valuations = pd.DataFrame({
        "player_id": [1, 1, 2, 3],
        "date": ["2021-01-01", "2022-10-01", "2022-05-01", "2030-01-01"],
        "market_value_in_eur": [30e6, 60e6, 50e6, 5e6],
    })
    (tmp_path / "players.csv").write_text(players.to_csv(index=False))
    (tmp_path / "vals.csv").write_text(valuations.to_csv(index=False))

    table = pd.DataFrame({
        "player": ["a", "b"], "name": ["Jules Kounde", "Lionel Messi"],
        "team": ["France", "Argentina"], "net_p90": [0.5, 0.4],
    })
    joined = attach_transfermarkt(table, str(tmp_path / "players.csv"),
                                  str(tmp_path / "vals.csv"), as_of="2022-11-20")
    # "Kounde" matches "Koundé" (accent-insensitive), and the Oct-2022 value wins
    kounde = joined[joined["name"] == "Jules Kounde"].iloc[0]
    assert kounde["market_value"] == 60e6
    # player 3's only valuation is after the cutoff -> not joined to anyone here
    assert set(joined["name"]) == {"Jules Kounde", "Lionel Messi"}


def test_attach_identities_drops_unknown_players():
    identities = pd.DataFrame(
        {"player_id": ["1"], "name": ["A"], "position": ["FW"], "team_name": ["X"]}
    )
    credits = pd.DataFrame({"player": ["1", "999"], "net_p90": [0.2, 0.3]})
    joined = attach_identities(credits, identities)
    assert list(joined["player"]) == ["1"]  # unknown id dropped
