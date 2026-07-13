"""Tests for the PFF FC 2022 World Cup loader (task 7.3).

Exercises the loader against a **synthetic** fixture written in PFF's real on-disk
format (two 4-3-3 teams, two periods that switch ends, a smoothed-frame variant
whose ball is a bare dict). We ship a synthetic fixture rather than redistributing
PFF's data; the loader is additionally verified on the real public sample via
``scripts/download_pff.py --sample`` + ``scripts/pff_demo.py``. The point of PFF
over Metrica is real player identities, so the jersey -> named-player join is the
load-bearing assertion here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from defcon.data.metrica import infer_playing_direction
from defcon.data.pff import (
    PFFMetadata,
    load_pff_metadata,
    load_pff_rosters,
    load_pff_tracking,
    pff_game_paths,
    pff_goalkeepers,
    pff_identity_table,
)

DATA = Path(__file__).parent / "data" / "pff"
GAME = 9001  # synthetic fixture in PFF's on-disk format


@pytest.fixture(scope="module")
def paths():
    return pff_game_paths(DATA, GAME)


@pytest.fixture(scope="module")
def metadata(paths):
    return load_pff_metadata(paths["metadata"])


@pytest.fixture(scope="module")
def rosters(paths, metadata):
    return load_pff_rosters(paths["rosters"], metadata)


@pytest.fixture(scope="module")
def tracking(paths, metadata, rosters):
    return load_pff_tracking(paths["tracking"], metadata, rosters)


def test_metadata_parses(metadata):
    assert isinstance(metadata, PFFMetadata)
    assert {metadata.home_team_name, metadata.away_team_name} == {"Test United", "Fixture City"}
    assert metadata.pitch_length == 105.0 and metadata.pitch_width == 68.0
    assert 29 < metadata.fps < 31  # PFF broadcast tracking is ~29.97 Hz
    assert metadata.side_of_team(metadata.home_team_id) == "home"
    assert metadata.side_of_team("does-not-exist") is None


def test_rosters_carry_identities(rosters):
    assert len(rosters) == 26  # 13 per side (11 starters + 2 subs)
    # every rostered player has an id, a name and a side
    assert (rosters["player_id"].str.len() > 0).all()
    assert (rosters["name"].str.len() > 0).all()
    assert set(rosters["side"].dropna().unique()) == {"home", "away"}
    # position buckets are coarse-grained
    assert set(rosters["position"].unique()) <= {"GK", "DF", "MF", "FW", ""}
    assert {"GK", "DF", "MF", "FW"} <= set(rosters["position"].unique())


def test_tracking_unified_schema(tracking):
    assert list(tracking.columns) == ["period", "frame", "time_s", "team", "player_id", "x", "y", "z"]
    assert set(tracking["team"].unique()) <= {"home", "away", "ball"}
    # coordinates are meters, origin at pitch center (PFF convention == ours)
    assert tracking["x"].abs().max() <= 55.0
    assert tracking["y"].abs().max() <= 40.0
    # every real frame has both teams on the pitch
    assert (tracking["team"] == "home").sum() > 0
    assert (tracking["team"] == "away").sum() > 0


def test_tracking_carries_player_identities(tracking, rosters):
    """A roster player id must appear in the tracking rows (identities flow through)."""
    started = rosters[rosters["started"]]
    a_player_id = started["player_id"].iloc[0]
    assert a_player_id in set(tracking["player_id"])


def test_playing_direction_flips_between_halves(tracking):
    """Reusing the Metrica GK heuristic, teams should switch ends at half time."""
    dirs = infer_playing_direction(tracking)
    # home attacks opposite directions in the two halves it appears in
    home_dirs = {p: d for (t, p), d in dirs.items() if t == "home"}
    if 1 in home_dirs and 2 in home_dirs:
        assert home_dirs[1] == -home_dirs[2]


def test_goalkeepers_from_roster(rosters):
    gks = pff_goalkeepers(rosters, periods=(1, 2))
    # one GK per side per period, and they are distinct players
    assert ("home", 1) in gks and ("away", 1) in gks
    assert gks[("home", 1)] != gks[("away", 1)]


def test_identity_table_is_one_row_per_player(rosters):
    idt = pff_identity_table(rosters)
    assert idt["player_id"].is_unique
    assert set(["player_id", "name", "team_name", "position", "side"]).issubset(idt.columns)
    assert len(idt) == 26


def test_unsmoothed_path_also_loads(paths, metadata, rosters):
    tr = load_pff_tracking(paths["tracking"], metadata, rosters, smoothed=False, limit=10)
    assert len(tr) > 0
    assert set(tr["team"].unique()) <= {"home", "away", "ball"}


def test_fallback_ids_without_roster(paths, metadata):
    """Without a roster, players still load under jersey-based fallback ids."""
    tr = load_pff_tracking(paths["tracking"], metadata, rosters=None, limit=5)
    outfield = tr[tr["team"].isin(["home", "away"])]
    assert outfield["player_id"].str.match(r"^[HA]\d+$").all()


def test_pff_frame_flows_through_credit_stack(tracking, rosters):
    """Integration: a PFF frame builds a GraphState and a valid responsibility.

    Proves real PFF tracking flows through the same feature/credit code path the
    Metrica pipeline uses — the prerequisite for the market-value study.
    """
    import numpy as np

    from defcon.credit.engine import CreditEngine
    from defcon.data.metrica import infer_playing_direction
    from defcon.data.pff import pff_goalkeepers
    from defcon.features.state import graph_state_from_action

    tr = tracking.copy()
    for col in ("vx", "vy", "speed", "accel"):
        tr[col] = 0.0

    directions = infer_playing_direction(tr)
    goalkeepers = pff_goalkeepers(rosters)

    # a frame with the ball present and both teams on the pitch
    ball_frames = tr[tr["team"] == "ball"][["period", "frame"]].drop_duplicates()
    period, frame = int(ball_frames.iloc[0]["period"]), int(ball_frames.iloc[0]["frame"])
    fr = tr[(tr["period"] == period) & (tr["frame"] == frame)]
    ball = fr[fr["team"] == "ball"].iloc[0]
    players = fr[fr["team"].isin(["home", "away"])].copy()
    players["d"] = np.hypot(players["x"] - ball["x"], players["y"] - ball["y"])
    carrier = players.sort_values("d").iloc[0]

    action = {"period": period, "frame": frame, "team": carrier["team"],
              "player": carrier["player_id"], "inferred_receiver": carrier["player_id"]}
    state = graph_state_from_action(tr, action, directions, goalkeepers, cfg=None)
    assert state is not None

    engine = CreditEngine(epv_engine=None)
    resp = engine._geometric_responsibility(state, state.carrier_idx)
    assert resp, "expected a non-empty responsibility distribution"
    assert abs(sum(resp.values()) - 1.0) < 1e-6  # it's a probability distribution
    # responsibility is over the *defending* team's real players
    att_ids = set(state.player_ids[i] for i in range(state.n_players) if state.is_attacking[i] == 1)
    assert set(resp).isdisjoint(att_ids)
