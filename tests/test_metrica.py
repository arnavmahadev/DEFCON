"""Tests for the Metrica loaders (tasks 1.1 tracking, 1.2 events).

The core parsing/geometry is tested on tiny synthetic Metrica-format CSVs so the
suite is hermetic. A couple of smoke tests run against the real (git-ignored)
sample data when it is present locally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defcon import load_config
from defcon.data.metrica import (
    infer_playing_direction,
    load_metrica_events,
    load_metrica_tracking,
    metrica_game_paths,
    orient_attacking_positive_x,
)

REAL_DATA = Path("data/raw/tracking/metrica")
HAS_REAL = (REAL_DATA / "Sample_Game_1_RawTrackingData_Home_Team.csv").exists()


def _write_team_csv(path: Path, team: str, jerseys: list[int], frames: list[dict]):
    """Write a minimal Metrica-format team tracking CSV.

    ``frames`` is a list of dicts: {"period","frame","time", jersey->(x,y), "ball":(x,y)}.
    """
    players = [f"Player{j}" for j in jerseys]
    header1 = ["", "", ""] + sum(([team, ""] for _ in players), []) + ["", ""]
    header2 = ["", "", ""] + sum(([str(j), ""] for j in jerseys), []) + ["", ""]
    header3 = ["Period", "Frame", "Time [s]"] + sum(([p, ""] for p in players), []) + ["Ball", ""]

    lines = [",".join(map(str, h)) for h in (header1, header2, header3)]
    for fr in frames:
        row = [fr["period"], fr["frame"], fr["time"]]
        for j in jerseys:
            x, y = fr.get(j, (np.nan, np.nan))
            row += [x, y]
        bx, by = fr["ball"]
        row += [bx, by]
        lines.append(",".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def synthetic_game(tmp_path):
    # Home GK (jersey 1) deep at x=0.05 (attacks +x); Away GK (jersey 15) at x=0.95.
    home_frames = [
        {"period": 1, "frame": 1, "time": 0.04, 1: (0.05, 0.5), 7: (0.6, 0.4), "ball": (0.6, 0.4)},
        {"period": 1, "frame": 2, "time": 0.08, 1: (0.05, 0.5), 7: (0.61, 0.41), "ball": (0.61, 0.41)},
    ]
    away_frames = [
        {"period": 1, "frame": 1, "time": 0.04, 15: (0.95, 0.5), 20: (0.4, 0.6), "ball": (0.6, 0.4)},
        {"period": 1, "frame": 2, "time": 0.08, 15: (0.95, 0.5), 20: (0.41, 0.61), "ball": (0.61, 0.41)},
    ]
    _write_team_csv(tmp_path / "Sample_Game_1_RawTrackingData_Home_Team.csv", "Home", [1, 7], home_frames)
    _write_team_csv(tmp_path / "Sample_Game_1_RawTrackingData_Away_Team.csv", "Away", [15, 20], away_frames)
    return tmp_path


def test_tracking_long_schema_and_units(synthetic_game):
    cfg = load_config()
    trk = load_metrica_tracking(game_dir=synthetic_game)
    assert list(trk.columns) == ["period", "frame", "time_s", "team", "player_id", "x", "y", "z"]
    # (0.5, 0.5) normalized -> (0, 0) meters.
    # Home GK at (0.05, 0.5): x = (0.05-0.5)*105 = -47.25 ; y = 0.
    gk = trk[(trk.player_id == "Player1") & (trk.frame == 1)].iloc[0]
    assert gk.x == pytest.approx((0.05 - 0.5) * cfg.pitch.length)
    assert gk.y == pytest.approx(0.0)
    # Ball present once per frame (deduplicated across the two files).
    ball = trk[(trk.team == "ball") & (trk.frame == 1)]
    assert len(ball) == 1


def test_tracking_within_pitch_bounds(synthetic_game):
    cfg = load_config()
    trk = load_metrica_tracking(game_dir=synthetic_game)
    assert trk.x.abs().max() <= cfg.pitch.length / 2 + 1e-9
    assert trk.y.abs().max() <= cfg.pitch.width / 2 + 1e-9


def test_infer_direction_from_gk(synthetic_game):
    trk = load_metrica_tracking(game_dir=synthetic_game)
    dirs = infer_playing_direction(trk)
    # Home GK deep at -x -> home attacks +x ; Away GK deep at +x -> away attacks -x.
    assert dirs[("home", 1)] == 1
    assert dirs[("away", 1)] == -1


def test_orient_flips_only_when_needed(synthetic_game):
    trk = load_metrica_tracking(game_dir=synthetic_game)
    dirs = infer_playing_direction(trk)
    frame = trk[trk.frame == 1]
    # Home already attacks +x -> unchanged.
    assert np.allclose(orient_attacking_positive_x(frame, "home", 1, dirs).x, frame.x)
    # Away attacks -x -> flipped.
    assert np.allclose(orient_attacking_positive_x(frame, "away", 1, dirs).x, -frame.x)


def _write_events_csv(path: Path, rows: list[dict]):
    cols = ["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]",
            "End Frame", "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_event_type_and_outcome_mapping(tmp_path):
    rows = [
        # completed pass
        {"Team": "Home", "Type": "PASS", "Subtype": np.nan, "Period": 1, "Start Frame": 1,
         "Start Time [s]": 0.04, "End Frame": 3, "End Time [s]": 0.12, "From": "Player1",
         "To": "Player7", "Start X": 0.5, "Start Y": 0.5, "End X": 0.6, "End Y": 0.5},
        # intercepted pass -> failed pass
        {"Team": "Home", "Type": "BALL LOST", "Subtype": "INTERCEPTION", "Period": 1, "Start Frame": 5,
         "Start Time [s]": 0.2, "End Frame": 9, "End Time [s]": 0.36, "From": "Player7",
         "To": np.nan, "Start X": 0.6, "Start Y": 0.5, "End X": 0.7, "End Y": 0.5},
        # goal
        {"Team": "Home", "Type": "SHOT", "Subtype": "ON TARGET-GOAL", "Period": 1, "Start Frame": 11,
         "Start Time [s]": 0.44, "End Frame": 13, "End Time [s]": 0.52, "From": "Player9",
         "To": np.nan, "Start X": 0.9, "Start Y": 0.5, "End X": 1.0, "End Y": 0.5},
        # missed shot
        {"Team": "Home", "Type": "SHOT", "Subtype": "OFF TARGET", "Period": 1, "Start Frame": 15,
         "Start Time [s]": 0.6, "End Frame": 17, "End Time [s]": 0.68, "From": "Player9",
         "To": np.nan, "Start X": 0.85, "Start Y": 0.55, "End X": 1.0, "End Y": 0.6},
    ]
    path = tmp_path / "events.csv"
    _write_events_csv(path, rows)
    ev = load_metrica_events(events_csv=path)

    assert ev.loc[0, "type"] == "pass" and ev.loc[0, "outcome"] == "success"
    assert ev.loc[0, "receiver"] == "Player7"
    assert ev.loc[1, "type"] == "pass" and ev.loc[1, "outcome"] == "fail"
    assert ev.loc[2, "type"] == "shot" and ev.loc[2, "outcome"] == "success"
    assert ev.loc[3, "type"] == "shot" and ev.loc[3, "outcome"] == "fail"


# --------------------------------------------------------------------------- #
# Smoke tests against the real sample data (skipped when absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_REAL, reason="Metrica sample data not downloaded")
def test_real_tracking_frame_is_11v11():
    trk = load_metrica_tracking(game_dir=REAL_DATA)
    frame = int(trk[trk.period == 1]["frame"].median())
    fr = trk[trk.frame == frame]
    assert (fr.team == "home").sum() == 11
    assert (fr.team == "away").sum() == 11


@pytest.mark.skipif(not HAS_REAL, reason="Metrica sample data not downloaded")
def test_real_ball_usually_at_a_player():
    """The ball sits at a player's feet most of the time (it's only far mid-pass)."""
    trk = load_metrica_tracking(game_dir=REAL_DATA)
    frames = np.linspace(trk.frame.min(), trk.frame.max(), 300).astype(int)
    sample = trk[trk.frame.isin(frames)]
    nearest = []
    for _, fr in sample.groupby("frame"):
        ball = fr[fr.team == "ball"][["x", "y"]].to_numpy()
        players = fr[fr.team.isin(["home", "away"])][["x", "y"]].to_numpy()
        if len(ball) and len(players):
            nearest.append(np.min(np.linalg.norm(players - ball, axis=1)))
    # Median ball-to-nearest-player distance should be small (~at feet).
    assert np.median(nearest) < 3.0


@pytest.mark.skipif(not HAS_REAL, reason="Metrica sample data not downloaded")
def test_real_directions_consistent():
    trk = load_metrica_tracking(game_dir=REAL_DATA)
    dirs = infer_playing_direction(trk)
    assert dirs[("home", 1)] == -dirs[("away", 1)]  # opposite ways
    assert dirs[("home", 1)] == -dirs[("home", 2)]  # flip at half


@pytest.mark.skipif(not HAS_REAL, reason="Metrica sample data not downloaded")
def test_real_pass_completion_sane():
    ev = load_metrica_events(game_dir=REAL_DATA)
    passes = ev[ev.type == "pass"]
    assert 300 < len(passes) < 1500
    assert 0.7 < passes.outcome.eq("success").mean() < 0.95
