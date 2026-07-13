"""Tests for goal labels / possession (1.5) and receiver inference (1.6)."""

import numpy as np
import pandas as pd

from defcon import load_config
from defcon.data.labels import add_goal_labels, add_possession_id
from defcon.data.receiver import infer_intended_receivers


def _events(seq):
    """Build a minimal events frame from (team, type, outcome) tuples."""
    rows = []
    for i, (team, typ, outcome) in enumerate(seq):
        rows.append({
            "action_id": i, "period": 1, "frame": i + 1, "type": typ, "outcome": outcome,
            "team": team, "player": f"{team}_p", "receiver": None,
            "start_x": 0.0, "start_y": 0.0, "end_x": 0.0, "end_y": 0.0,
        })
    return pd.DataFrame(rows)


def test_goal_labels_score_and_concede_windows():
    # home builds up (5 passes), away has one action, then home scores.
    seq = [("home", "pass", "success")] * 4 + [("away", "pass", "fail")] + [("home", "shot", "success")]
    ev = add_goal_labels(_events(seq), horizon=10)
    # The home actions before the goal are labeled scores_next.
    assert ev.loc[0:3, "scores_next"].sum() == 4
    # The away action before the home goal is labeled concedes_next.
    assert ev.loc[4, "concedes_next"] == 1
    assert ev.loc[4, "scores_next"] == 0
    # The goal event itself has nothing after it.
    assert ev.loc[5, "scores_next"] == 0


def test_goal_labels_respect_horizon():
    # Goal is 11 events after the first action -> outside a horizon of 10.
    seq = [("home", "pass", "success")] + [("home", "pass", "success")] * 10 + [("home", "shot", "success")]
    ev = add_goal_labels(_events(seq), horizon=10)
    assert ev.loc[0, "scores_next"] == 0  # goal is the 11th subsequent event
    assert ev.loc[1, "scores_next"] == 1  # now within 10


def test_goal_labels_do_not_cross_period():
    ev = _events([("home", "pass", "success"), ("home", "shot", "success")])
    ev.loc[1, "period"] = 2  # goal is in the next half
    ev = add_goal_labels(ev, horizon=10)
    assert ev.loc[0, "scores_next"] == 0


def test_possession_id_increments_on_team_change():
    ev = _events([("home", "pass", "success"), ("home", "pass", "success"),
                  ("away", "pass", "fail"), ("home", "recovery", "success")])
    ev = add_possession_id(ev)
    assert ev["possession_id"].tolist() == [0, 0, 1, 2]


def _tracking_frame(positions, period=1, frame=5):
    """positions: list of (team, player_id, x, y)."""
    rows = [{"period": period, "frame": frame, "time_s": 0.2, "team": t,
             "player_id": pid, "x": x, "y": y, "z": np.nan} for (t, pid, x, y) in positions]
    return pd.DataFrame(rows)


def test_receiver_picks_teammate_at_endpoint():
    cfg = load_config()
    # Pass from (0,0) to (20,0). Teammate A sits at the endpoint; B is far away.
    trk = _tracking_frame([
        ("home", "passer", 0.0, 0.0),
        ("home", "A", 20.0, 0.0),
        ("home", "B", 5.0, 25.0),
        ("away", "opp", 20.0, 1.0),  # opponent near endpoint must be ignored
    ])
    passes = pd.DataFrame([{
        "period": 1, "frame": 5, "end_frame": 5, "team": "home", "player": "passer",
        "start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 0.0,
    }])
    inferred = infer_intended_receivers(passes, trk, cfg)
    assert inferred.iloc[0] == "A"


def test_receiver_angle_term_discriminates():
    cfg = load_config()
    # Two teammates equidistant from the endpoint, but A is aligned with the pass
    # direction (on the pass line) and B is off to the side -> pick A.
    trk = _tracking_frame([
        ("home", "passer", 0.0, 0.0),
        ("home", "A", 25.0, 0.0),  # 5 m beyond endpoint, on the pass line
        ("home", "B", 20.0, 5.0),  # 5 m from endpoint, off to the side
    ])
    end = np.array([20.0, 0.0])
    a = np.array([25.0, 0.0]); b = np.array([20.0, 5.0])
    assert abs(np.linalg.norm(a - end) - np.linalg.norm(b - end)) < 0.05  # equidistant
    passes = pd.DataFrame([{
        "period": 1, "frame": 5, "end_frame": 5, "team": "home", "player": "passer",
        "start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 0.0,
    }])
    inferred = infer_intended_receivers(passes, trk, cfg)
    assert inferred.iloc[0] == "A"
