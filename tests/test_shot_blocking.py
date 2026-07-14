"""Tests for the shot-blocking (b2) data path and graph trainer (task 3.5).

Uses synthetic PFF-shaped events (nothing from PFF redistributed) so the whole
path — parse -> label -> build graphs -> train — runs on fixtures.
"""

from __future__ import annotations

from defcon import load_config
from defcon.data.pff import PFFMetadata
from defcon.data.pff_events import parse_pff_events
from defcon.features.dataset import build_shot_blocking_graphs

MD = PFFMetadata(
    game_id="1", home_team_id="10", home_team_name="Home", away_team_id="20",
    away_team_name="Away", fps=29.97, pitch_length=105.0, pitch_width=68.0,
    home_team_start_left=True,
)


def _frame():
    home_xy = {101: (-10.0, 0.0), 102: (-30.0, 5.0), 103: (-48.0, 0.0)}
    away_xy = {201: (10.0, 1.0), 202: (30.0, -5.0), 203: (48.0, 0.0)}
    hp = [{"playerId": p, "jerseyNum": i, "x": x, "y": y, "speed": 1.0}
          for i, (p, (x, y)) in enumerate(home_xy.items(), 1)]
    ap = [{"playerId": p, "jerseyNum": i, "x": x, "y": y, "speed": 1.0}
          for i, (p, (x, y)) in enumerate(away_xy.items(), 1)]
    return {"homePlayers": hp, "awayPlayers": ap, "ball": [{"x": 0.0, "y": 0.0, "z": 0.0}]}


def _shot(carrier, code, t):
    ge = {"gameEventType": "OTB", "period": 1, "homeTeam": True, "teamId": "10", "playerId": carrier}
    pe = {"possessionEventType": "SH", "shooterPlayerId": carrier, "shotOutcomeType": code}
    return {"eventTime": t, "gameEvents": ge, "possessionEvents": pe, **_frame()}


def test_shot_outcome_code_carried_through():
    actions, _ = parse_pff_events([_shot(101, "B", 10.0)], MD, "pff_1")
    assert actions.iloc[0]["type"] == "shot"
    assert actions.iloc[0]["shot_outcome"] == "B"
    assert actions.iloc[0]["outcome"] == "fail"  # blocked != goal


def test_blocked_shot_is_positive_others_negative():
    cfg = load_config()
    events = [_shot(101, "B", 10.0), _shot(102, "O", 20.0), _shot(103, "G", 30.0)]
    actions, tracking = parse_pff_events(events, MD, "pff_1")
    graphs = build_shot_blocking_graphs(actions, tracking, cfg)
    assert len(graphs) == 3
    labels = {int(g.action_id): g.y.item() for g in graphs}
    # exactly the blocked shot (first action) is the positive
    assert sum(labels.values()) == 1.0
    assert labels[0] == 1.0
    # graph-level label, 25-dim node features (not the 26-dim responsibility variant)
    assert graphs[0].x.shape[1] == cfg.features.node_dim
    assert graphs[0].is_proxy == 0
