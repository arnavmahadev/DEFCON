"""Tests for the PFF event -> action/tracking parser (task 7.3).

Uses inline, synthetic events written in PFF's real shape (gameEvents +
possessionEvents + embedded freeze-frame) so nothing from PFF is redistributed.
"""

from __future__ import annotations

from defcon.data.pff import PFFMetadata
from defcon.data.pff_events import parse_pff_events

MD = PFFMetadata(
    game_id="1", home_team_id="10", home_team_name="Home", away_team_id="20",
    away_team_name="Away", fps=29.97, pitch_length=105.0, pitch_width=68.0,
    home_team_start_left=True,
)


def _frame(home_xy, away_xy, ball):
    """Build homePlayers/awayPlayers/ball blocks from {pid: (x, y)} dicts."""
    hp = [{"playerId": p, "jerseyNum": i, "x": x, "y": y, "speed": 1.0}
          for i, (p, (x, y)) in enumerate(home_xy.items(), 1)]
    ap = [{"playerId": p, "jerseyNum": i, "x": x, "y": y, "speed": 1.0}
          for i, (p, (x, y)) in enumerate(away_xy.items(), 1)]
    return {"homePlayers": hp, "awayPlayers": ap, "ball": [{"x": ball[0], "y": ball[1], "z": 0.0}]}


def _event(period, home, carrier, ptype, **pe):
    home_xy = {101: (-10.0, 0.0), 102: (-30.0, 5.0), 103: (-48.0, 0.0)}
    away_xy = {201: (10.0, 1.0), 202: (30.0, -5.0), 203: (48.0, 0.0)}
    ff = _frame(home_xy, away_xy, (0.0, 0.0))
    ge = {"gameEventType": "OTB", "period": period, "homeTeam": home,
          "teamId": "10" if home else "20", "playerId": carrier}
    pe = {"possessionEventType": ptype, **pe}
    return {"eventTime": pe.pop("_t", 100.0), "gameEvents": ge, "possessionEvents": pe, **ff}


def test_parse_pass_and_shot_actions():
    events = [
        _event(1, True, 101, "PA", passerPlayerId=101, receiverPlayerId=102, passOutcomeType="C"),
        _event(1, True, 102, "PA", passerPlayerId=102, targetPlayerId=103, passOutcomeType="D"),
        _event(1, False, 201, "SH", shooterPlayerId=201, shotOutcomeType="G"),
        _event(1, True, 101, "IT"),  # non-action type -> dropped
    ]
    actions, tracking = parse_pff_events(events, MD, "pff_1")

    assert list(actions["type"]) == ["pass", "pass", "shot"]
    assert list(actions["outcome"]) == ["success", "fail", "success"]
    assert list(actions["team"]) == ["home", "home", "away"]
    # carrier + receiver carried through as string ids
    assert actions.iloc[0]["player"] == "101" and actions.iloc[0]["inferred_receiver"] == "102"
    # failed pass records the intended target as receiver
    assert actions.iloc[1]["inferred_receiver"] == "103"
    # every action's freeze-frame is present in tracking under its own frame
    assert set(tracking["team"]) == {"home", "away", "ball"}
    assert tracking["frame"].nunique() == 3


def test_interceptor_set_when_possession_flips():
    events = [
        _event(1, True, 101, "PA", passerPlayerId=101, targetPlayerId=102, passOutcomeType="D"),
        _event(1, False, 201, "PA", passerPlayerId=201, receiverPlayerId=202, passOutcomeType="C"),
    ]
    actions, _ = parse_pff_events(events, MD, "pff_1")
    # the failed home pass is intercepted by the away player who takes the next action
    assert actions.iloc[0]["interceptor"] == "201"


def test_carrier_absent_from_freezeframe_is_skipped():
    ev = _event(1, True, 999, "PA", passerPlayerId=999, receiverPlayerId=102, passOutcomeType="C")
    actions, _ = parse_pff_events([ev], MD, "pff_1")
    assert len(actions) == 0  # carrier 999 not in the freeze-frame -> no state buildable


def test_cross_maps_to_pass():
    ev = _event(1, True, 101, "CR", crosserPlayerId=101, targetPlayerId=102, crossOutcomeType="C")
    actions, _ = parse_pff_events([ev], MD, "pff_1")
    assert list(actions["type"]) == ["pass"]
    assert actions.iloc[0]["outcome"] == "success"
