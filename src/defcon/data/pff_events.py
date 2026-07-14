"""Parse PFF FC event files into DEFCON's action + tracking schema (task 7.3).

Each PFF event carries a **freeze-frame** — every visible player's ``playerId``,
``x``, ``y``, ``speed`` and the ball — at the moment of the on-ball action. That
means we can build a graph state per action directly from the event, with real
player identities, and never touch the multi-GB tracking files.

``parse_pff_events`` returns ``(actions, tracking)`` shaped exactly like the
Metrica pipeline's output, so :meth:`defcon.credit.engine.CreditEngine.process_match`
runs on it unchanged. Each action gets a unique synthetic ``frame`` whose tracking
rows are that action's freeze-frame; velocities are zeroed (PFF gives speed
magnitude, not a vector), which is a known approximation for this cross-provider
reuse.

Event-type mapping (``possessionEventType``):
    PA, CR  -> pass    (passer/crosser -> receiver/target)
    SH      -> shot
Outcome codes: pass/cross ``C`` = complete (else fail); shot ``G`` = goal (else fail).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.data.pff import PFFMetadata

__all__ = ["parse_pff_events", "PASS_TYPES", "SHOT_TYPES"]

PASS_TYPES = {"PA", "CR"}
SHOT_TYPES = {"SH"}

# possessionEventType -> which field holds the acting player's id
_CARRIER_FIELD = {"PA": "passerPlayerId", "CR": "crosserPlayerId", "SH": "shooterPlayerId"}


def _freeze_frame_rows(event: dict, frame: int, period: int, time_s: float) -> list[dict]:
    """Emit unified tracking rows for one event's embedded freeze-frame."""
    rows: list[dict] = []
    for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
        for p in event.get(key) or []:
            x, y = p.get("x"), p.get("y")
            pid = p.get("playerId")
            if x is None or y is None or pid is None:
                continue
            rows.append({
                "period": period, "frame": frame, "time_s": time_s,
                "team": side, "player_id": str(pid),
                "x": float(x), "y": float(y), "z": np.nan,
                "vx": 0.0, "vy": 0.0, "speed": float(p.get("speed") or 0.0), "accel": 0.0,
            })
    ball = event.get("ball") or []
    b = ball[0] if isinstance(ball, list) and ball else (ball if isinstance(ball, dict) else None)
    if b and b.get("x") is not None:
        rows.append({
            "period": period, "frame": frame, "time_s": time_s,
            "team": "ball", "player_id": "ball",
            "x": float(b["x"]), "y": float(b["y"]),
            "z": float(b["z"]) if b.get("z") is not None else np.nan,
            "vx": 0.0, "vy": 0.0, "speed": 0.0, "accel": 0.0,
        })
    return rows


def parse_pff_events(
    events: list[dict],
    metadata: PFFMetadata,
    match_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a PFF event list into ``(actions, tracking)`` DataFrames.

    ``actions`` holds one row per on-ball pass/shot (ordered), with the columns
    the credit engine reads. ``tracking`` holds the per-action freeze-frames keyed
    by the action's synthetic ``frame``.
    """
    action_rows: list[dict] = []
    tracking_rows: list[dict] = []
    frame = 0

    for e in events:
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        etype = pe.get("possessionEventType")
        if etype not in PASS_TYPES and etype not in SHOT_TYPES:
            continue

        carrier = pe.get(_CARRIER_FIELD[etype])
        if carrier is None:
            continue
        period = int(ge.get("period") or e.get("period") or 1)
        team = "home" if ge.get("homeTeam") else "away"
        time_s = float(e.get("eventTime") or e.get("startTime") or 0.0)

        ff = _freeze_frame_rows(e, frame, period, time_s)
        players_here = {r["player_id"] for r in ff if r["team"] in ("home", "away")}
        if str(carrier) not in players_here:
            continue  # carrier not visible in freeze-frame -> can't build a state

        shot_outcome = None
        if etype in PASS_TYPES:
            atype = "pass"
            outcome = "success" if pe.get("passOutcomeType") == "C" or \
                pe.get("crossOutcomeType") == "C" else "fail"
            receiver = pe.get("receiverPlayerId") or pe.get("targetPlayerId")
        else:
            atype = "shot"
            shot_outcome = pe.get("shotOutcomeType")  # G goal / B blocked / S saved / O off / ...
            outcome = "success" if shot_outcome == "G" else "fail"
            receiver = None

        cx = next((r["x"] for r in ff if r["player_id"] == str(carrier)), np.nan)
        cy = next((r["y"] for r in ff if r["player_id"] == str(carrier)), np.nan)

        action_rows.append({
            "action_id": frame,
            "period": period, "time_s": time_s, "frame": frame,
            "type": atype, "outcome": outcome, "team": team,
            "player": str(carrier),
            "receiver": None if receiver is None else str(receiver),
            "inferred_receiver": None if receiver is None else str(receiver),
            "interceptor": None,
            "shot_outcome": shot_outcome,
            "start_x": cx, "start_y": cy,
            "match_id": match_id,
        })
        tracking_rows.extend(ff)
        frame += 1

    actions = pd.DataFrame(action_rows)
    tracking = pd.DataFrame(tracking_rows)

    # interceptor: on a failed pass, credit the opponent who takes the next action
    if len(actions):
        for i in range(len(actions) - 1):
            a, nxt = actions.iloc[i], actions.iloc[i + 1]
            if a["type"] == "pass" and a["outcome"] == "fail" and nxt["team"] != a["team"]:
                actions.iat[i, actions.columns.get_loc("interceptor")] = nxt["player"]

    return actions, tracking
