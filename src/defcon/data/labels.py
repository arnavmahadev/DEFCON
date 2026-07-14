"""Possession sequences and goal labels (task 1.5).

The supervision signal for the goal-scoring (c1) and goal-conceding (c2) models:
for each on-ball action, does the acting team score / concede within the next N
events? The window deliberately crosses possession changes (a lost ball can lead
directly to conceding) but not half-time boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = ["add_goal_labels", "add_possession_id", "add_interceptor", "OPPONENT"]

OPPONENT = {"home": "away", "away": "home"}


def add_interceptor(events: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    """Add an ``interceptor`` column: for a failed pass, the opponent who next
    touches the ball (the ball-winner) — the ground truth for responsibility (d1).

    Metrica logs an intercepted pass as a ``BALL LOST`` folded into our pass set;
    the immediately following ``RECOVERY``/action by the opposing team identifies
    who won it. We scan a few events ahead for the first opponent-team actor.
    """
    out = events.sort_values(["period", "frame", "action_id"]).reset_index(drop=True)
    team = out["team"].to_numpy()
    player = out["player"].to_numpy()
    typ = out["type"].to_numpy()
    outcome = out["outcome"].to_numpy()
    n = len(out)
    interceptor = np.array([None] * n, dtype=object)
    for i in range(n):
        if typ[i] == "pass" and outcome[i] == "fail" and team[i] in OPPONENT:
            opp = OPPONENT[team[i]]
            for j in range(i + 1, min(i + 1 + lookahead, n)):
                if team[j] == opp and player[j] is not None and not pd.isna(player[j]):
                    interceptor[i] = player[j]
                    break
    out["interceptor"] = interceptor
    return out


def add_possession_id(events: pd.DataFrame) -> pd.DataFrame:
    """Add a ``possession_id`` that increments whenever the on-ball team changes.

    Only rows with a team in {home, away} participate; a new possession starts at
    each half and whenever the acting team differs from the previous action's.
    """
    out = events.sort_values(["period", "frame", "action_id"]).reset_index(drop=True)
    team = out["team"].to_numpy()
    period = out["period"].to_numpy()
    poss = np.zeros(len(out), dtype=int)
    current = -1
    prev_team = None
    prev_period = None
    for i in range(len(out)):
        if period[i] != prev_period or team[i] != prev_team:
            current += 1
        poss[i] = current
        prev_team, prev_period = team[i], period[i]
    out["possession_id"] = poss
    return out


def add_goal_labels(
    events: pd.DataFrame,
    cfg: Config | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Add ``scores_next`` / ``concedes_next`` binaries per action.

    A goal is a shot with outcome ``success``. For action k by team T, look at the
    next ``horizon`` events *within the same period*: ``scores_next`` = a goal by T
    occurs there; ``concedes_next`` = a goal by T's opponent occurs there.
    """
    cfg = cfg or load_config()
    horizon = horizon if horizon is not None else cfg.labels.horizon_events

    out = events.sort_values(["period", "frame", "action_id"]).reset_index(drop=True)
    is_goal = ((out["type"] == "shot") & (out["outcome"] == "success")).to_numpy()
    team = out["team"].to_numpy()
    period = out["period"].to_numpy()
    goal_team = np.where(is_goal, team, None)

    n = len(out)
    scores = np.zeros(n, dtype=int)
    concedes = np.zeros(n, dtype=int)
    for i in range(n):
        t = team[i]
        if t not in OPPONENT:
            continue
        opp = OPPONENT[t]
        # Window: next `horizon` events in the same period.
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if period[j] != period[i]:
                break
            gt = goal_team[j]
            if gt == t:
                scores[i] = 1
            elif gt == opp:
                concedes[i] = 1
        # Small optimization: nothing else to find once both are set.
    out["scores_next"] = scores
    out["concedes_next"] = concedes
    return out
