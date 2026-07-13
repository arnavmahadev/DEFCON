"""GraphState: an oriented snapshot of one on-ball action (supports Phase 2).

A GraphState collects everything the node/edge feature builders need for a
single action, already rotated so the *in-possession* team attacks +x:

- player arrays (position, velocity, team membership, GK flag),
- the ball-carrier index,
- the ball state,
- the attacking/defending goal geometry.

Nodes are ordered: players first (stable, recoverable order), then the two goal
nodes (the attacking goalposts). The player_id <-> node index map is preserved
so credit assignment (Phase 5) can map results back to players.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = ["GraphState", "graph_state_from_action"]


@dataclass
class GraphState:
    # Player arrays (length P), oriented so the attacking team attacks +x.
    player_ids: list[str]
    px: np.ndarray
    py: np.ndarray
    pvx: np.ndarray
    pvy: np.ndarray
    pspeed: np.ndarray
    paccel: np.ndarray
    is_attacking: np.ndarray  # 1 if on the in-possession team
    is_gk: np.ndarray
    carrier_idx: int

    # Ball state (oriented).
    ball_x: float
    ball_y: float
    ball_z: float
    ball_vx: float
    ball_vy: float

    # Geometry.
    pitch_length: float
    pitch_width: float
    goal_width: float

    attacking_team: str = ""
    period: int = 1
    frame: int = -1

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def attacking_goal_center(self) -> tuple[float, float]:
        return (self.pitch_length / 2.0, 0.0)

    @property
    def attacking_goalposts(self) -> np.ndarray:
        """The two attacking goalposts as a (2, 2) array [[x, y_left], [x, y_right]]."""
        x = self.pitch_length / 2.0
        h = self.goal_width / 2.0
        return np.array([[x, h], [x, -h]])

    @property
    def defending_goal_center(self) -> tuple[float, float]:
        return (-self.pitch_length / 2.0, 0.0)


def graph_state_from_action(
    tracking: pd.DataFrame,
    action: pd.Series | dict,
    directions: dict[tuple[str, int], int],
    goalkeepers: dict[tuple[str, int], str],
    cfg: Config | None = None,
) -> GraphState | None:
    """Build a GraphState for one action row using the synced frame.

    Returns None if the frame is missing the carrier or has no players.
    """
    cfg = cfg or load_config()
    action = dict(action)
    period = int(action["period"])
    frame = int(action.get("sync_frame", action.get("frame", -1)))
    attacking_team = action["team"]
    carrier_id = action["player"]

    fr = tracking[(tracking["frame"] == frame) & (tracking["period"] == period)]
    players = fr[fr["team"].isin(["home", "away"])]
    if len(players) == 0 or carrier_id not in set(players["player_id"]):
        return None

    direction = directions[(attacking_team, period)]

    def orient(arr):
        return arr * direction  # flip x and y together when direction == -1

    px = orient(players["x"].to_numpy())
    py = orient(players["y"].to_numpy())
    pvx = orient(players["vx"].to_numpy())
    pvy = orient(players["vy"].to_numpy())
    pspeed = players["speed"].to_numpy()
    paccel = players["accel"].to_numpy()
    player_ids = players["player_id"].astype(str).tolist()
    teams = players["team"].to_numpy()

    is_attacking = (teams == attacking_team).astype(float)
    gk_ids = {goalkeepers.get(("home", period)), goalkeepers.get(("away", period))}
    is_gk = np.array([1.0 if pid in gk_ids else 0.0 for pid in player_ids])
    carrier_idx = player_ids.index(str(carrier_id))

    ball = fr[fr["team"] == "ball"]
    if len(ball):
        b = ball.iloc[0]
        bz = 0.0 if pd.isna(b["z"]) else float(b["z"])
        ball_x, ball_y = direction * float(b["x"]), direction * float(b["y"])
        ball_vx = direction * (0.0 if pd.isna(b["vx"]) else float(b["vx"]))
        ball_vy = direction * (0.0 if pd.isna(b["vy"]) else float(b["vy"]))
    else:
        # Fall back to the carrier's position.
        ball_x, ball_y, bz, ball_vx, ball_vy = px[carrier_idx], py[carrier_idx], 0.0, 0.0, 0.0

    return GraphState(
        player_ids=player_ids,
        px=px, py=py, pvx=pvx, pvy=pvy, pspeed=pspeed, paccel=paccel,
        is_attacking=is_attacking, is_gk=is_gk, carrier_idx=carrier_idx,
        ball_x=ball_x, ball_y=ball_y, ball_z=bz, ball_vx=ball_vx, ball_vy=ball_vy,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
        attacking_team=attacking_team, period=period, frame=frame,
    )
