"""The 25-dimensional node feature vector (task 2.1 / paper Section 3.1).

Nodes = all players (both teams) + two goal nodes (the attacking goalposts).
Features are grouped and concatenated in a fixed order (see ``NODE_FEATURES``):

    4 binary   : is_carrier, is_teammate, is_gk, is_goal_node
    6 running  : x, y, vx, vy, speed, accel
    3 goal     : dist_to_goal, sin/cos(node->goal angle)
    6 ball     : ball_height, dist_to_ball, sin/cos(node->ball angle),
                 sin/cos(node_vel vs carrier_vel)
    4 opponent : dist_nearest_opp, n_opp_within_r, n_opp_goalside, n_opp_in_triangle
    2 passline : passline_dist_nearest_opp, n_opp_in_corridor

Coordinates are assumed already oriented so the attacking team attacks +x
(see :class:`defcon.features.state.GraphState`).
"""

from __future__ import annotations

import numpy as np

from defcon.config import Config, load_config
from defcon.features.geometry import (
    angle_sin_cos,
    dist_point_to_segments,
    points_in_triangle,
    vector_angle_sin_cos,
)
from defcon.features.state import GraphState

__all__ = ["NODE_FEATURES", "build_node_features", "NodeGraph"]

NODE_FEATURES = [
    # 4 binary
    "is_carrier", "is_teammate", "is_gk", "is_goal_node",
    # 6 running
    "x", "y", "vx", "vy", "speed", "accel",
    # 3 goal-relative
    "dist_to_goal", "goal_sin", "goal_cos",
    # 6 ball-related
    "ball_height", "dist_to_ball", "ball_sin", "ball_cos", "carrier_vel_sin", "carrier_vel_cos",
    # 4 opponent-context
    "dist_nearest_opp", "n_opp_within_r", "n_opp_goalside", "n_opp_in_triangle",
    # 2 passing-line
    "passline_dist_nearest_opp", "n_opp_in_corridor",
]
assert len(NODE_FEATURES) == 25


class NodeGraph:
    """A built node-feature matrix plus node metadata (ordering, roles)."""

    def __init__(self, x: np.ndarray, node_ids: list[str], node_types: list[str],
                 is_attacking: np.ndarray, carrier_idx: int):
        self.x = x                      # (N, 25)
        self.node_ids = node_ids        # player_id or "goal_left"/"goal_right"
        self.node_types = node_types    # "player" | "goal"
        self.is_attacking = is_attacking  # (N,) 1 attacking, 0 defending, 0 goal
        self.carrier_idx = carrier_idx

    @property
    def n_nodes(self) -> int:
        return self.x.shape[0]

    def defender_mask(self) -> np.ndarray:
        """Nodes that are opponents of the carrier (defending players)."""
        types = np.array(self.node_types)
        return (types == "player") & (self.is_attacking == 0)

    def teammate_mask(self) -> np.ndarray:
        """Attacking players excluding the carrier (pass targets)."""
        types = np.array(self.node_types)
        mask = (types == "player") & (self.is_attacking == 1)
        mask[self.carrier_idx] = False
        return mask

    def goal_mask(self) -> np.ndarray:
        return np.array([t == "goal" for t in self.node_types])


def build_node_features(state: GraphState, cfg: Config | None = None) -> NodeGraph:
    """Compute the (N, 25) node feature matrix for a GraphState."""
    cfg = cfg or load_config()
    r = cfg.features.nearby_opponent_radius
    corridor_half = cfg.features.passing_corridor_width / 2.0

    P = state.n_players
    posts = state.attacking_goalposts  # (2, 2)

    # ----- assemble all-node position/velocity arrays (players + 2 goals) -----
    nx = np.concatenate([state.px, posts[:, 0]])
    ny = np.concatenate([state.py, posts[:, 1]])
    nvx = np.concatenate([state.pvx, [0.0, 0.0]])
    nvy = np.concatenate([state.pvy, [0.0, 0.0]])
    nspeed = np.concatenate([state.pspeed, [0.0, 0.0]])
    naccel = np.concatenate([state.paccel, [0.0, 0.0]])
    N = P + 2

    is_goal_node = np.zeros(N)
    is_goal_node[P:] = 1.0
    is_carrier = np.zeros(N)
    is_carrier[state.carrier_idx] = 1.0
    is_teammate = np.concatenate([state.is_attacking, [0.0, 0.0]])
    is_gk = np.concatenate([state.is_gk, [0.0, 0.0]])

    # ----- goal-relative -----
    gx, gy = state.attacking_goal_center
    dgx, dgy = gx - nx, gy - ny
    dist_to_goal = np.hypot(dgx, dgy)
    goal_sin, goal_cos = angle_sin_cos(dgx, dgy)

    # ----- ball-related -----
    ball_height = np.full(N, state.ball_z)
    dbx, dby = state.ball_x - nx, state.ball_y - ny
    dist_to_ball = np.hypot(dbx, dby)
    ball_sin, ball_cos = angle_sin_cos(dbx, dby)
    carrier_vx = state.pvx[state.carrier_idx]
    carrier_vy = state.pvy[state.carrier_idx]
    carrier_vel_sin, carrier_vel_cos = vector_angle_sin_cos(nvx, nvy, carrier_vx, carrier_vy)

    # ----- opponent-context -----
    # "Opponents" = the defending (out-of-possession) players. A defending node
    # never counts itself. Attacking players and goal nodes see all defenders.
    defender_player_idx = np.flatnonzero(state.is_attacking == 0)
    cx, cy = state.px[state.carrier_idx], state.py[state.carrier_idx]

    dist_nearest_opp = np.zeros(N)
    n_opp_within_r = np.zeros(N)
    n_opp_goalside = np.zeros(N)
    n_opp_in_triangle = np.zeros(N)
    passline_dist_nearest_opp = np.zeros(N)
    n_opp_in_corridor = np.zeros(N)

    for i in range(N):
        # Opponent set for node i: defenders, excluding self if i is a defender.
        opp = defender_player_idx
        if i < P and state.is_attacking[i] == 0:
            opp = opp[opp != i]

        if len(opp) == 0:
            dist_nearest_opp[i] = state.pitch_length
            passline_dist_nearest_opp[i] = state.pitch_length
            continue

        ox, oy = state.px[opp], state.py[opp]
        d = np.hypot(ox - nx[i], oy - ny[i])
        dist_nearest_opp[i] = d.min()
        n_opp_within_r[i] = int(np.sum(d <= r))
        opp_dist_to_goal = np.hypot(gx - ox, gy - oy)
        n_opp_goalside[i] = int(np.sum(opp_dist_to_goal < dist_to_goal[i]))
        inside = points_in_triangle(ox, oy, nx[i], ny[i],
                                    posts[0, 0], posts[0, 1], posts[1, 0], posts[1, 1])
        n_opp_in_triangle[i] = int(np.sum(inside))

        # passing line carrier -> node (defenders that could intercept)
        if i == state.carrier_idx:
            passline_dist_nearest_opp[i] = 0.0  # no pass to oneself
        else:
            seg = dist_point_to_segments(ox, oy, cx, cy, nx[i], ny[i])
            passline_dist_nearest_opp[i] = seg.min()
            n_opp_in_corridor[i] = int(np.sum(seg <= corridor_half))

    columns = [
        is_carrier, is_teammate, is_gk, is_goal_node,
        nx, ny, nvx, nvy, nspeed, naccel,
        dist_to_goal, goal_sin, goal_cos,
        ball_height, dist_to_ball, ball_sin, ball_cos, carrier_vel_sin, carrier_vel_cos,
        dist_nearest_opp, n_opp_within_r, n_opp_goalside, n_opp_in_triangle,
        passline_dist_nearest_opp, n_opp_in_corridor,
    ]
    x = np.column_stack(columns).astype(np.float32)

    node_ids = list(state.player_ids) + ["goal_left", "goal_right"]
    node_types = ["player"] * P + ["goal", "goal"]
    is_attacking_full = np.concatenate([state.is_attacking, [0.0, 0.0]])
    return NodeGraph(x, node_ids, node_types, is_attacking_full, state.carrier_idx)
