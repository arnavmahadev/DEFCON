"""Build PyG graph datasets for the component models (supports Phase 3).

Turns a processed match (actions + tracking) into lists of
``torch_geometric.data.Data`` graphs with the right labels/masks per task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config
from defcon.data.metrica import identify_goalkeepers, infer_playing_direction
from defcon.features.graph import X_COL, add_action_of_interest, to_pyg_data
from defcon.features.nodes import build_node_features
from defcon.features.state import graph_state_from_action

__all__ = [
    "build_pass_success_graphs",
    "build_action_selection_graphs",
    "build_goal_condition_graphs",
    "build_responsibility_graphs",
    "build_shot_blocking_graphs",
]


def _match_context(tracking, cfg):
    return infer_playing_direction(tracking), identify_goalkeepers(tracking)


def build_pass_success_graphs(
    actions: pd.DataFrame,
    tracking: pd.DataFrame,
    cfg: Config | None = None,
) -> list:
    """Graphs for pass-success (b1): target node = intended receiver, y = completed.

    Skips passes whose intended receiver is not present in the synced frame.
    """
    cfg = cfg or load_config()
    directions = infer_playing_direction(tracking)
    goalkeepers = identify_goalkeepers(tracking)

    graphs = []
    passes = actions[actions["type"] == "pass"]
    for _, action in passes.iterrows():
        state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
        if state is None:
            continue
        recv = action.get("inferred_receiver")
        if recv is None or pd.isna(recv) or str(recv) not in state.player_ids:
            continue
        target_idx = state.player_ids.index(str(recv))
        # Don't let the target collapse onto the carrier (bad inference).
        if target_idx == state.carrier_idx:
            continue
        y = 1.0 if action["outcome"] == "success" else 0.0
        ng = build_node_features(state, cfg)
        data = to_pyg_data(ng, cfg, target_node=target_idx, y=y)
        data.action_id = int(action["action_id"])
        graphs.append(data)
    return graphs


def build_action_selection_graphs(actions, tracking, cfg: Config | None = None) -> list:
    """Graphs for action-selection (a1): candidates = teammates + goal nodes.

    Target = intended receiver (pass) or the nearer goal post (shot). Softmax runs
    over the candidate nodes; label is the chosen option.
    """
    import torch

    cfg = cfg or load_config()
    directions, goalkeepers = _match_context(tracking, cfg)
    graphs = []
    onball = actions[actions["type"].isin(["pass", "shot"])]
    for _, action in onball.iterrows():
        state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
        if state is None:
            continue
        ng = build_node_features(state, cfg)
        data = to_pyg_data(ng, cfg)
        candidate = data.teammate_mask | data.goal_mask

        if action["type"] == "pass":
            recv = action.get("inferred_receiver")
            if recv is None or pd.isna(recv) or str(recv) not in state.player_ids:
                continue
            target_idx = state.player_ids.index(str(recv))
            if target_idx == state.carrier_idx:
                continue
        else:  # shot -> nearer goal post node
            gx = data.x[:, X_COL["x"]].numpy()
            gy = data.x[:, X_COL["y"]].numpy()
            cx, cy = gx[state.carrier_idx], gy[state.carrier_idx]
            goal_ids = np.flatnonzero(data.goal_mask.numpy())
            d = np.hypot(gx[goal_ids] - cx, gy[goal_ids] - cy)
            target_idx = int(goal_ids[int(np.argmin(d))])

        if not bool(candidate[target_idx]):
            continue
        sel = torch.zeros(data.num_nodes, dtype=torch.bool)
        sel[target_idx] = True
        data.candidate_mask = candidate
        data.select_target = sel
        data.action_id = int(action["action_id"])
        graphs.append(data)
    return graphs


def build_goal_condition_graphs(actions, tracking, label_col: str, cfg: Config | None = None) -> list:
    """Graphs for goal-scoring (c1) / conceding (c2): outcome-conditioned per option.

    ``label_col`` is 'scores_next' or 'concedes_next'. Target node = intended
    receiver of the observed pass; ``obs_outcome`` selects which of the model's
    two per-node logits to train.
    """
    import torch

    cfg = cfg or load_config()
    directions, goalkeepers = _match_context(tracking, cfg)
    graphs = []
    passes = actions[actions["type"] == "pass"]
    for _, action in passes.iterrows():
        state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
        if state is None:
            continue
        recv = action.get("inferred_receiver")
        if recv is None or pd.isna(recv) or str(recv) not in state.player_ids:
            continue
        target_idx = state.player_ids.index(str(recv))
        if target_idx == state.carrier_idx:
            continue
        ng = build_node_features(state, cfg)
        data = to_pyg_data(ng, cfg, target_node=target_idx, y=float(action[label_col]))
        data.obs_outcome = torch.tensor([1 if action["outcome"] == "success" else 0], dtype=torch.long)
        data.action_id = int(action["action_id"])
        graphs.append(data)
    return graphs


def build_shot_blocking_graphs(
    actions,
    tracking,
    cfg: Config | None = None,
    uxg=None,
    augment: bool = False,
    uxg_threshold: float = 0.05,
    lane_half_width: float = 1.5,
) -> list:
    """Graphs for shot-blocking (b2): graph-level, ``y`` = the shot was blocked.

    Real positives are shots with ``shot_outcome == 'B'`` (PFF blocked code); all
    other shots are negatives. Shots are scarce and selection-biased (players do
    not shoot into blocks), so with ``augment=True`` and a fitted UxG model we add
    the paper's proxy positives (task 3.5): high-danger passes (UxG at the ball
    location > ``uxg_threshold``) that had a defender standing in the ball->goal
    lane — a shot there would very likely have been blocked. Proxy graphs are
    tagged ``is_proxy = 1`` so they can be down-weighted or excluded at eval time.
    """
    from defcon.features.geometry import dist_point_to_segments, points_in_triangle

    cfg = cfg or load_config()
    directions, goalkeepers = _match_context(tracking, cfg)
    graphs = []

    def _state_graph(action, y, proxy):
        state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
        if state is None:
            return None
        ng = build_node_features(state, cfg)
        data = to_pyg_data(ng, cfg, y=float(y))
        data.action_id = int(action["action_id"])
        data.is_proxy = int(proxy)
        return data, state

    shots = actions[actions["type"] == "shot"]
    for _, action in shots.iterrows():
        y = 1.0 if str(action.get("shot_outcome")) == "B" else 0.0
        res = _state_graph(action, y, proxy=False)
        if res is not None:
            graphs.append(res[0])

    if augment and uxg is not None:
        passes = actions[(actions["type"] == "pass")]
        posts = None
        for _, action in passes.iterrows():
            res = _state_graph(action, 0.0, proxy=True)
            if res is None:
                continue
            data, state = res
            # UxG at the ball-carrier location (shift oriented coords to corner origin).
            cx = state.px[state.carrier_idx] + cfg.pitch.length / 2.0
            cy = state.py[state.carrier_idx] + cfg.pitch.width / 2.0
            if float(uxg.score_location(cx, cy)) < uxg_threshold:
                continue
            # A defender standing in the ball->goal shooting lane blocks the shot.
            posts = state.attacking_goalposts  # [[x, +h], [x, -h]] in oriented coords
            bx, by = state.px[state.carrier_idx], state.py[state.carrier_idx]
            dfd = np.flatnonzero(state.is_attacking == 0)
            if len(dfd) == 0:
                continue
            in_tri = points_in_triangle(
                state.px[dfd], state.py[dfd], bx, by, posts[0, 0], posts[0, 1], posts[1, 0], posts[1, 1])
            gx, gy = state.attacking_goal_center
            lane_d = dist_point_to_segments(state.px[dfd], state.py[dfd], bx, by, gx, gy)
            blocked = bool(np.any(in_tri & (lane_d <= lane_half_width)))
            if blocked:
                data.y = data.y * 0 + 1.0  # relabel proxy as blocked
                graphs.append(data)

    return graphs


def build_responsibility_graphs(actions, tracking, cfg: Config | None = None) -> list:
    """Graphs for defender-responsibility (d1): 26-feature graph, softmax over defenders.

    Trained on failed passes with a known interceptor. The 26th feature marks the
    'action of interest' (the intended receiver being defended); the target is the
    defender who actually won the ball.
    """
    import torch

    cfg = cfg or load_config()
    directions, goalkeepers = _match_context(tracking, cfg)
    graphs = []
    failed = actions[(actions["type"] == "pass") & (actions["outcome"] == "fail")]
    for _, action in failed.iterrows():
        interceptor = action.get("interceptor")
        recv = action.get("inferred_receiver")
        if interceptor is None or pd.isna(interceptor):
            continue
        if recv is None or pd.isna(recv):
            continue
        state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
        if state is None or str(recv) not in state.player_ids:
            continue
        if str(interceptor) not in state.player_ids:
            continue
        option_idx = state.player_ids.index(str(recv))
        target_idx = state.player_ids.index(str(interceptor))
        ng = build_node_features(state, cfg)
        base = to_pyg_data(ng, cfg)
        # Target must be a defender (opponent of the possession).
        if not bool(base.defender_mask[target_idx]):
            continue
        data = add_action_of_interest(base, option_idx)  # 25 -> 26 features
        sel = torch.zeros(data.num_nodes, dtype=torch.bool)
        sel[target_idx] = True
        data.candidate_mask = base.defender_mask.clone()
        data.select_target = sel
        data.action_id = int(action["action_id"])
        graphs.append(data)
    return graphs
