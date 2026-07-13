"""Tests for the selection and outcome-conditioned trainers (a1/d1, c1/c2)."""

import numpy as np
import pytest

from defcon import load_config
from defcon.features.graph import to_pyg_data
from defcon.features.nodes import build_node_features
from defcon.features.state import GraphState

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from defcon.models.gat import GATNodeModel  # noqa: E402
from defcon.models.train import (  # noqa: E402
    selection_metrics,
    train_outcome_conditioned_model,
    train_selection_model,
)


def test_selection_metrics_math():
    m = selection_metrics([1, 2, 4], ce_sum=0.0)
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 3)
    assert m["n"] == 3


def _selection_graph(cfg, seed=0):
    """Carrier + 3 teammates at different distances + a defender. Target = the
    teammate nearest the carrier (a learnable positional signal)."""
    rng = np.random.default_rng(seed)
    dists = np.array([8.0, 18.0, 30.0]) + rng.uniform(-2, 2, 3)
    angles = rng.uniform(-0.6, 0.6, 3)
    tx = dists * np.cos(angles)
    ty = dists * np.sin(angles)
    px = np.concatenate([[0.0], tx, [10.0]])
    py = np.concatenate([[0.0], ty, [12.0]])
    state = GraphState(
        player_ids=["carrier", "m1", "m2", "m3", "d1"],
        px=px, py=py, pvx=np.zeros(5), pvy=np.zeros(5),
        pspeed=np.zeros(5), paccel=np.zeros(5),
        is_attacking=np.array([1.0, 1, 1, 1, 0.0]), is_gk=np.zeros(5),
        carrier_idx=0, ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=1.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
    )
    ng = build_node_features(state, cfg)
    data = to_pyg_data(ng, cfg)
    candidate = data.teammate_mask | data.goal_mask
    nearest = 1 + int(np.argmin(dists))  # node index of closest teammate
    sel = torch.zeros(data.num_nodes, dtype=torch.bool)
    sel[nearest] = True
    data.candidate_mask = candidate
    data.select_target = sel
    return data


def test_selection_trainer_learns_positional_signal():
    cfg = load_config(overrides={"training": {"max_epochs": 40, "patience": 40, "batch_size": 8}})
    graphs = [_selection_graph(cfg, s) for s in range(24)]
    model = GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=16, heads=2, mlp_hidden=16)
    res = train_selection_model(model, graphs, graphs, cfg, monitor="mrr", verbose=False)
    # Should beat random selection over ~5 candidates (MRR ~0.46) comfortably.
    assert res.val_metrics["mrr"] > 0.6


def _outcome_graph(cfg, positive=True, seed=0):
    """A graph whose label correlates with a node feature so c1/c2 can learn."""
    rng = np.random.default_rng(seed)
    # positives: target teammate very close to goal; negatives: far.
    tx = (95.0 if positive else 30.0) + rng.uniform(-3, 3)
    px = np.array([0.0, tx - 52.5, 20.0, 3.0])  # center origin coords (goal at +52.5)
    py = np.array([0.0, 0.0, 5.0, 0.0])
    state = GraphState(
        player_ids=["carrier", "recv", "m2", "d1"],
        px=px, py=py, pvx=np.zeros(4), pvy=np.zeros(4),
        pspeed=np.zeros(4), paccel=np.zeros(4),
        is_attacking=np.array([1.0, 1, 1, 0.0]), is_gk=np.zeros(4),
        carrier_idx=0, ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=1.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
    )
    ng = build_node_features(state, cfg)
    data = to_pyg_data(ng, cfg, target_node=1, y=1.0 if positive else 0.0)
    data.obs_outcome = torch.tensor([1], dtype=torch.long)  # observed success
    return data


def test_outcome_conditioned_trainer_runs_and_learns():
    cfg = load_config(overrides={"training": {"max_epochs": 40, "patience": 40, "batch_size": 8}})
    graphs = [_outcome_graph(cfg, positive=(i % 2 == 0), seed=i) for i in range(24)]
    model = GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=16, heads=2, mlp_hidden=16, out_dim=2)
    res = train_outcome_conditioned_model(model, graphs, graphs, cfg, pos_weight=1.0, monitor="auc")
    assert res.val_metrics["auc"] > 0.7
