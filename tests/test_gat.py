"""Tests for the GAT backbone / heads and the training harness (3.0, 3.7)."""

import numpy as np
import pytest

from defcon import load_config
from defcon.features.graph import to_pyg_data
from defcon.features.nodes import build_node_features
from defcon.features.state import GraphState

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from defcon.models.gat import GATGraphModel, GATNodeModel  # noqa: E402
from defcon.models.train import binary_metrics, train_binary_node_model  # noqa: E402


def _make_graph(cfg, completed=True, seed=0):
    """A separable toy graph: when the pass fails a defender sits on the
    carrier->receiver line; when it completes the defenders are far away. This
    gives the GAT a real signal (corridor/passline features) to learn."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-1.0, 1.0, 2)
    # carrier at origin, receiver (node 1) ~20 m ahead.
    recv = np.array([20.0 + jitter[0], 0.0 + jitter[1]])
    if completed:
        d1, d2 = np.array([10.0, 25.0]), np.array([30.0, -20.0])  # far from the lane
    else:
        d1, d2 = np.array([10.0, 0.0]), np.array([14.0, 1.0])     # blocking the lane
    px = np.array([0.0, recv[0], 40.0, d1[0], d2[0], -30.0])
    py = np.array([0.0, recv[1], 5.0, d1[1], d2[1], 0.0])
    state = GraphState(
        player_ids=[f"p{i}" for i in range(6)],
        px=px, py=py,
        pvx=np.zeros(6), pvy=np.zeros(6), pspeed=np.zeros(6), paccel=np.zeros(6),
        is_attacking=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        is_gk=np.array([0.0, 0, 0, 0, 0, 1.0]),
        carrier_idx=0, ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=1.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
    )
    ng = build_node_features(state, cfg)
    return to_pyg_data(ng, cfg, target_node=1, y=1.0 if completed else 0.0)


def test_node_model_forward_shape():
    cfg = load_config()
    model = GATNodeModel(in_dim=cfg.features.node_dim)
    data = _make_graph(cfg)
    out = model(data)
    assert out.shape == (data.x.shape[0],)  # one logit per node


def test_graph_model_forward_scalar():
    cfg = load_config()
    model = GATGraphModel(in_dim=cfg.features.node_dim)
    data = _make_graph(cfg)
    out = model(data)
    assert out.shape == () or out.shape == (1,)  # graph-level logit


def test_harness_can_overfit_tiny_dataset():
    """Sanity: the trainer should push train==val AUC toward 1 on separable data."""
    cfg = load_config(overrides={"training": {"max_epochs": 60, "patience": 60, "batch_size": 8}})
    # Class-separable by construction: completed vs not, distinct seeds.
    pos = [_make_graph(cfg, completed=True, seed=s) for s in range(8)]
    neg = [_make_graph(cfg, completed=False, seed=s + 100) for s in range(8)]
    graphs = pos + neg
    model = GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=16, heads=2, mlp_hidden=16)
    result = train_binary_node_model(model, graphs, graphs, cfg, monitor="auc", verbose=False)
    assert result.best_metric > 0.8  # can fit the training signal


def test_binary_metrics_single_class():
    m = binary_metrics(np.ones(5), np.array([0.9, 0.8, 0.7, 0.6, 0.95]))
    assert np.isnan(m["auc"])  # AUC undefined for one class
    assert m["base_rate"] == 1.0
