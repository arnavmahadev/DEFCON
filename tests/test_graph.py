"""Tests for edge features, PyG assembly, and the responsibility variant (2.2-2.4)."""

import numpy as np
import pytest

from defcon import load_config
from defcon.features.graph import add_action_of_interest, build_edges, to_pyg_data
from defcon.features.nodes import build_node_features
from defcon.features.state import GraphState

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")


def _state(cfg):
    return GraphState(
        player_ids=["carrier", "mate", "def"],
        px=np.array([0.0, 20.0, 3.0]), py=np.array([0.0, 0.0, 0.0]),
        pvx=np.array([5.0, 0.0, 0.0]), pvy=np.array([0.0, 0.0, 0.0]),
        pspeed=np.array([5.0, 0.0, 0.0]), paccel=np.array([0.0, 0.0, 0.0]),
        is_attacking=np.array([1.0, 1.0, 0.0]), is_gk=np.array([0.0, 0.0, 0.0]),
        carrier_idx=0, ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=5.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
    )


def test_edge_count_and_attr_shape():
    cfg = load_config()
    ng = build_node_features(_state(cfg), cfg)
    ei, ea = build_edges(ng)
    N = ng.n_nodes  # 5
    assert ei.shape == (2, N * (N - 1))
    assert ea.shape == (N * (N - 1), 2)
    # No self loops.
    assert np.all(ei[0] != ei[1])


def test_edge_distance_and_same_team():
    cfg = load_config()
    ng = build_node_features(_state(cfg), cfg)
    ei, ea = build_edges(ng)
    # Find the edge carrier(0) -> mate(1): distance 20, same team.
    k = np.flatnonzero((ei[0] == 0) & (ei[1] == 1))[0]
    assert ea[k, 0] == pytest.approx(20.0)
    assert ea[k, 1] == 1.0
    # carrier(0) -> def(2): distance 3, different team.
    k2 = np.flatnonzero((ei[0] == 0) & (ei[1] == 2))[0]
    assert ea[k2, 0] == pytest.approx(3.0)
    assert ea[k2, 1] == 0.0


def test_to_pyg_data_shapes_and_masks():
    cfg = load_config()
    ng = build_node_features(_state(cfg), cfg)
    data = to_pyg_data(ng, cfg, target_node=1, y=1.0)
    assert data.x.shape == (5, 25)
    assert data.edge_index.shape == (2, 20)
    assert data.edge_attr.shape == (20, 2)
    assert bool(data.target_mask[1]) and data.target_mask.sum() == 1
    assert data.y.item() == 1.0
    assert data.teammate_mask.tolist() == [False, True, False, False, False]


def test_responsibility_26th_feature():
    cfg = load_config()
    ng = build_node_features(_state(cfg), cfg)
    data = to_pyg_data(ng, cfg)
    aug = add_action_of_interest(data, option_node=1)
    assert aug.x.shape == (5, 26)
    # Exactly one column changed, one entry set.
    assert aug.x[:, -1].sum().item() == 1.0
    assert aug.x[1, -1].item() == 1.0
    # Original 25 columns unchanged.
    assert torch.allclose(aug.x[:, :25], data.x)


def test_dataloader_batches_graphs():
    from torch_geometric.loader import DataLoader

    cfg = load_config()
    graphs = []
    for _ in range(4):
        ng = build_node_features(_state(cfg), cfg)
        graphs.append(to_pyg_data(ng, cfg, target_node=1, y=1.0))
    loader = DataLoader(graphs, batch_size=2)
    batch = next(iter(loader))
    assert batch.x.shape == (10, 25)  # 2 graphs x 5 nodes
    assert batch.num_graphs == 2
    assert batch.target_mask.sum().item() == 2  # one target per graph
