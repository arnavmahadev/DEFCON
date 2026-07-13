"""Edge features and PyG graph assembly (tasks 2.2, 2.3, 2.4).

- 2.2 Edge features: fully-connected directed graph; each edge carries
  ``[euclidean_distance, same_team_indicator]``.
- 2.3 Graph builder: assemble a ``torch_geometric.data.Data`` with the 25-dim
  node matrix, edges, and node-level task masks/labels that batch cleanly.
- 2.4 Responsibility variant: append a 26th "action-of-interest" feature that is
  1 on a chosen option node (intended receiver or goal), 0 elsewhere.
"""

from __future__ import annotations

import numpy as np

from defcon.config import Config, load_config
from defcon.features.nodes import NODE_FEATURES, NodeGraph

__all__ = [
    "build_edges",
    "to_pyg_data",
    "add_action_of_interest",
    "X_COL",
]

# Column lookup into the node feature matrix.
X_COL = {name: i for i, name in enumerate(NODE_FEATURES)}


def build_edges(node_graph: NodeGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(edge_index, edge_attr)`` for a fully-connected directed graph.

    ``edge_index`` is (2, N*(N-1)); ``edge_attr`` is (N*(N-1), 2) with columns
    ``[distance, same_team]``. No self-loops.
    """
    N = node_graph.n_nodes
    idx = np.arange(N)
    rows = np.repeat(idx, N)
    cols = np.tile(idx, N)
    mask = rows != cols
    src, dst = rows[mask], cols[mask]

    px = node_graph.x[:, X_COL["x"]]
    py = node_graph.x[:, X_COL["y"]]
    dist = np.hypot(px[src] - px[dst], py[src] - py[dst])
    same_team = (node_graph.is_attacking[src] == node_graph.is_attacking[dst]).astype(float)

    edge_index = np.stack([src, dst]).astype(np.int64)
    edge_attr = np.stack([dist, same_team], axis=1).astype(np.float32)
    return edge_index, edge_attr


def to_pyg_data(
    node_graph: NodeGraph,
    cfg: Config | None = None,
    target_node: int | None = None,
    y: float | None = None,
    extra: dict | None = None,
):
    """Assemble a PyG ``Data`` object from a NodeGraph.

    ``target_node`` (if given) is marked in a node-level ``target_mask`` (batches
    cleanly); ``y`` is a graph-level label. ``extra`` attaches additional tensors.
    """
    import torch
    from torch_geometric.data import Data

    cfg = cfg or load_config()
    edge_index, edge_attr = build_edges(node_graph)

    data = Data(
        x=torch.from_numpy(node_graph.x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    N = node_graph.n_nodes
    data.num_nodes = N
    data.carrier_idx = torch.tensor([node_graph.carrier_idx], dtype=torch.long)
    data.teammate_mask = torch.from_numpy(node_graph.teammate_mask())
    data.defender_mask = torch.from_numpy(node_graph.defender_mask())
    data.goal_mask = torch.from_numpy(node_graph.goal_mask())
    data.node_ids = list(node_graph.node_ids)

    if target_node is not None:
        tmask = np.zeros(N, dtype=bool)
        tmask[target_node] = True
        data.target_mask = torch.from_numpy(tmask)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    if extra:
        for k, v in extra.items():
            setattr(data, k, v)
    return data


def add_action_of_interest(data, option_node: int):
    """Return a copy of ``data`` with a 26th feature = 1 on ``option_node`` (task 2.4).

    Used by the defender-responsibility model (d1), which conditions on a chosen
    option (the intended receiver or a goal node).
    """
    import torch

    new = data.clone()
    N = new.x.shape[0]
    flag = torch.zeros((N, 1), dtype=new.x.dtype)
    flag[option_node, 0] = 1.0
    new.x = torch.cat([new.x, flag], dim=1)
    return new
