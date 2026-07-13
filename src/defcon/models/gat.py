"""GAT backbone and component-model heads (task 3.0 + heads for 3.1-3.4).

Eq 18: ``H = GAT_theta2(GAT_theta1(X, E))`` — two GAT layers passing the 2-dim
edge features, producing node embeddings ``H`` that task-specific heads sit on
top of. One reusable backbone, several thin heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv

__all__ = ["GATBackbone", "GATNodeModel", "GATGraphModel", "make_backbone", "GNNNodeModel"]


class GATBackbone(nn.Module):
    """Two-layer GAT over the player/goal graph with edge features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 2,
    ):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        # Second layer averages heads (concat=False) -> output dim = hidden_dim.
        self.gat2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim, dropout=dropout
        )
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index, edge_attr)
        h = F.elu(h)
        return h


class GCNBackbone(nn.Module):
    """Two GCN layers (baseline). Edge distance becomes a positive edge weight."""

    def __init__(self, in_dim, hidden_dim=64, dropout=0.2, **_):
        super().__init__()
        self.dropout = dropout
        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        # closer nodes -> larger weight
        w = 1.0 / (1.0 + edge_attr[:, 0]) if edge_attr is not None else None
        h = F.elu(self.c1(x, edge_index, w))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.c2(h, edge_index, w))


class GINBackbone(nn.Module):
    """Two GIN layers (baseline). Ignores edge features (GIN is edge-agnostic)."""

    def __init__(self, in_dim, hidden_dim=64, dropout=0.2, **_):
        super().__init__()
        self.dropout = dropout

        def mlp(i):
            return nn.Sequential(nn.Linear(i, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.c1 = GINConv(mlp(in_dim))
        self.c2 = GINConv(mlp(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        h = F.elu(self.c1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.c2(h, edge_index))


def make_backbone(kind: str, in_dim: int, hidden_dim=64, heads=4, dropout=0.2, edge_dim=2):
    """Build a GAT / GCN / GIN backbone by name (for baseline comparison, task 7.2)."""
    kind = kind.lower()
    if kind == "gat":
        return GATBackbone(in_dim, hidden_dim, heads, dropout, edge_dim)
    if kind == "gcn":
        return GCNBackbone(in_dim, hidden_dim, dropout)
    if kind == "gin":
        return GINBackbone(in_dim, hidden_dim, dropout)
    raise ValueError(f"Unknown backbone kind {kind!r} (gat|gcn|gin)")


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        return self.net(h)


class GATNodeModel(nn.Module):
    """GAT backbone + node-wise MLP head producing per-node logits.

    Used for pass-success (b1), action-selection (a1), goal-scoring/conceding
    (c1/c2), and defender-responsibility (d1). ``out_dim`` controls how many
    logits per node (e.g. 2 for outcome-conditioned c1/c2).
    """

    def __init__(self, in_dim: int, hidden_dim=64, heads=4, dropout=0.2, mlp_hidden=64,
                 edge_dim=2, out_dim=1):
        super().__init__()
        self.backbone = GATBackbone(in_dim, hidden_dim, heads, dropout, edge_dim)
        self.head = _MLPHead(self.backbone.out_dim, mlp_hidden, out_dim, dropout)
        self.out_dim = out_dim

    def forward(self, data):
        h = self.backbone(data.x, data.edge_index, data.edge_attr)
        logits = self.head(h)  # (num_nodes, out_dim)
        if self.out_dim == 1:
            logits = logits.squeeze(-1)  # (num_nodes,)
        return logits


class GNNNodeModel(nn.Module):
    """Backbone-agnostic node model (GAT / GCN / GIN) + MLP head — for task 7.2."""

    def __init__(self, kind: str, in_dim: int, hidden_dim=64, heads=4, dropout=0.2,
                 mlp_hidden=64, edge_dim=2, out_dim=1):
        super().__init__()
        self.backbone = make_backbone(kind, in_dim, hidden_dim, heads, dropout, edge_dim)
        self.head = _MLPHead(self.backbone.out_dim, mlp_hidden, out_dim, dropout)
        self.out_dim = out_dim

    def forward(self, data):
        h = self.backbone(data.x, data.edge_index, data.edge_attr)
        logits = self.head(h)
        return logits.squeeze(-1) if self.out_dim == 1 else logits


class GATGraphModel(nn.Module):
    """GAT backbone + mean-pool + graph-level MLP (Eq 22, shot-blocking b2)."""

    def __init__(self, in_dim: int, hidden_dim=64, heads=4, dropout=0.2, mlp_hidden=64,
                 edge_dim=2, out_dim=1):
        super().__init__()
        self.backbone = GATBackbone(in_dim, hidden_dim, heads, dropout, edge_dim)
        self.head = _MLPHead(self.backbone.out_dim, mlp_hidden, out_dim, dropout)
        self.out_dim = out_dim

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        h = self.backbone(data.x, data.edge_index, data.edge_attr)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch)  # (num_graphs, hidden)
        logits = self.head(pooled)
        if self.out_dim == 1:
            logits = logits.squeeze(-1)
        return logits
