"""EPV assembly — combine the component models into expected possession value.

Tasks 4.1 (per-option EPV, Eqs 16-17) and 4.2 (state EPV, Eqs 1-2).

For a game state ``s`` the expected possession value is::

    EPV(s) = Σ_a  P(select a | s) · Σ_o P(o | s, a) · EPV(a, o)                (Eqs 1-2)

Options ``a`` are: pass to each teammate ``v``, or shoot (the goal option).

    pass:  EPV(a, o) = P(score | a, o) − P(concede | a, o)                    (Eq 16)
           option value = P(success)·EPV(a, succ) + P(fail)·EPV(a, fail)
    shot:  EPV = UxG on an unblocked shot, 0 if blocked                        (Eq 17)
           (shot-blocking b2 is deferred → treated as unblocked, value = UxG)

The dribble / self-pass option is not separately modeled (a1 ranks teammates +
goal), a documented simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from defcon.config import Config, load_config
from defcon.features.graph import X_COL, to_pyg_data
from defcon.features.nodes import build_node_features
from defcon.features.state import GraphState, graph_state_from_action

__all__ = ["EPVEngine", "combine_state_epv"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the masked positions; 0 elsewhere. Returns a full-length vector."""
    out = np.zeros_like(logits, dtype=float)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    z = logits[idx]
    z = z - z.max()
    e = np.exp(z)
    out[idx] = e / e.sum()
    return out


def combine_state_epv(option_table: pd.DataFrame) -> float:
    """State EPV = Σ_options P(select)·option_value (Eqs 1-2)."""
    return float((option_table["p_select"] * option_table["option_value"]).sum())


@dataclass
class EPVEngine:
    """Assembles trained component models into EPV for a game state."""

    a1: object   # action-selection  (GATNodeModel out_dim=1)
    b1: object   # pass-success      (GATNodeModel out_dim=1)
    c1: object   # goal-scoring      (GATNodeModel out_dim=2)
    c2: object   # goal-conceding    (GATNodeModel out_dim=2)
    uxg: object  # UxGModel
    cfg: Config = None

    def __post_init__(self):
        self.cfg = self.cfg or load_config()

    # -- loading --------------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, cfg: Config | None = None) -> "EPVEngine":
        import torch

        from defcon.models.gat import GATNodeModel
        from defcon.models.uxg import UxGModel

        cfg = cfg or load_config()
        ckpt = cfg.path("checkpoints")

        def load(name, out_dim):
            m = GATNodeModel(
                in_dim=cfg.features.node_dim, hidden_dim=cfg.model.hidden_dim,
                heads=cfg.model.heads, dropout=cfg.model.dropout,
                mlp_hidden=cfg.model.mlp_hidden, out_dim=out_dim,
            )
            m.load_state_dict(torch.load(ckpt / name, map_location="cpu"))
            m.eval()
            return m

        uxg = UxGModel.load(ckpt / "uxg.joblib", cfg)
        return cls(
            a1=load("a1_action_selection.pt", 1),
            b1=load("pass_success_gat.pt", 1),
            c1=load("c1_goal_model.pt", 2),
            c2=load("c2_goal_model.pt", 2),
            uxg=uxg,
            cfg=cfg,
        )

    # -- inference ------------------------------------------------------------
    def _logits(self, model, data) -> np.ndarray:
        import torch

        with torch.no_grad():
            return model(data).cpu().numpy()

    def option_table(self, state: GraphState) -> pd.DataFrame:
        """Figure-2-style per-option table for one state (task 4.1)."""
        cfg = self.cfg
        ng = build_node_features(state, cfg)
        data = to_pyg_data(ng, cfg)

        candidate = ng.teammate_mask() | ng.goal_mask()
        p_select = _masked_softmax(self._logits(self.a1, data), candidate)
        p_success = _sigmoid(self._logits(self.b1, data))          # (N,)
        p_score = _sigmoid(self._logits(self.c1, data))            # (N,2): [fail, succ]
        p_concede = _sigmoid(self._logits(self.c2, data))          # (N,2)

        rows = []
        for v in np.flatnonzero(ng.teammate_mask()):
            epv_succ = float(p_score[v, 1] - p_concede[v, 1])
            epv_fail = float(p_score[v, 0] - p_concede[v, 0])
            ps = float(p_success[v])
            rows.append({
                "target": ng.node_ids[v], "type": "pass",
                "p_select": float(p_select[v]), "p_success": ps,
                "epv_success": epv_succ, "epv_fail": epv_fail,
                "option_value": ps * epv_succ + (1 - ps) * epv_fail,
            })

        # Shot option: UxG at the carrier location (goal nodes' select mass summed).
        gx = data.x[:, X_COL["x"]].numpy()
        gy = data.x[:, X_COL["y"]].numpy()
        cx, cy = float(gx[state.carrier_idx]), float(gy[state.carrier_idx])
        # centered (+x attack) -> UxG frame (goal at x=length, y in [0,width]).
        x_uxg = cx + cfg.pitch.length / 2.0
        y_uxg = cy + cfg.pitch.width / 2.0
        uxg = float(self.uxg.score_location(x_uxg, y_uxg))
        p_select_goal = float(p_select[np.flatnonzero(ng.goal_mask())].sum())
        rows.append({
            "target": "goal", "type": "shot",
            "p_select": p_select_goal, "p_success": 1.0,  # unblocked (b2 deferred)
            "epv_success": uxg, "epv_fail": 0.0, "option_value": uxg,
        })
        return pd.DataFrame(rows)

    def state_epv(self, state: GraphState) -> float:
        """Scalar EPV for a state (task 4.2)."""
        return combine_state_epv(self.option_table(state))

    def state_epv_from_action(self, tracking, action, directions, goalkeepers) -> float | None:
        state = graph_state_from_action(tracking, action, directions, goalkeepers, self.cfg)
        if state is None:
            return None
        return self.state_epv(state)
