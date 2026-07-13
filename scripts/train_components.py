#!/usr/bin/env python
"""Train the remaining GAT component models (tasks 3.1, 3.3, 3.4).

  a1  action-selection      (softmax over teammates + goal)   -> Acc / CE / MRR
  c1  goal-scoring          (outcome-conditioned, per option) -> AUC / Brier
  c2  goal-conceding        (outcome-conditioned, per option) -> AUC / Brier
  d1  defender-responsibility (26-feat, softmax over defenders) -> Acc / CE / MRR

Split by match: train on Game 1, validate on Game 2.

Usage: python scripts/train_components.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from defcon import load_config
from defcon.data.pipeline import process_metrica_match
from defcon.features.dataset import (
    build_action_selection_graphs,
    build_goal_condition_graphs,
    build_responsibility_graphs,
)
from defcon.features.graph import X_COL
from defcon.models.gat import GATNodeModel
from defcon.models.train import (
    binary_metrics,
    selection_metrics,
    set_seed,
    train_outcome_conditioned_model,
    train_selection_model,
)


def _proximity_baseline(graphs, ref="carrier"):
    """Rank candidates by distance to a reference node; report Acc/MRR/CE-ish.

    ref='carrier' for a1; ref='option' (the 26th-feature node) for d1.
    """
    ranks = []
    for g in graphs:
        x = g.x[:, X_COL["x"]].numpy()
        y = g.x[:, X_COL["y"]].numpy()
        if ref == "carrier":
            ri = int(g.carrier_idx.item())
        else:
            ri = int(np.flatnonzero(g.x[:, -1].numpy() == 1.0)[0])
        cidx = np.flatnonzero(g.candidate_mask.numpy())
        tidx = int(np.flatnonzero(g.select_target.numpy())[0])
        d = np.hypot(x[cidx] - x[ri], y[cidx] - y[ri])
        order = cidx[np.argsort(d)]
        rank = int(np.flatnonzero(order == tidx)[0]) + 1
        ranks.append(rank)
    return selection_metrics(ranks, 0.0)


def main() -> None:
    cfg = load_config()
    set_seed(cfg.training.seed)
    game_dir = cfg.path("tracking_dir") / "metrica"

    print("[setup] processing games (force rebuild to add interceptor labels)...")
    a1, trk1 = process_metrica_match(game_dir, 1, cfg, force=True)
    a2, trk2 = process_metrica_match(game_dir, 2, cfg, force=True)

    results = {}

    # ---------------- a1: action selection ----------------
    print("\n=== a1 action-selection ===")
    t0 = time.time()
    tr = build_action_selection_graphs(a1, trk1, cfg)
    va = build_action_selection_graphs(a2, trk2, cfg)
    print(f"[a1] graphs: train {len(tr)} / val {len(va)} ({time.time()-t0:.1f}s)")
    model = GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=cfg.model.hidden_dim,
                         heads=cfg.model.heads, dropout=cfg.model.dropout,
                         mlp_hidden=cfg.model.mlp_hidden, out_dim=1)
    res = train_selection_model(model, tr, va, cfg, monitor="mrr")
    base = _proximity_baseline(va, ref="carrier")
    results["a1"] = {"gat": res.val_metrics, "baseline_nearest": base}
    print(f"[a1] GAT  Acc={res.val_metrics['accuracy']:.3f} CE={res.val_metrics['ce']:.3f} "
          f"MRR={res.val_metrics['mrr']:.3f}  (paper 0.674/0.924/0.800)")
    print(f"[a1] base Acc={base['accuracy']:.3f} MRR={base['mrr']:.3f} (nearest teammate)")
    torch.save(model.state_dict(), cfg.path("checkpoints") / "a1_action_selection.pt")

    # ---------------- c1 / c2: goal scoring / conceding ----------------
    for tag, col, paper in [("c1", "scores_next", "AUC~0.68"), ("c2", "concedes_next", "AUC~0.62")]:
        print(f"\n=== {tag} {col} ===")
        tr = build_goal_condition_graphs(a1, trk1, col, cfg)
        va = build_goal_condition_graphs(a2, trk2, col, cfg)
        pos = sum(g.y.item() for g in tr)
        pw = float((len(tr) - pos) / max(pos, 1))
        print(f"[{tag}] graphs: train {len(tr)} (pos {int(pos)}) / val {len(va)} | pos_weight {pw:.0f}")
        model = GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=cfg.model.hidden_dim,
                             heads=cfg.model.heads, dropout=cfg.model.dropout,
                             mlp_hidden=cfg.model.mlp_hidden, out_dim=2)
        res = train_outcome_conditioned_model(model, tr, va, cfg, pos_weight=pw, monitor="auc")
        results[tag] = res.val_metrics
        m = res.val_metrics
        print(f"[{tag}] GAT  AUC={m['auc']:.3f} Brier={m['brier']:.3f} F1={m['f1']:.3f} "
              f"(base rate {m['base_rate']:.3f}; paper {paper})")
        torch.save(model.state_dict(), cfg.path("checkpoints") / f"{tag}_goal_model.pt")

    # ---------------- d1: defender responsibility ----------------
    print("\n=== d1 defender-responsibility ===")
    tr = build_responsibility_graphs(a1, trk1, cfg)
    va = build_responsibility_graphs(a2, trk2, cfg)
    print(f"[d1] graphs: train {len(tr)} / val {len(va)} (failed passes w/ known interceptor)")
    rdim = cfg.features.node_dim + cfg.features.responsibility_extra_dim  # 26
    model = GATNodeModel(in_dim=rdim, hidden_dim=cfg.model.hidden_dim, heads=cfg.model.heads,
                         dropout=cfg.model.dropout, mlp_hidden=cfg.model.mlp_hidden, out_dim=1)
    res = train_selection_model(model, tr, va, cfg, monitor="mrr")
    base = _proximity_baseline(va, ref="option")
    results["d1"] = {"gat": res.val_metrics, "baseline_nearest": base}
    print(f"[d1] GAT  Acc={res.val_metrics['accuracy']:.3f} CE={res.val_metrics['ce']:.3f} "
          f"MRR={res.val_metrics['mrr']:.3f}  (paper 0.502/1.404/0.694)")
    print(f"[d1] base Acc={base['accuracy']:.3f} MRR={base['mrr']:.3f} (nearest defender)")
    torch.save(model.state_dict(), cfg.path("checkpoints") / "d1_responsibility.pt")

    out = cfg.path("checkpoints") / "components_metrics.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] saved metrics -> {out}")


if __name__ == "__main__":
    main()
