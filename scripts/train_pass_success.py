#!/usr/bin/env python
"""Train the pass-success GAT (b1) on Metrica (task 3.2).

Splits by match: train on Game 1, validate on Game 2 (no within-match leakage).

Usage:
    python scripts/train_pass_success.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from defcon import load_config
from defcon.data.pipeline import process_metrica_match
from defcon.features.dataset import build_pass_success_graphs
from defcon.models.gat import GATNodeModel
from defcon.models.train import binary_metrics, set_seed, train_binary_node_model


def _graphs_for_game(cfg, game_id):
    game_dir = cfg.path("tracking_dir") / "metrica"
    actions, tracking = process_metrica_match(game_dir, game_id, cfg)
    t0 = time.time()
    graphs = build_pass_success_graphs(actions, tracking, cfg)
    print(f"[b1] game {game_id}: {len(graphs)} pass graphs built in {time.time()-t0:.1f}s")
    return graphs


def nearest_teammate_baseline(val_graphs) -> dict:
    """Baseline: predict success = 1 - (defenders in the passing corridor > 0).

    A crude positional prior to check the GAT actually learns something.
    """
    from defcon.features.graph import X_COL

    probs, ys = [], []
    for g in val_graphs:
        tgt = int(np.flatnonzero(g.target_mask.numpy())[0])
        n_corridor = g.x[tgt, X_COL["n_opp_in_corridor"]].item()
        probs.append(1.0 / (1.0 + n_corridor))  # fewer blockers -> more likely complete
        ys.append(g.y.item())
    return binary_metrics(np.array(ys), np.array(probs))


def main() -> None:
    cfg = load_config()
    set_seed(cfg.training.seed)

    train_graphs = _graphs_for_game(cfg, 1)
    val_graphs = _graphs_for_game(cfg, 2)
    tr_pos = np.mean([g.y.item() for g in train_graphs])
    va_pos = np.mean([g.y.item() for g in val_graphs])
    print(f"[b1] train completion {tr_pos:.3f} | val completion {va_pos:.3f}")

    model = GATNodeModel(
        in_dim=cfg.features.node_dim,
        hidden_dim=cfg.model.hidden_dim,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
        mlp_hidden=cfg.model.mlp_hidden,
        edge_dim=cfg.features.edge_dim,
        out_dim=1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[b1] GATNodeModel params: {n_params:,}")

    print("[b1] training (monitor val AUC)...")
    result = train_binary_node_model(model, train_graphs, val_graphs, cfg, monitor="auc")

    print("\n[b1] baseline (inverse corridor-blockers):")
    base = nearest_teammate_baseline(val_graphs)
    print(f"  F1={base['f1']:.3f}  AUC={base['auc']:.3f}  Brier={base['brier']:.3f}")

    print("\n[b1] GAT best val metrics (paper GAT: F1 0.912 / AUC 0.917 / Brier 0.093):")
    m = result.val_metrics
    print(f"  F1={m['f1']:.3f}  AUC={m['auc']:.3f}  Brier={m['brier']:.3f}  "
          f"(epoch {result.best_epoch}, base rate {m['base_rate']:.3f})")

    out = cfg.path("checkpoints") / "pass_success_gat.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(model.state_dict(), out)
    (out.with_suffix(".metrics.json")).write_text(json.dumps(
        {"gat": m, "baseline": base, "best_epoch": result.best_epoch}, indent=2))
    print(f"\n[b1] saved model -> {out}")


if __name__ == "__main__":
    main()
