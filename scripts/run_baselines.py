#!/usr/bin/env python
"""Baseline comparison for Table 3 / task 7.2.

b1 pass-success across GAT / GCN / GIN / XGBoost / CatBoost (split by match), and
a1 action-selection across the three GNN backbones (boosting can't select a node).

Usage: python scripts/run_baselines.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from defcon import load_config
from defcon.data.pipeline import process_metrica_match
from defcon.eval.baselines import graphs_to_tabular
from defcon.features.dataset import build_action_selection_graphs, build_pass_success_graphs
from defcon.models.gat import GNNNodeModel
from defcon.models.train import set_seed, train_binary_node_model, train_selection_model

warnings.filterwarnings("ignore")


def boosting_isolated(Xtr, ytr, Xva, yva, kind, seed):
    """Train a boosting model in a torch-free subprocess (dodges the OpenMP crash)."""
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "data.npz"
        out = Path(d) / "m.json"
        np.savez(npz, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, seed=seed)
        worker = Path(__file__).parent / "_boost_worker.py"
        subprocess.run([sys.executable, str(worker), str(npz), kind, str(out)], check=True)
        return json.loads(out.read_text())


def main() -> None:
    cfg = load_config()
    set_seed(cfg.training.seed)
    gd = cfg.path("tracking_dir") / "metrica"
    a1_tr_actions, trk1 = process_metrica_match(gd, 1, cfg)
    a2_va_actions, trk2 = process_metrica_match(gd, 2, cfg)

    results = {"b1": {}, "a1": {}}

    # ---------- b1: pass success ----------
    print("=== b1 pass-success (AUC / Brier) — split G1→G2 ===")
    b1_tr = build_pass_success_graphs(a1_tr_actions, trk1, cfg)
    b1_va = build_pass_success_graphs(a2_va_actions, trk2, cfg)
    for kind in ("gat", "gcn", "gin"):
        set_seed(cfg.training.seed)
        model = GNNNodeModel(kind, in_dim=cfg.features.node_dim, hidden_dim=cfg.model.hidden_dim,
                             heads=cfg.model.heads, dropout=cfg.model.dropout,
                             mlp_hidden=cfg.model.mlp_hidden, out_dim=1)
        res = train_binary_node_model(model, b1_tr, b1_va, cfg, monitor="auc", verbose=False)
        results["b1"][kind.upper()] = res.val_metrics
        print(f"  {kind.upper():9s} AUC={res.val_metrics['auc']:.3f}  Brier={res.val_metrics['brier']:.3f}  F1={res.val_metrics['f1']:.3f}")

    Xtr, ytr = graphs_to_tabular(b1_tr)
    Xva, yva = graphs_to_tabular(b1_va)
    for kind in ("xgb", "cat"):
        m = boosting_isolated(Xtr, ytr, Xva, yva, kind, cfg.training.seed)
        results["b1"][kind.upper()] = m
        print(f"  {kind.upper():9s} AUC={m['auc']:.3f}  Brier={m['brier']:.3f}  F1={m['f1']:.3f}")

    # ---------- a1: action selection (GNN only) ----------
    print("\n=== a1 action-selection (MRR / Acc) — GNN backbones only ===")
    a1_tr = build_action_selection_graphs(a1_tr_actions, trk1, cfg)
    a1_va = build_action_selection_graphs(a2_va_actions, trk2, cfg)
    for kind in ("gat", "gcn", "gin"):
        set_seed(cfg.training.seed)
        model = GNNNodeModel(kind, in_dim=cfg.features.node_dim, hidden_dim=cfg.model.hidden_dim,
                             heads=cfg.model.heads, dropout=cfg.model.dropout,
                             mlp_hidden=cfg.model.mlp_hidden, out_dim=1)
        res = train_selection_model(model, a1_tr, a1_va, cfg, monitor="mrr", verbose=False)
        results["a1"][kind.upper()] = res.val_metrics
        print(f"  {kind.upper():9s} MRR={res.val_metrics['mrr']:.3f}  Acc={res.val_metrics['accuracy']:.3f}")
    print("  XGB/CAT   n/a (boosting cannot rank a variable candidate set)")

    out = cfg.path("checkpoints") / "baselines.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] saved -> {out}")


if __name__ == "__main__":
    main()
