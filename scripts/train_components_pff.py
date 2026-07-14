#!/usr/bin/env python
"""Retrain the data-starved component models on the PFF World Cup pilot.

On 2-3 Metrica matches three models were not reproducible:
  c2  goal-conceding   -> 0 conceding-labelled passes in a single match
  d1  responsibility   -> only ~146 failed-pass examples
  b2  shot-blocking     -> never built (deferred), too few shots

The PFF FC 2022 pilot (7 knockout games) carries far more of these rare events:
~7.9k passes, ~1k failed passes with a known interceptor, 199 shots (46 blocked),
71 conceding-labelled passes. Each PFF event embeds a freeze-frame, so we build the
same 25/26-dim graphs used on Metrica — velocities are approximated as 0 (PFF gives
speed magnitude only), a known cross-provider limitation we report honestly.

Split is by match (never within a match): train on 5 games, validate on 2.

Usage: python scripts/train_components_pff.py
"""

from __future__ import annotations

import json
import time
import warnings

import torch

from defcon import load_config
from defcon.data.labels import add_goal_labels
from defcon.data.pff import load_pff_metadata, pff_game_paths
from defcon.data.pff_events import parse_pff_events
from defcon.features.dataset import (
    build_goal_condition_graphs,
    build_responsibility_graphs,
    build_shot_blocking_graphs,
)
from defcon.models.gat import GATGraphModel, GATNodeModel
from defcon.models.train import (
    selection_metrics,
    set_seed,
    train_binary_graph_model,
    train_outcome_conditioned_model,
    train_selection_model,
)
from defcon.models.uxg import UxGModel

warnings.filterwarnings("ignore")

TRAIN_GAMES = [10504, 10511, 10513, 10514, 10515]
VAL_GAMES = [10503, 10517]


def load_game(gid: int, cfg):
    """Parse one PFF game into (actions with goal labels, tracking)."""
    pff_dir = cfg.path("tracking_dir") / "pff"
    ev_dir = cfg.path("data_raw") / "pff_events"
    md = load_pff_metadata(pff_game_paths(pff_dir, gid)["metadata"])
    events = json.load(open(ev_dir / f"{gid}.json"))
    actions, tracking = parse_pff_events(events, md, f"pff_{gid}")
    actions = add_goal_labels(actions, cfg)
    return actions, tracking


def _nearest_defender_baseline(graphs):
    """d1 baseline: rank defenders by distance to the option (26th-feature) node."""
    import numpy as np

    from defcon.features.graph import X_COL

    ranks = []
    for g in graphs:
        x = g.x[:, X_COL["x"]].numpy()
        y = g.x[:, X_COL["y"]].numpy()
        ri = int(np.flatnonzero(g.x[:, -1].numpy() == 1.0)[0])
        cidx = np.flatnonzero(g.candidate_mask.numpy())
        tidx = int(np.flatnonzero(g.select_target.numpy())[0])
        d = np.hypot(x[cidx] - x[ri], y[cidx] - y[ri])
        order = cidx[np.argsort(d)]
        ranks.append(int(np.flatnonzero(order == tidx)[0]) + 1)
    return selection_metrics(ranks, 0.0)


def main() -> int:
    cfg = load_config()
    set_seed(cfg.training.seed)

    print(f"[setup] loading PFF pilot — train {TRAIN_GAMES}  val {VAL_GAMES}")
    t0 = time.time()
    train_games = {gid: load_game(gid, cfg) for gid in TRAIN_GAMES}
    val_games = {gid: load_game(gid, cfg) for gid in VAL_GAMES}
    print(f"[setup] parsed 7 games in {time.time() - t0:.1f}s")

    def build(builder, *args):
        tr = [g for gid in TRAIN_GAMES for g in builder(*train_games[gid], *args)]
        va = [g for gid in VAL_GAMES for g in builder(*val_games[gid], *args)]
        return tr, va

    results = {}
    md = cfg.model
    node_dim = cfg.features.node_dim

    # ---------------- c2: goal-conceding ----------------
    print("\n=== c2 goal-conceding (outcome-conditioned) ===")
    tr, va = build(build_goal_condition_graphs, "concedes_next")
    pos = int(sum(g.y.item() for g in tr))
    pw = float((len(tr) - pos) / max(pos, 1))
    print(f"[c2] graphs: train {len(tr)} (pos {pos}) / val {len(va)} "
          f"(pos {int(sum(g.y.item() for g in va))}) | pos_weight {pw:.0f}")
    model = GATNodeModel(in_dim=node_dim, hidden_dim=md.hidden_dim, heads=md.heads,
                         dropout=md.dropout, mlp_hidden=md.mlp_hidden, out_dim=2)
    res = train_outcome_conditioned_model(model, tr, va, cfg, pos_weight=pw, monitor="auc")
    results["c2"] = res.val_metrics
    m = res.val_metrics
    print(f"[c2] GAT AUC={m['auc']:.3f} Brier={m['brier']:.3f} F1={m['f1']:.3f} "
          f"(base rate {m['base_rate']:.3f}; paper AUC~0.62)")
    torch.save(model.state_dict(), cfg.path("checkpoints") / "c2_goal_model_pff.pt")

    # ---------------- d1: defender responsibility ----------------
    print("\n=== d1 defender-responsibility ===")
    tr, va = build(build_responsibility_graphs)
    print(f"[d1] graphs: train {len(tr)} / val {len(va)} (failed passes w/ known interceptor)")
    rdim = node_dim + cfg.features.responsibility_extra_dim  # 26
    model = GATNodeModel(in_dim=rdim, hidden_dim=md.hidden_dim, heads=md.heads,
                         dropout=md.dropout, mlp_hidden=md.mlp_hidden, out_dim=1)
    res = train_selection_model(model, tr, va, cfg, monitor="mrr")
    base = _nearest_defender_baseline(va)
    results["d1"] = {"gat": res.val_metrics, "baseline_nearest": base}
    v = res.val_metrics
    print(f"[d1] GAT  Acc={v['accuracy']:.3f} CE={v['ce']:.3f} MRR={v['mrr']:.3f}  "
          f"(paper 0.502/1.404/0.694)")
    print(f"[d1] base Acc={base['accuracy']:.3f} MRR={base['mrr']:.3f} (nearest defender)")
    torch.save(model.state_dict(), cfg.path("checkpoints") / "d1_responsibility_pff.pt")

    # ---------------- b2: shot-blocking ----------------
    print("\n=== b2 shot-blocking (graph-level, Eq 22) ===")
    uxg = UxGModel.load(cfg.path("checkpoints") / "uxg.joblib")
    # Real shots only, then with proxy augmentation, to show augmentation lifts recall.
    tr_real, va = build(build_shot_blocking_graphs, cfg, None, False)
    tr_aug, _ = build(build_shot_blocking_graphs, cfg, uxg, True)
    n_pos_real = int(sum(g.y.item() for g in tr_real))
    n_pos_aug = int(sum(g.y.item() for g in tr_aug))
    n_pos_val = int(sum(g.y.item() for g in va))
    print(f"[b2] real train {len(tr_real)} (blocked {n_pos_real}) | "
          f"augmented train {len(tr_aug)} (blocked {n_pos_aug}) | val {len(va)} (blocked {n_pos_val})")

    def fit_b2(train_graphs, tag):
        pos = int(sum(g.y.item() for g in train_graphs))
        pw = float((len(train_graphs) - pos) / max(pos, 1))
        mdl = GATGraphModel(in_dim=node_dim, hidden_dim=md.hidden_dim, heads=md.heads,
                            dropout=md.dropout, mlp_hidden=md.mlp_hidden, out_dim=1)
        r = train_binary_graph_model(mdl, train_graphs, va, cfg, pos_weight=pw,
                                     monitor="auc", verbose=False)
        mm = r.val_metrics
        print(f"[b2:{tag}] AUC={mm['auc']:.3f} F1={mm['f1']:.3f} Brier={mm['brier']:.3f} "
              f"(paper GAT 0.398/0.672/0.204)")
        return mdl, mm

    model, m_real = fit_b2(tr_real, "real")
    _, m_aug = fit_b2(tr_aug, "augmented")
    results["b2"] = {"real": m_real, "augmented": m_aug}
    # Persist the real-shot model: on the pilot the proxy augmentation hurt AUC,
    # so the honest deliverable is the model trained on real blocked shots.
    torch.save(model.state_dict(), cfg.path("checkpoints") / "b2_shot_blocking_pff.pt")

    out = cfg.path("checkpoints") / "components_metrics_pff.json"
    out.write_text(json.dumps(
        {"train_games": TRAIN_GAMES, "val_games": VAL_GAMES, **results}, indent=2, default=float))
    print(f"\n[done] saved metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
