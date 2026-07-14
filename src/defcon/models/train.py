"""Training harness for the GAT component models (task 3.7).

One loop parametrized by task via the loss/metric. Early stopping on a validation
metric, best-checkpoint restore, and a metrics dict that reproduces a Table-3 row.
Splitting is the caller's responsibility (split by match/season — never within a
match) to avoid leakage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from defcon.config import Config, load_config

__all__ = [
    "set_seed",
    "train_binary_node_model",
    "train_binary_graph_model",
    "train_selection_model",
    "train_outcome_conditioned_model",
    "binary_metrics",
    "TrainResult",
]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """F1 / AUC / Brier (+ base rate). Robust to a single-class validation set."""
    from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    out["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    val_metrics: dict
    history: list = field(default_factory=list)


def _target_logits(model, batch):
    """Per-graph logit at each graph's target node (order matches batch.y)."""
    logits = model(batch)  # (num_nodes,)
    return logits[batch.target_mask]


def train_binary_node_model(
    model,
    train_graphs: list,
    val_graphs: list,
    cfg: Config | None = None,
    pos_weight: float | None = None,
    monitor: str = "auc",
    verbose: bool = True,
) -> TrainResult:
    """Train a per-node binary model (b1/c1/c2). Returns best val metrics.

    The model's per-node logits are gathered at each graph's ``target_mask`` node
    and trained with ``BCEWithLogitsLoss`` against the graph label ``y``.
    """
    import torch
    from torch_geometric.loader import DataLoader

    cfg = cfg or load_config()
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.training.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    higher_is_better = monitor in ("auc", "f1")
    best_score = -np.inf if higher_is_better else np.inf
    best_state = None
    best_epoch = -1
    best_metrics: dict = {}
    history = []
    patience = cfg.training.patience
    bad = 0

    for epoch in range(cfg.training.max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = _target_logits(model, batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()

        # ---- validation ----
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = _target_logits(model, batch)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        metrics = binary_metrics(ys, probs)
        score = metrics.get(monitor, float("nan"))
        history.append({"epoch": epoch, **metrics})

        improved = (score > best_score) if higher_is_better else (score < best_score)
        if not np.isnan(score) and improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | val {monitor}={score:.3f} f1={metrics['f1']:.3f} "
                  f"brier={metrics['brier']:.3f}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_metric=best_score, best_epoch=best_epoch,
                       val_metrics=best_metrics, history=history)


def train_binary_graph_model(
    model,
    train_graphs: list,
    val_graphs: list,
    cfg: Config | None = None,
    pos_weight: float | None = None,
    monitor: str = "auc",
    verbose: bool = True,
) -> TrainResult:
    """Train a graph-level binary model (b2 shot-blocking, Eq 22).

    The model mean-pools node embeddings to one logit per graph, trained with
    ``BCEWithLogitsLoss`` against the graph label ``y``. Validation reports only
    on real (non-proxy) graphs when an ``is_proxy`` flag is present, so augmented
    proxy positives never inflate the reported metrics.
    """
    import torch
    from torch_geometric.loader import DataLoader

    cfg = cfg or load_config()
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.training.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    higher_is_better = monitor in ("auc", "f1")
    best_score = -np.inf if higher_is_better else np.inf
    best_state = None
    best_epoch = -1
    best_metrics: dict = {}
    history = []
    patience = cfg.training.patience
    bad = 0

    for epoch in range(cfg.training.max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)  # (num_graphs,)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()

        model.eval()
        probs, ys, real = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
                is_proxy = getattr(batch, "is_proxy", None)
                real.append(np.zeros_like(ys[-1]) if is_proxy is None
                            else (is_proxy.cpu().numpy() == 0).astype(float))
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        keep = np.concatenate(real).astype(bool)
        if keep.any():  # evaluate on real shots only
            probs, ys = probs[keep], ys[keep]
        metrics = binary_metrics(ys, probs)
        score = metrics.get(monitor, float("nan"))
        history.append({"epoch": epoch, **metrics})

        improved = (score > best_score) if higher_is_better else (score < best_score)
        if not np.isnan(score) and improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | val {monitor}={score:.3f} f1={metrics['f1']:.3f} "
                  f"brier={metrics['brier']:.3f}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_metric=best_score, best_epoch=best_epoch,
                       val_metrics=best_metrics, history=history)


# --------------------------------------------------------------------------- #
# Masked-softmax selection tasks (a1 action-selection, d1 responsibility)
# --------------------------------------------------------------------------- #
def _selection_stats(logits, batch):
    """Per-graph softmax over candidate nodes; return CE losses, ranks, hits."""
    import torch
    import torch.nn.functional as F

    bvec = batch.batch
    cand = batch.candidate_mask
    tgt = batch.select_target
    num_graphs = int(bvec.max().item()) + 1
    losses, ranks, hits = [], [], []
    for gi in range(num_graphs):
        sel = bvec == gi
        cidx = torch.nonzero(sel & cand, as_tuple=False).squeeze(1)
        tidx = torch.nonzero(sel & tgt, as_tuple=False).squeeze(1)
        if cidx.numel() == 0 or tidx.numel() == 0:
            continue
        cl = logits[cidx]
        pos = int((cidx == tidx[0]).nonzero(as_tuple=False)[0, 0].item())
        logp = F.log_softmax(cl, dim=0)
        losses.append(-logp[pos])
        order = torch.argsort(cl, descending=True)
        rank = int((order == pos).nonzero(as_tuple=False)[0, 0].item()) + 1
        ranks.append(rank)
        hits.append(rank == 1)
    return losses, ranks, hits


def selection_metrics(ranks: list[int], ce_sum: float) -> dict[str, float]:
    if not ranks:
        return {"accuracy": float("nan"), "mrr": float("nan"), "ce": float("nan"), "n": 0}
    ranks_arr = np.array(ranks)
    return {
        "accuracy": float(np.mean(ranks_arr == 1)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "ce": float(ce_sum / len(ranks)),
        "n": int(len(ranks)),
    }


def train_selection_model(
    model, train_graphs, val_graphs, cfg: Config | None = None,
    monitor: str = "mrr", verbose: bool = True,
) -> TrainResult:
    """Train a node-selection model (a1/d1) with per-graph masked softmax + CE."""
    import torch
    from torch_geometric.loader import DataLoader

    cfg = cfg or load_config()
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.training.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    best_score, best_epoch, best_metrics, best_state = -np.inf, -1, {}, None
    history, bad = [], 0
    for epoch in range(cfg.training.max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            losses, _, _ = _selection_stats(logits, batch)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()

        model.eval()
        all_ranks, ce_sum = [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                losses, ranks, _ = _selection_stats(logits, batch)
                all_ranks.extend(ranks)
                ce_sum += float(sum(l.item() for l in losses))
        metrics = selection_metrics(all_ranks, ce_sum)
        score = metrics.get(monitor, float("nan"))
        history.append({"epoch": epoch, **metrics})
        if not np.isnan(score) and score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | val mrr={metrics['mrr']:.3f} "
                  f"acc={metrics['accuracy']:.3f} ce={metrics['ce']:.3f}")
        if bad >= cfg.training.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_score, best_epoch, best_metrics, history)


# --------------------------------------------------------------------------- #
# Outcome-conditioned per-node binary tasks (c1 goal-scoring, c2 conceding)
# --------------------------------------------------------------------------- #
def _outcome_target_logits(model, batch):
    """Gather the target node's outcome-matched logit (of its two per-node logits)."""
    import torch

    logits = model(batch)  # (num_nodes, 2)
    tl = logits[batch.target_mask]  # (B, 2)
    branch = batch.obs_outcome.view(-1)
    return tl[torch.arange(tl.shape[0], device=tl.device), branch]


def train_outcome_conditioned_model(
    model, train_graphs, val_graphs, cfg: Config | None = None,
    pos_weight: float | None = None, monitor: str = "auc", verbose: bool = True,
) -> TrainResult:
    """Train c1/c2: two logits per node, branch chosen by the observed outcome."""
    import torch
    from torch_geometric.loader import DataLoader

    cfg = cfg or load_config()
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(train_graphs, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.training.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    higher = monitor in ("auc", "f1")
    best_score = -np.inf if higher else np.inf
    best_epoch, best_metrics, best_state, history, bad = -1, {}, None, [], 0
    for epoch in range(cfg.training.max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = _outcome_target_logits(model, batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = _outcome_target_logits(model, batch)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
        metrics = binary_metrics(np.concatenate(ys), np.concatenate(probs))
        score = metrics.get(monitor, float("nan"))
        history.append({"epoch": epoch, **metrics})
        improved = (score > best_score) if higher else (score < best_score)
        if not np.isnan(score) and improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | val {monitor}={score:.3f} "
                  f"brier={metrics['brier']:.3f} base={metrics['base_rate']:.3f}")
        if bad >= cfg.training.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_score, best_epoch, best_metrics, history)
