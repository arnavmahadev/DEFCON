"""Boosting baselines on tabular features (task 7.2).

Gradient boosting can't do the "select one node" tasks (a1, d1), but it competes
on the per-node binary tasks (b1, c1, c2) using a flat feature vector: the paper
concatenates the ball-carrier's node features with the intended receiver's. We
extract exactly that from the graph datasets and fit XGBoost / CatBoost.
"""

from __future__ import annotations

import numpy as np

from defcon.features.nodes import NODE_FEATURES
from defcon.models.train import binary_metrics

__all__ = ["graphs_to_tabular", "train_boosting", "TABULAR_FEATURES"]

TABULAR_FEATURES = [f"carrier_{f}" for f in NODE_FEATURES] + [f"target_{f}" for f in NODE_FEATURES]


def graphs_to_tabular(graphs) -> tuple[np.ndarray, np.ndarray]:
    """Flatten each graph to [carrier features | target features] + label y."""
    X, y = [], []
    for g in graphs:
        carrier = int(g.carrier_idx.item())
        target = int(np.flatnonzero(g.target_mask.numpy())[0])
        row = np.concatenate([g.x[carrier].numpy(), g.x[target].numpy()])
        X.append(row)
        y.append(float(g.y.item()))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)


def train_boosting(
    Xtr, ytr, Xva, yva, kind: str = "xgb", seed: int = 42, scale_pos_weight: float | None = None
) -> dict:
    """Fit XGBoost or CatBoost and return F1/AUC/Brier on the validation set."""
    kind = kind.lower()
    if kind in ("xgb", "xgboost"):
        from xgboost import XGBClassifier

        spw = scale_pos_weight or ((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=seed,
            scale_pos_weight=spw, n_jobs=4,
        )
    elif kind in ("cat", "catboost"):
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05, loss_function="Logloss",
            random_seed=seed, verbose=False, auto_class_weights="Balanced",
        )
    else:
        raise ValueError(f"Unknown boosting kind {kind!r}")

    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    return binary_metrics(yva, p)
