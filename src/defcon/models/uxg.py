"""UxG — Unblocked expected-goal model (task 3.6 / paper Section 3.2).

A standalone logistic-regression xG model trained on external Wyscout shot data.
It needs no tracking data, so it ships first (Milestone M1). Later phases call it
to value unblocked shots inside the EPV assembly (Eq 17).

Six features (paper Sec 3.2), all derived from a shot's pitch location plus two
flags:

1. ``x_rel``     — distance from the goal line along the pitch length (meters)
2. ``y_rel``     — absolute lateral offset from the goal center (meters)
3. ``distance``  — straight-line distance to the goal center (meters)
4. ``angle``     — angle (radians) subtended by the goal mouth at the shot spot
5. ``is_set_piece`` — 1 for free-kick shots / penalties, else 0
6. ``is_header``    — 1 if headed, else 0

The label is whether the shot was a goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defcon.config import Config, load_config

__all__ = [
    "SHOT_FEATURES",
    "shot_geometry",
    "compute_uxg_features",
    "UxGModel",
]

SHOT_FEATURES = ["x_rel", "y_rel", "distance", "angle", "is_set_piece", "is_header"]


def shot_geometry(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    pitch_length: float,
    pitch_width: float,
    goal_width: float,
) -> dict[str, np.ndarray]:
    """Vectorized shot geometry relative to the attacking goal at (length, width/2).

    Returns a dict with ``x_rel``, ``y_rel`` (abs lateral offset), ``distance``,
    and ``angle`` (goal-mouth angle in radians, in [0, pi]).
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)

    goal_x = pitch_length
    goal_y = pitch_width / 2.0
    half_goal = goal_width / 2.0

    x_rel = goal_x - x_m               # distance to goal line along length (>= 0 in play)
    y_off = y_m - goal_y               # signed lateral offset
    y_rel = np.abs(y_off)              # symmetric lateral offset
    distance = np.hypot(x_rel, y_off)

    # Angle subtended by the two posts at the shot location.
    # Standard xG formula: atan2(goal_width * x_rel, x_rel^2 + y_off^2 - half_goal^2).
    numerator = goal_width * x_rel
    denominator = x_rel**2 + y_off**2 - half_goal**2
    angle = np.arctan2(numerator, denominator)
    # atan2 returns (-pi, pi]; the goal angle is non-negative for x_rel >= 0.
    angle = np.where(angle < 0, angle + np.pi, angle)

    return {"x_rel": x_rel, "y_rel": y_rel, "distance": distance, "angle": angle}


def compute_uxg_features(shots: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Build the 6-column UxG feature frame from a shots DataFrame.

    ``shots`` must have columns ``x_m``, ``y_m``, ``is_set_piece``, ``is_header``.
    """
    cfg = cfg or load_config()
    geom = shot_geometry(
        shots["x_m"].to_numpy(),
        shots["y_m"].to_numpy(),
        cfg.pitch.length,
        cfg.pitch.width,
        cfg.pitch.goal_width,
    )
    out = pd.DataFrame(index=shots.index)
    out["x_rel"] = geom["x_rel"]
    out["y_rel"] = geom["y_rel"]
    out["distance"] = geom["distance"]
    out["angle"] = geom["angle"]
    out["is_set_piece"] = shots["is_set_piece"].astype(float).to_numpy()
    out["is_header"] = shots["is_header"].astype(float).to_numpy()
    return out[SHOT_FEATURES]


@dataclass
class UxGMetrics:
    auc: float
    brier: float
    log_loss: float
    n: int
    base_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "n": self.n,
            "base_rate": self.base_rate,
        }


class UxGModel:
    """Logistic-regression expected-goal model with a StandardScaler front-end."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=self.cfg.uxg.C,
                        max_iter=2000,
                        class_weight=None,
                    ),
                ),
            ]
        )
        self.feature_names = list(SHOT_FEATURES)
        self._fitted = False

    # -- training / inference -------------------------------------------------
    def fit(self, shots: pd.DataFrame, labels: np.ndarray | pd.Series | None = None) -> "UxGModel":
        """Fit on a shots DataFrame. Labels default to the ``is_goal`` column."""
        X = compute_uxg_features(shots, self.cfg).to_numpy()
        y = self._resolve_labels(shots, labels)
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, shots: pd.DataFrame) -> np.ndarray:
        """Return P(goal) for each shot."""
        self._check_fitted()
        X = compute_uxg_features(shots, self.cfg).to_numpy()
        return self.pipeline.predict_proba(X)[:, 1]

    def score_location(
        self,
        x_m: float,
        y_m: float,
        is_header: bool = False,
        is_set_piece: bool = False,
    ) -> float:
        """Convenience: UxG for a single (x, y) shot in pitch meters."""
        df = pd.DataFrame(
            [{"x_m": x_m, "y_m": y_m, "is_header": int(is_header), "is_set_piece": int(is_set_piece)}]
        )
        return float(self.predict_proba(df)[0])

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, shots: pd.DataFrame, labels: np.ndarray | pd.Series | None = None) -> UxGMetrics:
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        y = self._resolve_labels(shots, labels)
        p = self.predict_proba(shots)
        return UxGMetrics(
            auc=float(roc_auc_score(y, p)),
            brier=float(brier_score_loss(y, p)),
            log_loss=float(log_loss(y, p, labels=[0, 1])),
            n=int(len(y)),
            base_rate=float(np.mean(y)),
        )

    @property
    def coefficients(self) -> dict[str, float]:
        """Feature -> logistic-regression coefficient (on standardized inputs)."""
        self._check_fitted()
        lr: LogisticRegression = self.pipeline.named_steps["lr"]
        coefs = dict(zip(self.feature_names, lr.coef_.ravel().tolist()))
        coefs["intercept"] = float(lr.intercept_[0])
        return coefs

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names, "fitted": self._fitted},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, cfg: Config | None = None) -> "UxGModel":
        obj = joblib.load(Path(path))
        model = cls(cfg)
        model.pipeline = obj["pipeline"]
        model.feature_names = obj["feature_names"]
        model._fitted = obj["fitted"]
        return model

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _resolve_labels(shots: pd.DataFrame, labels) -> np.ndarray:
        if labels is not None:
            return np.asarray(labels).astype(int)
        if "is_goal" not in shots.columns:
            raise ValueError("No labels provided and no 'is_goal' column in shots frame.")
        return shots["is_goal"].to_numpy().astype(int)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("UxGModel is not fitted yet. Call .fit(...) first.")
