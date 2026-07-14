"""Per-player and ball velocity / acceleration from tracking (task 1.4).

Raw finite differences on tracking positions explode because of measurement
jitter, so we smooth each trajectory with a Savitzky-Golay filter before
differentiating. Derivatives are computed per (team, player, period) so we never
differentiate across a half-time discontinuity, then clipped to physically
plausible caps from the config.

Adds columns: ``vx, vy, speed, ax, ay, accel`` (meters, seconds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from defcon.config import Config, load_config

__all__ = ["add_kinematics", "smooth_series"]


def smooth_series(a: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing that degrades gracefully for short trajectories."""
    n = len(a)
    if n < 3:
        return a.astype(float)
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return a.astype(float)
    poly = min(polyorder, w - 1)
    return savgol_filter(a.astype(float), w, poly)


def _derive_group(x: np.ndarray, y: np.ndarray, dt: float, cfg: Config):
    """Smooth positions then compute velocity/acceleration for one trajectory."""
    xs = smooth_series(x, cfg.tracking.savgol_window, cfg.tracking.savgol_polyorder)
    ys = smooth_series(y, cfg.tracking.savgol_window, cfg.tracking.savgol_polyorder)

    vx = np.gradient(xs) / dt
    vy = np.gradient(ys) / dt
    speed = np.hypot(vx, vy)

    # Clip implausible speeds, rescaling the velocity vector to keep direction.
    over = speed > cfg.tracking.max_speed
    if np.any(over):
        scale = np.where(over, cfg.tracking.max_speed / np.maximum(speed, 1e-9), 1.0)
        vx = vx * scale
        vy = vy * scale
        speed = np.hypot(vx, vy)

    ax = np.gradient(vx) / dt
    ay = np.gradient(vy) / dt
    accel = np.hypot(ax, ay)
    accel = np.clip(accel, 0.0, cfg.tracking.max_accel)

    return vx, vy, speed, ax, ay, accel


def add_kinematics(tracking: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Return a copy of ``tracking`` with velocity/acceleration columns added."""
    cfg = cfg or load_config()
    dt = cfg.tracking.dt

    df = tracking.sort_values(["team", "player_id", "period", "frame"]).reset_index(drop=True)
    for col in ("vx", "vy", "speed", "ax", "ay", "accel"):
        df[col] = np.nan

    # Derive per entity (player or ball) per period.
    for _, idx in df.groupby(["team", "player_id", "period"], sort=False).groups.items():
        rows = list(idx)
        sub = df.loc[rows]
        vx, vy, speed, ax, ay, accel = _derive_group(
            sub["x"].to_numpy(), sub["y"].to_numpy(), dt, cfg
        )
        df.loc[rows, "vx"] = vx
        df.loc[rows, "vy"] = vy
        df.loc[rows, "speed"] = speed
        df.loc[rows, "ax"] = ax
        df.loc[rows, "ay"] = ay
        df.loc[rows, "accel"] = accel

    return df
