"""Intended-receiver inference for passes (task 1.6 / paper Eq 20).

For a failed pass we don't know who it was meant for. The paper picks the
teammate v maximizing::

    (min_u Dist(a,u) / Dist(a,v)) * (min_u Angle(a,u) / Angle(a,v))

where ``Dist`` is the distance from the pass endpoint to teammate v (at the
moment the ball arrives) and ``Angle`` is the angle between the pass line and the
passer->v line. This rewards teammates who are both close to where the ball went
and aligned with the pass direction.

We validate on *completed* passes (true receiver known): the heuristic should
recover the real receiver > 80% of the time before we trust it on failed passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = ["infer_intended_receivers", "receiver_recovery_rate"]

_EPS = 1e-6


def _positions_at_frames(tracking: pd.DataFrame, needed: set[tuple[int, int]]) -> dict:
    """Map (period, frame) -> DataFrame of player rows, only for needed frames."""
    players = tracking[tracking["team"].isin(["home", "away"])]
    key = list(zip(players["period"].to_numpy(), players["frame"].to_numpy()))
    players = players.assign(_key=key)
    wanted = players[players["_key"].isin(needed)]
    return {k: g for k, g in wanted.groupby("_key")}


def infer_intended_receivers(
    passes: pd.DataFrame,
    tracking: pd.DataFrame,
    cfg: Config | None = None,
) -> pd.Series:
    """Return the inferred receiver player_id for each pass row (index-aligned).

    Uses each pass's reception frame (``end_frame`` if > 0 else ``sync_frame``/
    ``frame``) to read teammate positions.
    """
    cfg = cfg or load_config()

    def reception_frame(row) -> int:
        ef = row.get("end_frame", 0)
        if ef and ef > 0:
            return int(ef)
        sf = row.get("sync_frame", -1)
        return int(sf if sf and sf > 0 else row["frame"])

    recept = passes.apply(reception_frame, axis=1)
    needed = set(zip(passes["period"].astype(int), recept.astype(int)))
    lookup = _positions_at_frames(tracking, needed)

    results = {}
    for (idx, row), rframe in zip(passes.iterrows(), recept):
        team = row["team"]
        passer = row["player"]
        start = np.array([row["start_x"], row["start_y"]], dtype=float)
        end = np.array([row["end_x"], row["end_y"]], dtype=float)
        frame_players = lookup.get((int(row["period"]), int(rframe)))
        if frame_players is None or np.isnan(end).any() or np.isnan(start).any():
            results[idx] = None
            continue
        mates = frame_players[(frame_players["team"] == team) & (frame_players["player_id"] != passer)]
        if len(mates) == 0:
            results[idx] = None
            continue

        pos = mates[["x", "y"]].to_numpy()
        ids = mates["player_id"].to_numpy()

        dist = np.linalg.norm(pos - end, axis=1)  # distance to pass endpoint
        pass_vec = end - start
        pass_norm = np.linalg.norm(pass_vec) + _EPS
        to_mate = pos - start
        to_mate_norm = np.linalg.norm(to_mate, axis=1) + _EPS
        cos = np.clip((to_mate @ pass_vec) / (to_mate_norm * pass_norm), -1.0, 1.0)
        angle = np.arccos(cos)  # radians, angle between pass line and passer->mate

        dist_score = (dist.min() + _EPS) / (dist + _EPS)
        angle_score = (angle.min() + _EPS) / (angle + _EPS)
        score = dist_score * angle_score
        results[idx] = ids[int(np.argmax(score))]

    return pd.Series(results, name="inferred_receiver")


def receiver_recovery_rate(
    completed_passes: pd.DataFrame,
    tracking: pd.DataFrame,
    cfg: Config | None = None,
) -> float:
    """Fraction of completed passes whose true receiver the heuristic recovers."""
    inferred = infer_intended_receivers(completed_passes, tracking, cfg)
    true = completed_passes["receiver"]
    valid = inferred.notna() & true.notna()
    if valid.sum() == 0:
        return float("nan")
    return float((inferred[valid].values == true[valid].values).mean())
