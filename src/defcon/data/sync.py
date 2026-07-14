"""Event <-> tracking synchronization (task 1.3).

The paper syncs events to frames with ELASTIC; we approximate: for each event we
search a small window of frames around its recorded timestamp and pick the frame
whose ball position is closest to the event's start location. Manual event times
drift by ~1s, so we do not trust them alone — the ball-distance objective anchors
the match. Passes are synced to the *release* frame (the event start), not the
reception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = ["sync_events_to_tracking"]


def sync_events_to_tracking(
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    cfg: Config | None = None,
    window_s: float = 1.0,
) -> pd.DataFrame:
    """Attach ``sync_frame`` and ``sync_dist`` (ball-to-start distance, m) to events.

    For each event, candidate frames lie within ``window_s`` of the event's
    recorded frame (same period); the chosen frame minimizes the distance from
    the ball to the event's start location.
    """
    cfg = cfg or load_config()
    window_frames = int(round(window_s * cfg.tracking.frame_rate))

    ball = tracking[tracking["team"] == "ball"]
    # Per-period lookup arrays for fast slicing.
    ball_by_period: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for period, g in ball.groupby("period"):
        g = g.sort_values("frame")
        ball_by_period[int(period)] = (
            g["frame"].to_numpy(),
            g["x"].to_numpy(),
            g["y"].to_numpy(),
        )

    sync_frame = np.full(len(events), -1, dtype=int)
    sync_dist = np.full(len(events), np.nan)

    for i, (_, ev) in enumerate(events.iterrows()):
        period = int(ev["period"])
        if period not in ball_by_period:
            continue
        frames, bx, by = ball_by_period[period]
        center = ev["frame"]
        lo, hi = center - window_frames, center + window_frames
        mask = (frames >= lo) & (frames <= hi)
        if not np.any(mask):
            # Fall back to the nearest frame overall.
            j = int(np.argmin(np.abs(frames - center)))
        else:
            cand = np.flatnonzero(mask)
            sx, sy = ev["start_x"], ev["start_y"]
            if np.isnan(sx) or np.isnan(sy):
                # No start location -> just take the recorded frame.
                j = cand[int(np.argmin(np.abs(frames[cand] - center)))]
            else:
                d = np.hypot(bx[cand] - sx, by[cand] - sy)
                j = cand[int(np.argmin(d))]
        sync_frame[i] = int(frames[j])
        if not (np.isnan(ev["start_x"]) or np.isnan(ev["start_y"])):
            sync_dist[i] = float(np.hypot(bx[j] - ev["start_x"], by[j] - ev["start_y"]))

    out = events.copy()
    out["sync_frame"] = sync_frame
    out["sync_dist"] = sync_dist
    return out
