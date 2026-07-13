"""Tests for kinematics (1.4) and event-tracking sync (1.3)."""

import numpy as np
import pandas as pd
import pytest

from defcon import load_config
from defcon.data.kinematics import add_kinematics, smooth_series
from defcon.data.sync import sync_events_to_tracking


def _straight_line_track(n=100, vx=5.0, dt=0.04):
    """A player moving at constant vx along x, plus a stationary ball."""
    frames = np.arange(1, n + 1)
    x = vx * (frames - 1) * dt
    rows = []
    for f, xi in zip(frames, x):
        rows.append({"period": 1, "frame": f, "time_s": (f - 1) * dt,
                     "team": "home", "player_id": "Player1", "x": xi, "y": 0.0, "z": np.nan})
        rows.append({"period": 1, "frame": f, "time_s": (f - 1) * dt,
                     "team": "ball", "player_id": "ball", "x": 10.0, "y": 0.0, "z": np.nan})
    return pd.DataFrame(rows)


def test_kinematics_constant_velocity():
    cfg = load_config()
    trk = _straight_line_track(vx=5.0)
    out = add_kinematics(trk, cfg)
    player = out[out.team == "home"]
    # Interior points should recover vx ~ 5 m/s, vy ~ 0, speed ~ 5.
    interior = player.iloc[5:-5]
    assert np.allclose(interior.vx, 5.0, atol=0.1)
    assert np.allclose(interior.vy, 0.0, atol=0.1)
    assert np.allclose(interior.speed, 5.0, atol=0.1)
    # Constant velocity -> ~zero acceleration.
    assert interior.accel.abs().max() < 0.2


def test_kinematics_speed_cap():
    cfg = load_config()
    trk = _straight_line_track(vx=50.0)  # absurdly fast
    out = add_kinematics(trk, cfg)
    assert out[out.team == "home"].speed.max() <= cfg.tracking.max_speed + 1e-6


def test_smooth_series_short_input():
    # Should not raise on very short trajectories.
    assert len(smooth_series(np.array([1.0, 2.0]), 13, 2)) == 2
    assert len(smooth_series(np.array([1.0]), 13, 2)) == 1


def test_kinematics_no_diff_across_periods():
    cfg = load_config()
    trk = _straight_line_track(vx=5.0)
    # Add a second-half segment far away; velocity must not blow up at the seam.
    second = _straight_line_track(vx=5.0)
    second["period"] = 2
    second["x"] = second["x"] + 1000.0  # discontinuous jump
    both = pd.concat([trk, second], ignore_index=True)
    out = add_kinematics(both, cfg)
    # No single speed should reflect the 1000 m jump (would be huge without the split).
    assert out.speed.max() <= cfg.tracking.max_speed + 1e-6


def test_sync_picks_closest_ball_frame():
    cfg = load_config()
    # Ball moves along x; an event at start_x=3 should sync to the frame where ball~3.
    frames = np.arange(1, 51)
    dt = cfg.tracking.dt
    ball = pd.DataFrame({
        "period": 1, "frame": frames, "time_s": (frames - 1) * dt,
        "team": "ball", "player_id": "ball", "x": (frames - 1) * 0.2, "y": 0.0, "z": np.nan,
    })
    events = pd.DataFrame([{
        "action_id": 0, "period": 1, "time_s": 1.0, "frame": 20, "type": "pass",
        "team": "home", "player": "Player1", "receiver": "Player2",
        "start_x": 3.0, "start_y": 0.0, "end_x": 5.0, "end_y": 0.0, "outcome": "success",
    }])
    out = sync_events_to_tracking(events, ball, cfg, window_s=1.0)
    # Ball x=3 at frame 16 ((16-1)*0.2=3.0); within the window around frame 20.
    assert out.loc[0, "sync_frame"] == 16
    assert out.loc[0, "sync_dist"] == pytest.approx(0.0, abs=1e-6)
