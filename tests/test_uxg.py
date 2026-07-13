"""Tests for the UxG expected-goal model (task 3.6).

Geometry is tested against hand-computed values; the model is exercised on a
synthetic dataset so the suite runs without the (git-ignored) Wyscout download.
"""

import numpy as np
import pandas as pd
import pytest

from defcon.models.uxg import (
    SHOT_FEATURES,
    UxGModel,
    compute_uxg_features,
    shot_geometry,
)

L, W, G = 105.0, 68.0, 7.32  # pitch length, width, goal width


def test_geometry_on_goal_line_center_is_max_angle():
    g = shot_geometry(np.array([L]), np.array([W / 2]), L, W, G)
    assert g["x_rel"][0] == pytest.approx(0.0)
    assert g["y_rel"][0] == pytest.approx(0.0)
    assert g["distance"][0] == pytest.approx(0.0)
    # On the goal line between the posts the subtended angle is pi.
    assert g["angle"][0] == pytest.approx(np.pi, abs=1e-6)


def test_geometry_penalty_spot():
    # Penalty spot: 11 m from goal line, centered.
    g = shot_geometry(np.array([L - 11.0]), np.array([W / 2]), L, W, G)
    assert g["x_rel"][0] == pytest.approx(11.0)
    assert g["distance"][0] == pytest.approx(11.0)
    expected = np.arctan2(G * 11.0, 11.0**2 - (G / 2) ** 2)
    assert g["angle"][0] == pytest.approx(expected, abs=1e-9)


def test_geometry_angle_decreases_with_width():
    # Same distance from goal line, wider shot -> smaller angle.
    center = shot_geometry(np.array([L - 16.0]), np.array([W / 2]), L, W, G)["angle"][0]
    wide = shot_geometry(np.array([L - 16.0]), np.array([W / 2 + 20.0]), L, W, G)["angle"][0]
    assert wide < center


def test_geometry_distance_increases_away_from_goal():
    near = shot_geometry(np.array([L - 6.0]), np.array([W / 2]), L, W, G)["distance"][0]
    far = shot_geometry(np.array([L - 30.0]), np.array([W / 2]), L, W, G)["distance"][0]
    assert far > near


def test_compute_features_shape_and_columns():
    shots = pd.DataFrame(
        {
            "x_m": [90.0, 100.0, 70.0],
            "y_m": [34.0, 20.0, 50.0],
            "is_set_piece": [0, 1, 0],
            "is_header": [0, 0, 1],
        }
    )
    feats = compute_uxg_features(shots)
    assert list(feats.columns) == SHOT_FEATURES
    assert feats.shape == (3, 6)
    assert not feats.isnull().any().any()


def _synthetic_shots(n=4000, seed=0):
    """Shots whose goal probability rises as they get closer/more central."""
    rng = np.random.default_rng(seed)
    x_m = rng.uniform(75, 105, n)
    y_m = rng.uniform(10, 58, n)
    is_header = rng.integers(0, 2, n)
    is_set_piece = rng.integers(0, 2, n)
    dist = np.hypot(105 - x_m, y_m - 34)
    logit = 2.5 - 0.18 * dist - 0.4 * is_header + 0.3 * is_set_piece
    p = 1 / (1 + np.exp(-logit))
    is_goal = rng.binomial(1, p)
    return pd.DataFrame(
        {
            "match_id": rng.integers(0, 50, n),
            "x_m": x_m,
            "y_m": y_m,
            "is_header": is_header,
            "is_set_piece": is_set_piece,
            "is_goal": is_goal,
        }
    )


def test_model_fit_predict_and_metrics():
    df = _synthetic_shots()
    model = UxGModel().fit(df)
    p = model.predict_proba(df)
    assert p.shape == (len(df),)
    assert np.all((p >= 0) & (p <= 1))
    metrics = model.evaluate(df)
    # The signal is real and linear-ish -> AUC should be clearly > chance.
    assert metrics.auc > 0.7
    assert 0.0 <= metrics.brier <= 0.25


def test_model_closer_shot_scores_higher():
    df = _synthetic_shots()
    model = UxGModel().fit(df)
    near = model.score_location(102.0, 34.0)  # 3 m out, centered
    far = model.score_location(80.0, 34.0)    # 25 m out, centered
    assert near > far


def test_model_save_load_roundtrip(tmp_path):
    df = _synthetic_shots()
    model = UxGModel().fit(df)
    path = model.save(tmp_path / "uxg.joblib")
    reloaded = UxGModel.load(path)
    np.testing.assert_allclose(model.predict_proba(df), reloaded.predict_proba(df))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        UxGModel().predict_proba(_synthetic_shots(10))
