"""Tests for geometry helpers and the 25-dim node feature vector (task 2.1).

The acceptance test crafts a tiny 3-player frame and checks feature values by
hand, including the tricky triangle / corridor / goalside counts.
"""

import numpy as np
import pytest

from defcon import load_config
from defcon.features.geometry import (
    dist_point_to_segments,
    points_in_triangle,
    vector_angle_sin_cos,
)
from defcon.features.nodes import NODE_FEATURES, build_node_features
from defcon.features.state import GraphState


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_point_in_triangle_basic():
    # Triangle (0,0),(4,0),(0,4).
    inside = points_in_triangle(np.array([1.0, 3.0]), np.array([1.0, 3.0]),
                                0, 0, 4, 0, 0, 4)
    assert inside[0]  # (1,1) inside
    assert not inside[1]  # (3,3) outside (above hypotenuse)


def test_dist_point_to_segment():
    # Segment (0,0)->(10,0). Point (5,3) -> distance 3; point (-2,0) -> clamps to A, dist 2.
    d = dist_point_to_segments(np.array([5.0, -2.0]), np.array([3.0, 0.0]), 0, 0, 10, 0)
    assert d[0] == pytest.approx(3.0)
    assert d[1] == pytest.approx(2.0)


def test_vector_angle_perpendicular():
    sin, cos = vector_angle_sin_cos(np.array([0.0]), np.array([1.0]), 1.0, 0.0)
    assert cos[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(sin[0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Node feature acceptance
# --------------------------------------------------------------------------- #
def _crafted_state(cfg):
    """Carrier at origin moving +x; teammate at (20,0); one defender at (3,0)."""
    return GraphState(
        player_ids=["carrier", "mate", "def"],
        px=np.array([0.0, 20.0, 3.0]),
        py=np.array([0.0, 0.0, 0.0]),
        pvx=np.array([5.0, 0.0, 0.0]),
        pvy=np.array([0.0, 0.0, 0.0]),
        pspeed=np.array([5.0, 0.0, 0.0]),
        paccel=np.array([0.0, 0.0, 0.0]),
        is_attacking=np.array([1.0, 1.0, 0.0]),
        is_gk=np.array([0.0, 0.0, 0.0]),
        carrier_idx=0,
        ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=5.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
        attacking_team="home", period=1, frame=1,
    )


def test_node_vector_length_and_nodes():
    cfg = load_config()
    ng = build_node_features(_crafted_state(cfg), cfg)
    assert ng.x.shape == (5, 25)  # 3 players + 2 goal nodes
    assert len(NODE_FEATURES) == 25
    assert ng.node_types == ["player", "player", "player", "goal", "goal"]
    assert ng.node_ids[-2:] == ["goal_left", "goal_right"]


def test_node_features_hand_values_carrier():
    cfg = load_config()
    ng = build_node_features(_crafted_state(cfg), cfg)
    f = dict(zip(NODE_FEATURES, ng.x[0]))  # carrier row
    assert f["is_carrier"] == 1 and f["is_teammate"] == 1
    assert f["is_gk"] == 0 and f["is_goal_node"] == 0
    assert f["x"] == 0 and f["y"] == 0 and f["speed"] == pytest.approx(5.0)
    assert f["dist_to_goal"] == pytest.approx(52.5)
    assert f["goal_cos"] == pytest.approx(1.0) and f["goal_sin"] == pytest.approx(0.0)
    assert f["dist_to_ball"] == pytest.approx(0.0)
    # carrier vs carrier velocity -> angle 0.
    assert f["carrier_vel_cos"] == pytest.approx(1.0)
    # nearest defender is 3 m away and within the 3 m radius.
    assert f["dist_nearest_opp"] == pytest.approx(3.0)
    assert f["n_opp_within_r"] == 1
    # defender at (3,0) is goalside and inside the cone to the goal mouth.
    assert f["n_opp_goalside"] == 1
    assert f["n_opp_in_triangle"] == 1


def test_node_features_hand_values_teammate():
    cfg = load_config()
    ng = build_node_features(_crafted_state(cfg), cfg)
    f = dict(zip(NODE_FEATURES, ng.x[1]))  # teammate row
    assert f["is_carrier"] == 0 and f["is_teammate"] == 1
    assert f["dist_to_goal"] == pytest.approx(32.5)
    assert f["dist_nearest_opp"] == pytest.approx(17.0)
    assert f["n_opp_goalside"] == 0
    assert f["n_opp_in_triangle"] == 0
    # The defender lies exactly on the carrier->teammate passing line.
    assert f["passline_dist_nearest_opp"] == pytest.approx(0.0)
    assert f["n_opp_in_corridor"] == 1


def test_defender_node_excludes_self():
    cfg = load_config()
    ng = build_node_features(_crafted_state(cfg), cfg)
    f = dict(zip(NODE_FEATURES, ng.x[2]))  # defender row
    assert f["is_teammate"] == 0
    # Only one defender exists -> no *other* opponents -> "far" sentinel.
    assert f["dist_nearest_opp"] == pytest.approx(cfg.pitch.length)
    assert f["n_opp_within_r"] == 0


def test_masks():
    cfg = load_config()
    ng = build_node_features(_crafted_state(cfg), cfg)
    assert ng.teammate_mask().tolist() == [False, True, False, False, False]
    assert ng.defender_mask().tolist() == [False, False, True, False, False]
    assert ng.goal_mask().tolist() == [False, False, False, True, True]
