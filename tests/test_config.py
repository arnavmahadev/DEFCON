"""Tests for the configuration system (task 0.2)."""

import pytest

from defcon import Config, load_config
from defcon.config import repo_root


def test_package_imports():
    """Smoke test: the package imports and exposes a version (task 0.1)."""
    import defcon

    assert defcon.__version__


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.pitch.length == 105.0
    assert cfg.pitch.width == 68.0
    assert cfg.features.node_dim == 25
    assert cfg.features.edge_dim == 2


def test_pitch_geometry_derived():
    cfg = load_config()
    assert cfg.pitch.attacking_goal_center == (52.5, 0.0)
    assert cfg.pitch.defending_goal_center == (-52.5, 0.0)
    left, right = cfg.pitch.attacking_goalposts
    assert left == (52.5, 3.66)
    assert right == (52.5, -3.66)


def test_changing_pitch_length_propagates():
    """Task 0.2 acceptance: changing pitch length changes it everywhere."""
    cfg = load_config(overrides={"pitch": {"length": 100.0}})
    assert cfg.pitch.length == 100.0
    # Derived geometry must follow.
    assert cfg.pitch.attacking_goal_center == (50.0, 0.0)
    assert cfg.pitch.attacking_goalposts[0] == (50.0, 3.66)


def test_responsibility_dim():
    cfg = load_config()
    assert cfg.features.responsibility_node_dim == 26


def test_tracking_dt():
    cfg = load_config()
    assert cfg.tracking.dt == pytest.approx(1.0 / 25.0)


def test_paths_resolve_absolute():
    cfg = load_config()
    p = cfg.path("wyscout_dir")
    assert p.is_absolute()
    assert p == repo_root() / "data/raw/wyscout"


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(overrides={"pitch": {"not_a_field": 1}})


def test_unknown_section_raises():
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(overrides={"nonsense": {"a": 1}})
