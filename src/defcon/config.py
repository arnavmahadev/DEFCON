"""Configuration system for DEFCON.

A single YAML (``configs/default.yaml``) is the source of truth for pitch
geometry, kinematics, feature toggles, model dimensions, and paths. It is
parsed into nested frozen dataclasses so downstream code reads
``cfg.pitch.length`` rather than juggling dicts or hard-coded constants.

Usage::

    from defcon import load_config
    cfg = load_config()                       # loads configs/default.yaml
    cfg = load_config(overrides={"pitch": {"length": 100.0}})
    goal = cfg.pitch.attacking_goal_center    # (52.5, 0.0)

Changing a value in the YAML changes it everywhere, satisfying task 0.2.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config", "repo_root"]


def repo_root() -> Path:
    """Return the repository root (nearest ancestor containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from src/defcon/config.py
    return here.parents[2]


# --------------------------------------------------------------------------- #
# Nested config sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32

    @property
    def attacking_goal_center(self) -> tuple[float, float]:
        """Center of the goal the possessing team attacks (+x end)."""
        return (self.length / 2.0, 0.0)

    @property
    def defending_goal_center(self) -> tuple[float, float]:
        """Center of the goal the possessing team defends (-x end)."""
        return (-self.length / 2.0, 0.0)

    @property
    def attacking_goalposts(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two posts of the attacking goal: (left post, right post)."""
        x = self.length / 2.0
        half = self.goal_width / 2.0
        return ((x, +half), (x, -half))


@dataclass(frozen=True)
class Tracking:
    frame_rate: float = 25.0
    savgol_window: int = 13
    savgol_polyorder: int = 2
    max_speed: float = 12.0
    max_accel: float = 8.0

    @property
    def dt(self) -> float:
        """Seconds per frame."""
        return 1.0 / self.frame_rate


@dataclass(frozen=True)
class Labels:
    horizon_events: int = 10


@dataclass(frozen=True)
class Features:
    node_dim: int = 25
    edge_dim: int = 2
    responsibility_extra_dim: int = 1
    nearby_opponent_radius: float = 3.0
    passing_corridor_width: float = 10.0

    @property
    def responsibility_node_dim(self) -> int:
        return self.node_dim + self.responsibility_extra_dim


@dataclass(frozen=True)
class Model:
    hidden_dim: int = 64
    heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    mlp_hidden: int = 64


@dataclass(frozen=True)
class Training:
    seed: int = 42
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    val_fraction: float = 0.2
    split_by: str = "match"


@dataclass(frozen=True)
class UxG:
    test_fraction: float = 0.2
    seed: int = 42
    C: float = 1.0


@dataclass(frozen=True)
class Epv:
    penalty_kick_epv: float = 0.7884


@dataclass(frozen=True)
class Paths:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_cache: str = "data/cache"
    checkpoints: str = "checkpoints"
    wyscout_dir: str = "data/raw/wyscout"
    tracking_dir: str = "data/raw/tracking"

    def resolve(self, key: str, root: Path | None = None) -> Path:
        """Resolve a path field to an absolute Path (relative to repo root)."""
        root = root or repo_root()
        value = Path(getattr(self, key))
        return value if value.is_absolute() else (root / value)


@dataclass(frozen=True)
class Dataset:
    tracking_source: str = "metrica"


@dataclass(frozen=True)
class Config:
    pitch: Pitch = Pitch()
    tracking: Tracking = Tracking()
    labels: Labels = Labels()
    features: Features = Features()
    model: Model = Model()
    training: Training = Training()
    uxg: UxG = UxG()
    epv: Epv = Epv()
    paths: Paths = Paths()
    dataset: Dataset = Dataset()

    def path(self, key: str) -> Path:
        """Convenience: absolute Path for a `paths` entry, e.g. cfg.path('wyscout_dir')."""
        return self.paths.resolve(key)


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
_SECTION_TYPES: dict[str, type] = {
    "pitch": Pitch,
    "tracking": Tracking,
    "labels": Labels,
    "features": Features,
    "model": Model,
    "training": Training,
    "uxg": UxG,
    "epv": Epv,
    "paths": Paths,
    "dataset": Dataset,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build_section(section_type: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass section, ignoring unknown keys but warning-free."""
    if not is_dataclass(section_type):
        return data
    valid = {f.name for f in fields(section_type)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for section '{section_type.__name__}': {sorted(unknown)}"
        )
    return section_type(**kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration.

    Parameters
    ----------
    path : path to a YAML file. Defaults to ``configs/default.yaml`` at repo root.
    overrides : nested dict deep-merged over the file contents (e.g. from CLI).
    """
    if path is None:
        path = repo_root() / "configs" / "default.yaml"
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")

    sections = {
        name: _build_section(section_type, raw.get(name, {}))
        for name, section_type in _SECTION_TYPES.items()
    }
    return Config(**sections)
