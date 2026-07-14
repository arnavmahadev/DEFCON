"""Parquet caching layer (task 1.7).

Persist processed per-action records keyed by ``(match_id, ...)`` so we never
recompute the data pipeline. Graph tensors (task 2.3) will reuse the same key
scheme under a different ``kind`` subdirectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from defcon.config import Config, load_config

__all__ = ["cache_path", "save_parquet", "load_parquet", "cached_frame"]


def cache_path(kind: str, key: str, cfg: Config | None = None, suffix: str = ".parquet") -> Path:
    """Return ``data/cache/<kind>/<key><suffix>`` as an absolute path."""
    cfg = cfg or load_config()
    root = cfg.path("data_cache") / kind
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}{suffix}"


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cached_frame(
    kind: str,
    key: str,
    builder: Callable[[], pd.DataFrame],
    cfg: Config | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Memoize a DataFrame to parquet: load if present, else build and save.

    ``force=True`` recomputes and overwrites.
    """
    cfg = cfg or load_config()
    path = cache_path(kind, key, cfg)
    if path.exists() and not force:
        return load_parquet(path)
    df = builder()
    save_parquet(df, path)
    return df
