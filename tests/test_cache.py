"""Tests for the parquet caching layer (task 1.7)."""

import pandas as pd

from defcon import load_config
from defcon.data.cache import cache_path, cached_frame, load_parquet, save_parquet


def test_save_load_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "t.parquet"
    save_parquet(df, path)
    pd.testing.assert_frame_equal(load_parquet(path), df)


def test_cached_frame_builds_once(tmp_path):
    cfg = load_config(overrides={"paths": {"data_cache": str(tmp_path)}})
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return pd.DataFrame({"v": [calls["n"]]})

    first = cached_frame("actions", "match1", builder, cfg)
    second = cached_frame("actions", "match1", builder, cfg)  # should hit cache
    assert calls["n"] == 1
    assert first.equals(second)

    # force=True recomputes.
    third = cached_frame("actions", "match1", builder, cfg, force=True)
    assert calls["n"] == 2
    assert third["v"].iloc[0] == 2


def test_cache_path_layout(tmp_path):
    cfg = load_config(overrides={"paths": {"data_cache": str(tmp_path)}})
    p = cache_path("graphs", "metrica_game1", cfg)
    assert p.parent.name == "graphs"
    assert p.name == "metrica_game1.parquet"
