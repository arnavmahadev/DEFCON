"""End-to-end Phase 1 pipeline for one match (ties tasks 1.1-1.7 together).

Raw Metrica files -> unified tracking (+ kinematics) and an enriched per-action
table (synced frame, possession id, goal labels, inferred receiver). The action
table is cached to parquet keyed by match id (task 1.7).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from defcon.config import Config, load_config
from defcon.data.cache import cached_frame
from defcon.data.kinematics import add_kinematics
from defcon.data.labels import add_goal_labels, add_interceptor, add_possession_id
from defcon.data.metrica import load_metrica_events, load_metrica_tracking
from defcon.data.receiver import infer_intended_receivers
from defcon.data.sync import sync_events_to_tracking

__all__ = ["process_metrica_match", "ON_BALL_TYPES"]

ON_BALL_TYPES = ("pass", "shot")


def process_metrica_match(
    game_dir: str | Path,
    game_id: int = 1,
    cfg: Config | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(actions, tracking)`` for one Metrica game.

    ``actions`` is the cached, enriched on-ball action table; ``tracking`` is the
    full long-format tracking with kinematics (recomputed each call, not cached).
    """
    cfg = cfg or load_config()
    match_id = f"metrica_game{game_id}"

    tracking = load_metrica_tracking(game_dir=game_dir, game_id=game_id, cfg=cfg)
    tracking = add_kinematics(tracking, cfg)

    def build_actions() -> pd.DataFrame:
        events = load_metrica_events(game_dir=game_dir, game_id=game_id, cfg=cfg)
        events = sync_events_to_tracking(events, tracking, cfg)
        events = add_possession_id(events)
        events = add_goal_labels(events, cfg)
        events = add_interceptor(events)
        events["match_id"] = match_id

        # Intended receiver for every pass (true one for completed, inferred for failed).
        passes = events[events["type"] == "pass"].copy()
        inferred = infer_intended_receivers(passes, tracking, cfg)
        events["inferred_receiver"] = None
        events.loc[passes.index, "inferred_receiver"] = inferred
        # Prefer the recorded receiver when the pass completed.
        completed = (events["type"] == "pass") & (events["outcome"] == "success")
        events.loc[completed, "inferred_receiver"] = events.loc[completed, "receiver"]

        actions = events[events["type"].isin(ON_BALL_TYPES)].reset_index(drop=True)
        return actions

    if use_cache:
        actions = cached_frame("actions", match_id, build_actions, cfg, force=force)
    else:
        actions = build_actions()

    return actions, tracking
