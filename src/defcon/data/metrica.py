"""Loaders for the Metrica Sports sample data (tasks 1.1 and 1.2).

Metrica ships two raw tracking CSVs per game (Home / Away) plus one events CSV.
Tracking coordinates are normalized to [0, 1] with the origin at the top-left
corner (x rightward, y downward). We convert to meters with the origin at the
pitch center and y pointing up (standard math orientation), producing the
unified long-format schema the rest of the pipeline expects.

Unified tracking schema (one row per player-or-ball per frame)::

    period, frame, time_s, team, player_id, x, y, z

where ``team`` is ``home`` / ``away`` / ``ball`` and coordinates are meters.

Unified event schema (task 1.2)::

    action_id, period, time_s, frame, type, team, player, receiver,
    start_x, start_y, end_x, end_y, outcome, subtype

with ``type`` in {pass, dribble, shot, ...} and ``outcome`` in
{success, fail, out, offside, ...}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = [
    "load_metrica_tracking",
    "load_metrica_events",
    "infer_playing_direction",
    "identify_goalkeepers",
    "orient_attacking_positive_x",
    "metrica_game_paths",
]

# Metrica event Type -> our canonical action type (default when no special rule).
_TYPE_MAP = {
    "PASS": "pass",
    "SHOT": "shot",
    "RECOVERY": "recovery",
    "BALL LOST": "ball_lost",
    "BALL OUT": "ball_out",
    "CHALLENGE": "challenge",
    "SET PIECE": "set_piece",
    "FAULT RECEIVED": "foul",
    "CARD": "card",
}


def metrica_game_paths(game_dir: str | Path, game_id: int = 1) -> dict[str, Path]:
    """Return the three Metrica file paths for a sample game directory."""
    game_dir = Path(game_dir)
    stem = f"Sample_Game_{game_id}"
    return {
        "home": game_dir / f"{stem}_RawTrackingData_Home_Team.csv",
        "away": game_dir / f"{stem}_RawTrackingData_Away_Team.csv",
        "events": game_dir / f"{stem}_RawEventsData.csv",
    }


def _normalize_xy(x: np.ndarray, y: np.ndarray, length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Metrica [0,1] top-left coords -> meters, origin at center, y up."""
    x_m = (x - 0.5) * length
    y_m = -(y - 0.5) * width
    return x_m, y_m


def _parse_team_tracking(path: Path, team: str, cfg: Config) -> pd.DataFrame:
    """Parse one Metrica team tracking CSV into long format (players + ball)."""
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    # First three columns are metadata; the rest are (x, y) pairs whose label
    # sits on the x column ("Player11", ..., "Ball").
    meta = cols[:3]  # Period, Frame, Time [s]
    assert meta[0] == "Period" and meta[1] == "Frame", f"Unexpected header in {path.name}: {meta}"

    records = []
    for i in range(3, len(cols), 2):
        label = cols[i]
        x_col, y_col = cols[i], cols[i + 1]
        x_m, y_m = _normalize_xy(
            raw[x_col].to_numpy(dtype=float),
            raw[y_col].to_numpy(dtype=float),
            cfg.pitch.length,
            cfg.pitch.width,
        )
        is_ball = label.lower() == "ball"
        block = pd.DataFrame(
            {
                "period": raw["Period"].to_numpy(),
                "frame": raw["Frame"].to_numpy(),
                "time_s": raw["Time [s]"].to_numpy(),
                "team": "ball" if is_ball else team,
                "player_id": "ball" if is_ball else label,
                "x": x_m,
                "y": y_m,
                "z": np.nan,  # Metrica CSV has no ball height
            }
        )
        records.append(block)

    df = pd.concat(records, ignore_index=True)
    # Off-pitch / not-yet-subbed players have NaN positions -> drop them.
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return df


def load_metrica_tracking(
    game_dir: str | Path | None = None,
    game_id: int = 1,
    home_csv: str | Path | None = None,
    away_csv: str | Path | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Load Metrica tracking (home + away + ball) into the unified long schema.

    Ball rows are de-duplicated (both team files contain identical ball columns).
    """
    cfg = cfg or load_config()
    if game_dir is not None:
        paths = metrica_game_paths(game_dir, game_id)
        home_csv, away_csv = paths["home"], paths["away"]
    if home_csv is None or away_csv is None:
        raise ValueError("Provide either game_dir or both home_csv and away_csv.")

    home = _parse_team_tracking(Path(home_csv), "home", cfg)
    away = _parse_team_tracking(Path(away_csv), "away", cfg)
    # Ball appears in both files; keep it once (from home).
    away = away[away["team"] != "ball"]
    df = pd.concat([home, away], ignore_index=True)
    df = df.sort_values(["period", "frame", "team", "player_id"]).reset_index(drop=True)
    return df


def infer_playing_direction(tracking: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Infer each (team, period)'s attacking direction in *raw* coordinates.

    Returns +1 if the team attacks toward +x, -1 toward -x. The goalkeeper is the
    deepest player (largest |mean x|) and sits in front of the goal the team
    *defends*, so the attacking direction is the opposite sign of the GK's x.
    This is far more robust than a kick-off heuristic.
    """
    out: dict[tuple[str, int], int] = {}
    players = tracking[tracking["team"].isin(["home", "away"])]
    for (period, team), group in players.groupby(["period", "team"]):
        mean_x = group.groupby("player_id")["x"].mean()
        gk_x = mean_x.loc[mean_x.abs().idxmax()]  # deepest player ~ goalkeeper
        out[(team, int(period))] = -1 if gk_x > 0 else 1
    return out


def identify_goalkeepers(tracking: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Return the goalkeeper player_id for each (team, period).

    The GK is the player with the largest |mean x| over the half (they sit
    deepest, in front of their own goal). Consistent with
    :func:`infer_playing_direction`.
    """
    out: dict[tuple[str, int], str] = {}
    players = tracking[tracking["team"].isin(["home", "away"])]
    for (period, team), group in players.groupby(["period", "team"]):
        mean_x = group.groupby("player_id")["x"].mean()
        out[(team, int(period))] = str(mean_x.abs().idxmax())
    return out


def orient_attacking_positive_x(
    frame_df: pd.DataFrame,
    attacking_team: str,
    period: int,
    directions: dict[tuple[str, int], int],
) -> pd.DataFrame:
    """Flip a frame so ``attacking_team`` attacks +x (task 1.1 normalization).

    ``frame_df`` is a slice of the tracking frame (players + ball). If the team
    already attacks +x nothing changes; otherwise x and y are negated (a 180°
    rotation about the pitch center, preserving left/right handedness).
    """
    direction = directions[(attacking_team, int(period))]
    out = frame_df.copy()
    if direction == -1:
        out["x"] = -out["x"]
        out["y"] = -out["y"]
    return out


def load_metrica_events(
    events_csv: str | Path | None = None,
    game_dir: str | Path | None = None,
    game_id: int = 1,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Load Metrica events into the unified event schema (task 1.2).

    Outcome is inferred per Metrica conventions: a PASS/SHOT is a success unless
    its subtype/next context marks it lost, out, or offside. Metrica encodes the
    receiver in the ``To`` column for completed passes.
    """
    cfg = cfg or load_config()
    if events_csv is None:
        if game_dir is None:
            raise ValueError("Provide either events_csv or game_dir.")
        events_csv = metrica_game_paths(game_dir, game_id)["events"]
    raw = pd.read_csv(events_csv)

    length, width = cfg.pitch.length, cfg.pitch.width
    sx, sy = _normalize_xy(raw["Start X"].to_numpy(float), raw["Start Y"].to_numpy(float), length, width)
    ex, ey = _normalize_xy(raw["End X"].to_numpy(float), raw["End Y"].to_numpy(float), length, width)

    action_type, outcome = _classify_events(raw)
    df = pd.DataFrame(
        {
            "action_id": np.arange(len(raw)),
            "period": raw["Period"].to_numpy(),
            "time_s": raw["Start Time [s]"].to_numpy(),
            "end_time_s": raw["End Time [s]"].to_numpy(),
            "frame": raw["Start Frame"].to_numpy(),
            "end_frame": raw["End Frame"].to_numpy(),
            "type": action_type,
            "outcome": outcome,
            "subtype": raw["Subtype"].fillna("").to_numpy(),
            "team": raw["Team"].str.lower().to_numpy(),
            "player": raw["From"].to_numpy(),
            "receiver": raw["To"].to_numpy(),
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
        }
    )
    return df


def _classify_events(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map Metrica (Type, Subtype) to canonical (type, outcome).

    The key Metrica quirk: *completed* passes are ``PASS`` while *intercepted*
    passes are ``BALL LOST`` events (with the pass trajectory in the end
    coordinates). We fold interception-type ball losses back into the pass set
    as failures so downstream models see a realistic completion rate and the
    Intercept credit rules have failed passes to work with.
    """
    etype = raw["Type"].to_numpy()
    subtype = raw["Subtype"].fillna("").str.upper().to_numpy()

    action_type = np.array([_TYPE_MAP.get(t, str(t).lower()) for t in etype], dtype=object)
    outcome = np.full(len(raw), "success", dtype=object)

    has = lambda key: np.char.find(subtype.astype(str), key) >= 0  # noqa: E731

    is_pass = etype == "PASS"
    outcome[is_pass] = "success"

    is_ball_lost = etype == "BALL LOST"
    intercepted = is_ball_lost & has("INTERCEPTION")
    # Intercepted deliveries become failed passes.
    action_type[intercepted] = "pass"
    outcome[intercepted] = "fail"
    # Other ball losses (miscontrol, theft, forced) stay as failed ball losses.
    outcome[is_ball_lost & ~intercepted] = "fail"
    outcome[is_ball_lost & has("OFFSIDE")] = "offside"

    is_ball_out = etype == "BALL OUT"
    outcome[is_ball_out] = "out"

    # Shots: success only if a goal marker is present ('GOAL' but not 'GOAL KICK').
    is_shot = etype == "SHOT"
    is_goal = has("GOAL") & ~has("GOAL KICK")
    outcome[is_shot] = "fail"
    outcome[is_shot & is_goal] = "success"

    return action_type, outcome
