"""Loaders for PFF FC's FIFA World Cup 2022 broadcast tracking data (task 7.3).

PFF FC released synchronized broadcast tracking + event data for all 64 games of
the 2022 men's World Cup. Unlike the anonymized Metrica sample, **PFF ships real
player identities** (rosters map jersey numbers to player ids + names), which is
exactly what the market-value validation (Sec 4.3, the paper's headline "better
prevent than tackle" finding) needs to join onto Transfermarkt values.

Access is free but gated behind a request form (see ``scripts/download_pff.py``).
This module parses the released format into the same unified schema the Metrica
loader produces, so the rest of the pipeline runs unchanged:

Per game PFF ships three files (kloppy's naming convention, mirrored here)::

    {game_id}.jsonl[.bz2]      # tracking: one JSON object per frame
    metadata_{game_id}.json    # teams, pitch, fps, kick-off side
    rosters_{game_id}.json     # jersey <-> player id + name + position

Unified tracking schema (identical to :mod:`defcon.data.metrica`)::

    period, frame, time_s, team, player_id, x, y, z

with ``team`` in {home, away, ball}, coordinates in meters (origin at pitch
center, already PFF's convention), and ``player_id`` the **real PFF player id**
when a roster is supplied (so identities flow through to the credit tables).
"""

from __future__ import annotations

import bz2
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from defcon.config import Config, load_config

__all__ = [
    "PFFMetadata",
    "load_pff_metadata",
    "load_pff_rosters",
    "load_pff_tracking",
    "pff_goalkeepers",
    "pff_identity_table",
    "pff_game_paths",
]

# PFF positionGroupType -> coarse position bucket used for the market-value study.
_POS_BUCKET = {
    "GK": "GK",
    "LB": "DF", "RB": "DF", "LCB": "DF", "RCB": "DF", "CB": "DF", "MCB": "DF",
    "LWB": "DF", "RWB": "DF",
    "LDM": "MF", "RDM": "MF", "CDM": "MF", "DM": "MF", "LCM": "MF", "RCM": "MF", "CM": "MF",
    "LAM": "MF", "RAM": "MF", "CAM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
    "LW": "FW", "RW": "FW", "LF": "FW", "RF": "FW", "CF": "FW", "ST": "FW", "SS": "FW",
}


@dataclass(frozen=True)
class PFFMetadata:
    game_id: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    fps: float
    pitch_length: float
    pitch_width: float
    home_team_start_left: bool
    date: str = ""

    def side_of_team(self, team_id: str) -> str | None:
        """Return 'home'/'away' for a PFF team id, else None."""
        if str(team_id) == self.home_team_id:
            return "home"
        if str(team_id) == self.away_team_id:
            return "away"
        return None


def pff_game_paths(game_dir: str | Path, game_id: int | str) -> dict[str, Path]:
    """Return the three PFF file paths for a game directory.

    Accepts either ``{game_id}.jsonl`` or ``{game_id}.jsonl.bz2`` for tracking.
    """
    game_dir = Path(game_dir)
    raw = game_dir / f"{game_id}.jsonl.bz2"
    if not raw.exists():
        raw = game_dir / f"{game_id}.jsonl"
    return {
        "tracking": raw,
        "metadata": game_dir / f"metadata_{game_id}.json",
        "rosters": game_dir / f"rosters_{game_id}.json",
    }


def load_pff_metadata(path: str | Path) -> PFFMetadata:
    """Parse a PFF ``metadata_{game_id}.json`` file."""
    obj = json.loads(Path(path).read_text())
    # PFF wraps the record in a single-element list.
    rec = obj[0] if isinstance(obj, list) else obj
    pitch = (rec.get("stadium", {}) or {}).get("pitches", [{}])
    pitch = pitch[0] if pitch else {}
    return PFFMetadata(
        game_id=str(rec.get("id", "")),
        home_team_id=str(rec["homeTeam"]["id"]),
        home_team_name=rec["homeTeam"]["name"],
        away_team_id=str(rec["awayTeam"]["id"]),
        away_team_name=rec["awayTeam"]["name"],
        fps=float(rec.get("fps", 29.97)),
        pitch_length=float(pitch.get("length", 105.0)),
        pitch_width=float(pitch.get("width", 68.0)),
        home_team_start_left=bool(rec.get("homeTeamStartLeft", True)),
        date=str(rec.get("date", "")),
    )


def load_pff_rosters(path: str | Path, metadata: PFFMetadata | None = None) -> pd.DataFrame:
    """Parse a PFF ``rosters_{game_id}.json`` into a tidy roster table.

    Columns: ``player_id, name, team_id, team_name, shirt, position,
    position_group, started, side`` (``side`` = home/away when ``metadata`` is
    given, else NaN). This is the bridge from anonymous jersey numbers to the
    real identities the market-value join needs.
    """
    rows = json.loads(Path(path).read_text())
    recs = []
    for e in rows:
        player = e.get("player", {}) or {}
        team = e.get("team", {}) or {}
        pos_group = e.get("positionGroupType") or ""
        team_id = str(team.get("id", ""))
        recs.append(
            {
                "player_id": str(player.get("id", "")),
                "name": player.get("nickname") or player.get("name") or "",
                "team_id": team_id,
                "team_name": team.get("name", ""),
                "shirt": str(e.get("shirtNumber", "")),
                "position_group": pos_group,
                "position": _POS_BUCKET.get(pos_group.upper(), "MF" if pos_group else ""),
                "started": bool(e.get("started", False)),
                "side": metadata.side_of_team(team_id) if metadata else None,
            }
        )
    return pd.DataFrame(recs)


def pff_goalkeepers(rosters: pd.DataFrame, periods=(1, 2, 3, 4)) -> dict[tuple[str, int], str]:
    """Map (side, period) -> starting goalkeeper player_id from the roster.

    Uses the roster's position group (``GK``) rather than the deepest-player
    heuristic, since PFF gives us true positions. Falls back to nothing if a
    side has no rostered GK (the pipeline's geometric heuristic still applies).
    """
    out: dict[tuple[str, int], str] = {}
    gks = rosters[(rosters["position_group"].str.upper() == "GK")]
    for side in ("home", "away"):
        sub = gks[gks["side"] == side]
        if len(sub) == 0:
            continue
        started = sub[sub["started"]]
        pid = (started if len(started) else sub).iloc[0]["player_id"]
        for p in periods:
            out[(side, int(p))] = str(pid)
    return out


def _open_maybe_bz2(path: Path):
    if str(path).endswith(".bz2"):
        return bz2.open(path, "rt")
    return open(path)


def _pick(frame: dict, key_smoothed: str, key_raw: str, smoothed: bool):
    """Prefer a frame's smoothed field, falling back to the raw one."""
    if smoothed:
        val = frame.get(key_smoothed)
        if val:
            return val
    return frame.get(key_raw) or []


def _player_records(players, side, id_map):
    """Yield ``(side, player_id, x, y, z)`` rows from PFF player dicts for one side."""
    for p in players:
        jersey = str(p.get("jerseyNum", p.get("jersey", "")))
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        pid = id_map.get((side, jersey)) or f"{'H' if side == 'home' else 'A'}{jersey}"
        yield side, pid, float(x), float(y), np.nan


def load_pff_tracking(
    raw_data: str | Path,
    metadata: PFFMetadata,
    rosters: pd.DataFrame | None = None,
    smoothed: bool = True,
    cfg: Config | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load PFF tracking JSONL into the unified long schema.

    ``rosters`` (if given) maps (side, jersey) -> real player id so ``player_id``
    carries identities. ``smoothed`` prefers PFF's smoothed positions and falls
    back to the raw ones per frame. Frames with no players are dropped.
    """
    cfg = cfg or load_config()
    raw_data = Path(raw_data)

    id_map: dict[tuple[str, str], str] = {}
    if rosters is not None and "side" in rosters and rosters["side"].notna().any():
        for _, r in rosters.iterrows():
            if r["side"] in ("home", "away"):
                id_map[(r["side"], str(r["shirt"]))] = str(r["player_id"])

    records: list[tuple] = []
    n = 0
    with _open_maybe_bz2(raw_data) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fr = json.loads(line)
            period = int(fr.get("period", 1))
            frame = int(fr.get("frameNum", n))
            time_s = float(fr.get("periodElapsedTime", fr.get("periodGameClockTime", 0.0)) or 0.0)

            home = _pick(fr, "homePlayersSmoothed", "homePlayers", smoothed)
            away = _pick(fr, "awayPlayersSmoothed", "awayPlayers", smoothed)
            # PFF stores the smoothed ball as a single dict but the raw ball as a
            # list; normalize both to a list.
            balls = _pick(fr, "ballsSmoothed", "balls", smoothed)
            if isinstance(balls, dict):
                balls = [balls]

            frame_rows: list[tuple] = []
            for side, players in (("home", home), ("away", away)):
                for team, pid, x, y, z in _player_records(players, side, id_map):
                    frame_rows.append((period, frame, time_s, team, pid, x, y, z))
            if balls:
                b = balls[0]
                bx, by = b.get("x"), b.get("y")
                if bx is not None and by is not None:
                    bz = b.get("z")
                    frame_rows.append(
                        (period, frame, time_s, "ball", "ball", float(bx), float(by),
                         float(bz) if bz is not None else np.nan)
                    )

            if not any(r[3] in ("home", "away") for r in frame_rows):
                continue
            records.extend(frame_rows)
            n += 1
            if limit is not None and n >= limit:
                break

    df = pd.DataFrame(
        records,
        columns=["period", "frame", "time_s", "team", "player_id", "x", "y", "z"],
    )
    # Some PFF files emit a frameNum more than once (the WC-Final file repeats ~61
    # event-anchored frames 32x). Left in, these stack a single instant into a
    # multi-hundred-row "frame" that quietly breaks anything keyed on (period,
    # frame). Collapse to one row per player per frame.
    df = df.drop_duplicates(subset=["period", "frame", "team", "player_id"], keep="first")
    df = df.sort_values(["period", "frame", "team", "player_id"]).reset_index(drop=True)
    return df


def pff_identity_table(rosters: pd.DataFrame) -> pd.DataFrame:
    """Per-player identity table for the market-value join (task 7.3).

    Columns: ``player_id, name, team_name, position, side``. One row per
    rostered player — the key the credit aggregation joins onto to attach a
    real name (and then a Transfermarkt market value).
    """
    cols = ["player_id", "name", "team_name", "position", "side"]
    out = rosters[cols].drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    return out
