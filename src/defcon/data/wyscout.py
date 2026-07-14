"""Download and load the Wyscout open soccer-logs dataset (Pappalardo et al.,
*Scientific Data* 2019), used to train the UxG expected-goal model (task 3.6).

The dataset lives in a figshare collection. Download URLs are resolved through
the figshare API at runtime (so we never hard-code fragile file IDs), then
cached locally under ``data/raw/wyscout/``.

Collection: https://figshare.com/collections/Soccer_match_event_dataset/4415000

Event schema (per competition JSON): each event has ``eventId``/``eventName``,
``subEventId``/``subEventName``, ``tags`` (list of ``{"id": int}``),
``positions`` (list of ``{"x": 0-100, "y": 0-100}`` — pitch percentage,
attacking toward x=100, goal center at (100, 50)), plus ``matchId``,
``teamId``, ``playerId``, ``matchPeriod``, ``eventSec``.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

__all__ = [
    "FIGSHARE_ARTICLES",
    "WYSCOUT_COMPETITIONS",
    "download_wyscout",
    "load_shots",
    "resolve_figshare_files",
]

# --- Wyscout event/tag constants ------------------------------------------- #
EVENT_SHOT = 10
EVENT_FREE_KICK = 3
SUBEVENT_FREE_KICK_SHOT = 33
SUBEVENT_PENALTY = 35
TAG_GOAL = 101
TAG_HEADER = 403  # head/body

FIGSHARE_API = "https://api.figshare.com/v2"

# Logical name -> figshare article id (from collection 4415000).
FIGSHARE_ARTICLES: dict[str, int] = {
    "events": 7770599,
    "matches": 7770422,
    "players": 7765196,
    "teams": 7765310,
    "competitions": 7765316,
    "eventid2name": 11743836,
    "tags2name": 11743818,
}

# Competition JSONs bundled inside events.zip.
WYSCOUT_COMPETITIONS = [
    "England",
    "France",
    "Germany",
    "Italy",
    "Spain",
    "European_Championship",
    "World_Cup",
]


@dataclass(frozen=True)
class FigshareFile:
    name: str
    size: int
    download_url: str


def resolve_figshare_files(article_id: int, timeout: float = 30.0) -> list[FigshareFile]:
    """Return the downloadable files for a figshare article via its API."""
    resp = requests.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return [
        FigshareFile(name=f["name"], size=f["size"], download_url=f["download_url"])
        for f in data.get("files", [])
    ]


def _download_file(url: str, dest: Path, timeout: float = 60.0, chunk: int = 1 << 20) -> Path:
    """Stream a URL to ``dest`` with a progress bar; skip if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
                bar.update(len(block))
    tmp.replace(dest)
    return dest


def download_wyscout(
    dest_dir: str | Path,
    items: tuple[str, ...] = (
        "events",
        "players",
        "teams",
        "competitions",
        "eventid2name",
        "tags2name",
    ),
    extract_events: bool = True,
) -> Path:
    """Download the requested Wyscout artifacts into ``dest_dir``.

    ``events`` is delivered as ``events.zip`` (~77 MB) and, when
    ``extract_events`` is set, unzipped into per-competition JSON files.
    Everything is idempotent: existing files are skipped.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for item in items:
        if item not in FIGSHARE_ARTICLES:
            raise ValueError(f"Unknown Wyscout item '{item}'. Known: {list(FIGSHARE_ARTICLES)}")
        files = resolve_figshare_files(FIGSHARE_ARTICLES[item])
        if not files:
            raise RuntimeError(f"No files found for figshare article of '{item}'")
        for f in files:
            out = dest / f.name
            print(f"[wyscout] {item}: {f.name} ({f.size / 1e6:.1f} MB)")
            _download_file(f.download_url, out)

    if extract_events:
        zip_path = dest / "events.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    target = dest / Path(member).name
                    if target.exists() and target.stat().st_size > 0:
                        continue
                    print(f"[wyscout] extracting {member}")
                    with zf.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())

    return dest


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _has_tag(event: dict, tag_id: int) -> bool:
    return any(t.get("id") == tag_id for t in event.get("tags", []))


def _is_shot(event: dict) -> bool:
    """Open-play shot, free-kick shot, or penalty."""
    if event.get("eventId") == EVENT_SHOT:
        return True
    if event.get("eventId") == EVENT_FREE_KICK and event.get("subEventId") in (
        SUBEVENT_FREE_KICK_SHOT,
        SUBEVENT_PENALTY,
    ):
        return True
    return False


def load_shots(
    wyscout_dir: str | Path,
    competitions: list[str] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> pd.DataFrame:
    """Load all shots from the Wyscout event JSONs into a flat DataFrame.

    Wyscout positions are pitch percentages (0-100) already oriented so the
    shooting team attacks toward x=100; no direction flip is needed. Returns one
    row per shot with raw location, meters location, and the flags/label the UxG
    model consumes: ``x_m, y_m, is_set_piece, is_header, is_goal``.
    """
    wyscout_dir = Path(wyscout_dir)
    if competitions is None:
        competitions = [
            c for c in WYSCOUT_COMPETITIONS if (wyscout_dir / f"events_{c}.json").exists()
        ]
    if not competitions:
        raise FileNotFoundError(
            f"No Wyscout event files found in {wyscout_dir}. Run download_wyscout() first."
        )

    rows: list[dict] = []
    for comp in competitions:
        path = wyscout_dir / f"events_{comp}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        for e in _load_json(path):
            if not _is_shot(e):
                continue
            positions = e.get("positions") or []
            if not positions:
                continue
            x_pct = positions[0].get("x")
            y_pct = positions[0].get("y")
            if x_pct is None or y_pct is None:
                continue
            is_set_piece = int(e.get("eventId") == EVENT_FREE_KICK)
            rows.append(
                {
                    "match_id": e.get("matchId"),
                    "competition": comp,
                    "team_id": e.get("teamId"),
                    "player_id": e.get("playerId"),
                    "period": e.get("matchPeriod"),
                    "event_sec": e.get("eventSec"),
                    "x_pct": x_pct,
                    "y_pct": y_pct,
                    "x_m": x_pct / 100.0 * pitch_length,
                    "y_m": y_pct / 100.0 * pitch_width,
                    "is_set_piece": is_set_piece,
                    "is_header": int(_has_tag(e, TAG_HEADER)),
                    "is_goal": int(_has_tag(e, TAG_GOAL)),
                }
            )

    return pd.DataFrame(rows)
