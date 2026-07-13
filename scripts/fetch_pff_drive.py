#!/usr/bin/env python
"""Fetch the PFF FC 2022 World Cup data from its shared Google Drive folder.

PFF grants access via a Google Drive folder laid out as::

    Metadata/{game_id}.json
    Rosters/{game_id}.json
    Tracking Data/{game_id}.jsonl.bz2      (~40 MB each, 64 games)
    Event Data/<date>/{game_id}.json
    competitions.csv, players.csv

Browser downloads of the big tracking files often fail silently (Drive's
"can't scan for viruses" confirmation gets popup-blocked). ``gdown`` handles that
confirmation automatically, so this script pulls files reliably from the CLI.

It lands everything in the flat naming our loader expects
(``data/raw/tracking/pff/{metadata_,rosters_,}{id}...``) plus events under
``data/raw/pff_events/``.

Usage:
    # tiny files for every game (a few MB) + list all fixtures
    python scripts/fetch_pff_drive.py --url <folder_url> --small --list

    # tracking + events for a chosen subset
    python scripts/fetch_pff_drive.py --url <folder_url> --games 10517 10515 10514

    # everything (~2.5 GB) — usually unnecessary
    python scripts/fetch_pff_drive.py --url <folder_url> --all
"""

from __future__ import annotations

import argparse
import json
import re
import time

from defcon import load_config

DEFAULT_URL = "https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"


def resolve_tree(url: str):
    """Return [(path, drive_id), ...] for every file in the folder (no download)."""
    import gdown

    files = gdown.download_folder(url=url, skip_download=True, quiet=True, use_cookies=False)
    return [(f.path, f.id) for f in files]


def _download_resilient(gdown, fid: str, out: str, quiet: bool, retries: int = 6,
                        base_sleep: float = 20.0) -> bool:
    """Download one Drive file, backing off on Drive's 'too many accesses' throttle."""
    for attempt in range(retries):
        try:
            gdown.download(id=fid, output=out, quiet=quiet)
            return True
        except Exception as e:  # gdown raises FileURLRetrievalError on throttle
            wait = base_sleep * (attempt + 1)
            msg = str(e).splitlines()[0][:80]
            print(f"[pff]   throttled ({msg}); waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    print(f"[pff]   GAVE UP on {out} — grab it in-browser (Download anyway popup)")
    return False


def _game_id(path: str) -> str | None:
    m = re.search(r"(\d+)\.(?:json|jsonl\.bz2)$", path.split("/")[-1])
    return m.group(1) if m else None


def plan(tree, pff_dir, events_dir):
    """Map Drive entries -> local targets. Dedupes Event Data across dated folders."""
    targets: dict[str, tuple[str, str]] = {}   # drive_id -> (local_path, kind)
    fixtures: dict[str, str] = {}
    event_pick: dict[str, tuple[str, str]] = {}  # game_id -> (date_folder, drive_id)

    for path, fid in tree:
        gid = _game_id(path)
        if path.startswith("Metadata/") and gid:
            targets[fid] = (str(pff_dir / f"metadata_{gid}.json"), "meta")
        elif path.startswith("Rosters/") and gid:
            targets[fid] = (str(pff_dir / f"rosters_{gid}.json"), "roster")
        elif path.startswith("Tracking Data/") and gid:
            targets[fid] = (str(pff_dir / f"{gid}.jsonl.bz2"), "tracking")
        elif path.startswith("Event Data/") and gid:
            # keep the entry from the latest dated subfolder
            parts = path.split("/")
            date_folder = parts[1] if len(parts) > 2 else ""
            if gid not in event_pick or date_folder > event_pick[gid][0]:
                event_pick[gid] = (date_folder, fid)
        elif path.endswith(".csv"):
            targets[fid] = (str(pff_dir / path.split("/")[-1]), "csv")

    for gid, (_, fid) in event_pick.items():
        targets[fid] = (str(events_dir / f"{gid}.json"), "event")
    return targets, fixtures


def fixtures_from_disk(pff_dir):
    """Read downloaded metadata to map game_id -> 'Home vs Away (date)'."""
    out = {}
    for f in sorted(pff_dir.glob("metadata_*.json")):
        gid = f.stem.replace("metadata_", "")
        obj = json.loads(f.read_text())
        rec = obj[0] if isinstance(obj, list) else obj
        out[gid] = f"{rec['homeTeam']['name']} vs {rec['awayTeam']['name']} ({rec.get('date','')[:10]})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="shared Drive folder URL")
    ap.add_argument("--small", action="store_true", help="download all metadata/rosters/csv (tiny)")
    ap.add_argument("--games", type=int, nargs="+", help="game ids to fetch tracking + events for")
    ap.add_argument("--all", action="store_true", help="download everything (~2.5 GB)")
    ap.add_argument("--list", action="store_true", help="print fixtures (from downloaded metadata)")
    ap.add_argument("--pace", type=float, default=5.0, help="seconds to sleep between files")
    args = ap.parse_args()

    import gdown

    cfg = load_config()
    pff_dir = cfg.path("tracking_dir") / "pff"
    events_dir = cfg.path("data_raw") / "pff_events"
    pff_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    print("[pff] resolving Drive folder ...")
    tree = resolve_tree(args.url)
    targets, _ = plan(tree, pff_dir, events_dir)

    want_games = set(map(str, args.games or []))

    def wanted(kind: str, local_path: str) -> bool:
        if args.all:
            return True
        gid_m = re.search(r"(\d+)", local_path.split("/")[-1])
        gid = gid_m.group(1) if gid_m else ""
        if kind == "csv":
            return args.small
        if kind in ("meta", "roster"):
            return args.small or gid in want_games
        if kind in ("tracking", "event"):
            return gid in want_games
        return False

    todo = [(fid, lp, kind) for fid, (lp, kind) in targets.items() if wanted(kind, lp)]
    print(f"[pff] {len(todo)} files to download "
          f"({sum(k in ('tracking',) for _,_,k in todo)} tracking @ ~40 MB each)")
    from pathlib import Path
    failed = []
    for i, (fid, lp, kind) in enumerate(todo):
        if Path(lp).exists() and Path(lp).stat().st_size > 0:
            print(f"[pff] skip (exists) {Path(lp).name}")
            continue
        print(f"[pff] [{i + 1}/{len(todo)}] {kind}: {Path(lp).name}")
        ok = _download_resilient(gdown, fid, lp, quiet=(kind != "tracking"))
        if not ok:
            failed.append((fid, lp))
        time.sleep(args.pace)  # pace requests to avoid the throttle
    if failed:
        print(f"\n[pff] {len(failed)} file(s) failed (Drive throttle). Retry later or in-browser:")
        for fid, lp in failed:
            print(f"   https://drive.google.com/uc?id={fid}  ->  {lp}")

    if args.list or args.small:
        fx = fixtures_from_disk(pff_dir)
        print(f"\n[pff] {len(fx)} fixtures available:")
        for gid in sorted(fx, key=int):
            print(f"   {gid}  {fx[gid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
