#!/usr/bin/env python
"""Access + validate PFF FC 2022 World Cup tracking data (task 7.3).

The *full* 64-game PFF FC dataset is **free but gated behind a request form** — it
cannot be fetched non-interactively. This script therefore does three things:

1. ``--instructions`` (default): print how to request the free dataset.
2. ``--sample``: download the small public kloppy PFF fixture (the 2022 World Cup
   Final, Argentina vs France) into data/raw/tracking/pff/ so the loader is
   immediately runnable on *real* PFF-format data with real identities.
3. ``--validate DIR``: verify a local PFF directory parses, and export the
   per-player identity table (real names + positions) used for the market-value
   join, to data/processed/pff_identities_{game}.csv.

Usage:
    python scripts/download_pff.py                 # access instructions
    python scripts/download_pff.py --sample        # fetch the WC-Final sample
    python scripts/download_pff.py --validate data/raw/tracking/pff --game 10517
"""

from __future__ import annotations

import argparse
import sys

import requests

from defcon import load_config
from defcon.data.pff import (
    load_pff_metadata,
    load_pff_rosters,
    load_pff_tracking,
    pff_game_paths,
    pff_identity_table,
)

# kloppy ships two real PFF-format sample games in its test fixtures.
KLOPPY_FILES = "https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/tests/files"
SAMPLE_GAMES = {
    10517: "2022 World Cup Final — Argentina vs France",
    3812: "2022 World Cup — sample game",
}

REQUEST_URL = "https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset"

INSTRUCTIONS = f"""
PFF FC 2022 World Cup data — free, but requires a one-time access request.

  1. Open:  {REQUEST_URL}
  2. Fill out the "Get Free Access" form (name + email + intended use).
  3. PFF emails a download link (or contact fchelp@pff.com). The bundle contains,
     per game:
         {{game_id}}.jsonl.bz2      tracking (29.97 Hz, all players + ball)
         metadata_{{game_id}}.json  teams, pitch, kick-off side
         rosters_{{game_id}}.json   jersey -> real player id + name + position
     plus consolidated events.json and players/rosters/competitions CSVs.
  4. Drop the per-game files into:  data/raw/tracking/pff/
  5. Validate with:  python scripts/download_pff.py --validate data/raw/tracking/pff --game <id>

Why PFF: unlike the anonymized Metrica sample, PFF carries real player identities,
which is what the market-value validation (the paper's headline finding) needs to
join onto Transfermarkt values.

Tip: run  python scripts/download_pff.py --sample  to pull the public WC-Final
sample right now and see the loader work on real PFF data.
"""


def _download(url: str, dest, timeout: float = 120.0) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[pff] skip (exists) {dest.name}")
        return
    print(f"[pff] downloading {dest.name} ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def fetch_sample(dest_dir) -> None:
    """Fetch the public kloppy PFF sample games into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for game in SAMPLE_GAMES:
        # kloppy names them pff_{kind}_{id} / pff_{id}.jsonl; we store them under
        # the {game}.jsonl / metadata_{game}.json / rosters_{game}.json convention.
        _download(f"{KLOPPY_FILES}/pff_{game}.jsonl", dest_dir / f"{game}.jsonl")
        _download(f"{KLOPPY_FILES}/pff_metadata_{game}.json", dest_dir / f"metadata_{game}.json")
        _download(f"{KLOPPY_FILES}/pff_rosters_{game}.json", dest_dir / f"rosters_{game}.json")
    print(f"[done] PFF sample in {dest_dir}")
    print("       Note: these are short broadcast slices (not full matches) — enough")
    print("       to exercise the loader, not to run the full market-value study.")


def validate(game_dir, game_id: int, cfg) -> int:
    """Parse a local PFF game and export its identity table. Returns exit code."""
    paths = pff_game_paths(game_dir, game_id)
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"[error] missing {missing} for game {game_id} in {game_dir}", file=sys.stderr)
        return 1

    md = load_pff_metadata(paths["metadata"])
    rosters = load_pff_rosters(paths["rosters"], md)
    tracking = load_pff_tracking(paths["tracking"], md, rosters)

    print(f"[ok] {md.home_team_name} vs {md.away_team_name}  ({md.date})")
    periods = sorted(int(p) for p in tracking["period"].unique())
    print(f"     tracking: {len(tracking):,} rows | {tracking['frame'].nunique()} frames "
          f"| periods {periods} | fps {md.fps}")
    print(f"     roster:   {len(rosters)} players | "
          f"positions {rosters['position'].value_counts().to_dict()}")

    idt = pff_identity_table(rosters)
    out = cfg.path("data_processed") / f"pff_identities_{game_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    idt.to_csv(out, index=False)
    print(f"[export] identity table ({len(idt)} players) -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="fetch the public kloppy WC-Final sample")
    ap.add_argument("--validate", metavar="DIR", help="validate a local PFF directory")
    ap.add_argument("--game", type=int, default=10517, help="game id to validate")
    ap.add_argument("--instructions", action="store_true", help="print access instructions")
    args = ap.parse_args()

    cfg = load_config()
    if args.sample:
        fetch_sample(cfg.path("tracking_dir") / "pff")
        return 0
    if args.validate:
        return validate(args.validate, args.game, cfg)
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
