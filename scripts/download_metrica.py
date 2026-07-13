#!/usr/bin/env python
"""Download the Metrica Sports sample data (Games 1 & 2, CSV format).

Fetches the raw tracking + event CSVs from the public metrica-sports/sample-data
repository into data/raw/tracking/metrica/. Game 3 uses a different (EPTS) format
and is not used by this pipeline.

Usage:
    python scripts/download_metrica.py
    python scripts/download_metrica.py --games 1
"""

from __future__ import annotations

import argparse

import requests

from defcon import load_config

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"


def _download(url: str, dest, timeout: float = 120.0) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[metrica] skip (exists) {dest.name}")
        return
    print(f"[metrica] downloading {dest.name} ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    args = ap.parse_args()

    cfg = load_config()
    dest_dir = cfg.path("tracking_dir") / "metrica"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for g in args.games:
        stem = f"Sample_Game_{g}"
        for suffix in (
            "RawTrackingData_Home_Team.csv",
            "RawTrackingData_Away_Team.csv",
            "RawEventsData.csv",
        ):
            fname = f"{stem}_{suffix}"
            _download(f"{BASE}/{stem}/{fname}", dest_dir / fname)
    print(f"[done] Metrica data in {dest_dir}")


if __name__ == "__main__":
    main()
