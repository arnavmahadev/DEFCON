#!/usr/bin/env python
"""Download the Wyscout open event dataset into data/raw/wyscout/.

Usage:
    python scripts/download_wyscout.py                 # everything
    python scripts/download_wyscout.py --items events players
"""

from __future__ import annotations

import argparse

from defcon import load_config
from defcon.data.wyscout import FIGSHARE_ARTICLES, download_wyscout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        nargs="+",
        default=["events", "players", "teams", "competitions", "eventid2name", "tags2name"],
        choices=list(FIGSHARE_ARTICLES),
        help="Which Wyscout artifacts to download.",
    )
    parser.add_argument("--dest", default=None, help="Destination dir (default: config wyscout_dir).")
    args = parser.parse_args()

    cfg = load_config()
    dest = args.dest or cfg.path("wyscout_dir")
    download_wyscout(dest, items=tuple(args.items))
    print(f"[done] Wyscout data in {dest}")


if __name__ == "__main__":
    main()
