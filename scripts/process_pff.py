#!/usr/bin/env python
"""Run the DEFCON credit pipeline across PFF World Cup games (task 7.3).

For each game: parse the PFF event freeze-frames -> actions + per-action
tracking, run EPV + the credit engine, and accumulate per-player defensive
credit. Aggregates to a per-90 table with **real player names**, prints the
defensive-credit leaderboards (paper's Tables 5-7), and — if Transfermarkt
market values are present — runs the market-value correlation (the headline
"better prevent than tackle" study).

NOTE: EPV's component models are trained on Metrica, so applying them to PFF is a
cross-provider domain shift. Credit magnitudes here are approximate; the point is
the *ranking* pattern and the market-value correlation, not exact values.

Usage:
    python scripts/process_pff.py --games 10503 10504 10511 10513 10514 10515 10517
"""

from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from defcon import load_config
from defcon.credit.aggregate import credits_to_items, season_rollup
from defcon.credit.engine import CreditEngine
from defcon.data.pff import load_pff_metadata, load_pff_rosters, pff_game_paths, pff_identity_table
from defcon.data.pff_events import parse_pff_events
from defcon.epv.epv import EPVEngine

warnings.filterwarnings("ignore")

PILOT = [10503, 10504, 10511, 10513, 10514, 10515, 10517]


def minutes_from_appearances(tracking: pd.DataFrame) -> dict[str, float]:
    """Minutes per player from their freeze-frame appearance window, per period.

    Robust to PFF's sparse per-action frames and to substitutions: a player's
    active window in a period is (last - first) event time they appear in; summed
    across periods and converted to minutes.
    """
    pl = tracking[tracking["team"].isin(["home", "away"])]
    span = pl.groupby(["player_id", "period"])["time_s"].agg(["min", "max"])
    span["dur"] = span["max"] - span["min"]
    return (span.groupby("player_id")["dur"].sum() / 60.0).to_dict()


def process_game(gid: int, cfg, epv, pff_dir, events_dir):
    paths = pff_game_paths(pff_dir, gid)
    md = load_pff_metadata(paths["metadata"])
    rosters = load_pff_rosters(paths["rosters"], md)
    events = json.load(open(events_dir / f"{gid}.json"))

    actions, tracking = parse_pff_events(events, md, f"pff_{gid}")
    credits = CreditEngine(epv, cfg).process_match(actions, tracking)

    minutes = minutes_from_appearances(tracking)
    items = credits_to_items(credits)
    player_team = dict(zip(rosters["player_id"], rosters["team_name"], strict=False))
    fixture = f"{md.home_team_name} vs {md.away_team_name}"
    return items, minutes, player_team, rosters, fixture, len(actions)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, nargs="+", default=PILOT)
    ap.add_argument("--min-minutes", type=float, default=120.0,
                    help="filter players below this many total minutes")
    args = ap.parse_args()

    cfg = load_config()
    pff_dir = cfg.path("tracking_dir") / "pff"
    events_dir = cfg.path("data_raw") / "pff_events"
    epv = EPVEngine.from_checkpoints(cfg)

    matches, player_team, id_frames = [], {}, []
    for gid in args.games:
        try:
            items, minutes, pteam, rosters, fixture, n_act = process_game(
                gid, cfg, epv, pff_dir, events_dir)
        except FileNotFoundError as e:
            print(f"[skip] game {gid}: missing {e.filename}")
            continue
        matches.append((items, minutes))
        player_team.update(pteam)
        id_frames.append(pff_identity_table(rosters))
        print(f"[game {gid}] {fixture:28s} {n_act:4d} actions, "
              f"{len(minutes)} players, {len(items)} credits")

    if not matches:
        print("[error] no games processed — check data/raw/tracking/pff and data/raw/pff_events")
        return 1

    table = season_rollup(matches, player_team, min_minutes=args.min_minutes)
    identities = pd.concat(id_frames).drop_duplicates("player_id")
    table = table.merge(
        identities[["player_id", "name", "position"]].rename(columns={"player_id": "player"}),
        on="player", how="left")

    out = cfg.path("data_processed") / "pff_credit_table.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    print(f"\n[saved] per-player credit table ({len(table)} players >= "
          f"{args.min_minutes:.0f} min) -> {out}")

    _leaderboards(table)
    _market_value(table, cfg)
    return 0


def _fmt(df, cols):
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:6.2f}")


def _leaderboards(table: pd.DataFrame) -> None:
    show = ["name", "position", "team", "minutes", "net_p90", "concede_p90", "intercept_p90"]
    show = [c for c in show if c in table.columns]
    print("\n=== WC pilot — defensive-credit leaderboards (per 90; domain-shift caveat) ===")
    print("\nTop 12 by NET credit / 90:")
    print(_fmt(table.sort_values("net_p90", ascending=False).head(12), show))
    print("\nTop 10 by INTERCEPT / 90 (visible ball-winning):")
    print(_fmt(table.sort_values("intercept_p90", ascending=False).head(10), show))
    print("\nTop 10 by DETER + DISTURB / 90 (prevention):")
    t = table.assign(prevent_p90=table["deter_p90"] + table["disturb_p90"])
    print(_fmt(t.sort_values("prevent_p90", ascending=False).head(10),
              [c for c in show if c != "intercept_p90"] + ["prevent_p90"]))


def _market_value(table: pd.DataFrame, cfg) -> None:
    from defcon.eval.market_value import (
        add_log_value,
        attach_transfermarkt,
        correlate_value,
        plot_value_scatter,
    )

    mv_dir = cfg.path("data_raw") / "market_value"
    players = mv_dir / "players.csv"
    valuations = mv_dir / "player_valuations.csv"
    if not (players.exists() and valuations.exists()):
        print("\n[market value] pending — drop Transfermarkt players.csv + "
              "player_valuations.csv into data/raw/market_value/ to run the correlation.")
        return

    joined = attach_transfermarkt(table, players, valuations)
    joined["prevent_p90"] = joined["deter_p90"] + joined["disturb_p90"]
    matched = joined["market_value"].notna().sum()
    out = cfg.path("data_processed") / "pff_market_value.csv"
    joined.to_csv(out, index=False)
    print(f"\n[market value] matched {matched}/{len(joined)} players to Transfermarkt "
          f"values (as of 2022-11-20) -> {out}")

    metrics = ["intercept_p90", "disturb_p90", "deter_p90", "concede_p90", "prevent_p90", "net_p90"]
    res = correlate_value(joined, metrics=tuple(metrics))
    cols = ["group", "n"] + [f"r_{m}" for m in metrics]
    print("\n=== 'Better prevent than tackle' — Pearson r of credit/90 vs log market value ===")
    print("    (Metrica-trained EPV applied to PFF: directional, small-sample; not the paper's 564 matches)")
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print("\n  Read: for DEFENDERS, raw Intercept tracks value weakly while prevention "
          "(Disturb/Deter) tracks it more — the paper's core pattern.")

    from defcon.config import repo_root
    fig = repo_root() / "docs" / "img" / "pff_market_value.png"
    try:
        plot_value_scatter(add_log_value(joined), metric="prevent_p90",
                           out_path=str(fig), annotate=True)
        print(f"[market value] scatter -> {fig}")
    except Exception as e:  # plotting is optional
        print(f"[market value] scatter skipped ({e})")


if __name__ == "__main__":
    raise SystemExit(main())
