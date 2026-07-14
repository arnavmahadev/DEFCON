#!/usr/bin/env python
"""Honest end-to-end demo of DEFCON on *real* PFF World Cup Final tracking.

Loads the public PFF sample (2022 World Cup Final, Argentina vs France), picks a
live frame, and renders the **defender-responsibility** figure with **real player
names** on it. It uses DEFCON's model-free geometric responsibility prior (the
CreditEngine default) — not the Metrica-trained GNN, which would be a domain-shift
overclaim — so the figure is an honest illustration of the framework on real,
identified players rather than a trained-model result.

Output: docs/img/pff_wc_final_responsibility.png

Needs the public PFF World Cup Final sample first:
    python scripts/download_pff.py --sample     # -> data/raw/tracking/pff/

Usage:
    python scripts/pff_demo.py                              # WC Final (game 10517)
    python scripts/pff_demo.py --dir tests/data/pff --game 9001   # synthetic fixture
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from defcon import load_config
from defcon.credit.engine import CreditEngine
from defcon.data.metrica import infer_playing_direction
from defcon.data.pff import (
    load_pff_metadata,
    load_pff_rosters,
    load_pff_tracking,
    pff_game_paths,
    pff_goalkeepers,
)
from defcon.features.state import graph_state_from_action
from defcon.viz.pitch import draw_pitch

warnings.filterwarnings("ignore")


def _surname(name: str) -> str:
    return name.split()[-1] if name else ""


def _pick_frame(tracking: pd.DataFrame, directions: dict, n_scan: int = 500) -> tuple[int, int]:
    """Pick a full 11v11 frame in a genuine attacking moment.

    A meaningful defender-responsibility figure needs the possessing team pushed
    into the final third with a clear carrier and a forward pass option, not the
    random midfield scramble you get from "most players on screen". We restrict to
    frames that have the ball plus a full complement of players, scan a strided,
    deterministic sample of them, and keep the one where possession is cleanest and
    the ball is furthest advanced toward the attacking goal.
    """
    outfield = tracking[tracking["team"].isin(["home", "away"])]
    counts = outfield.groupby(["period", "frame"]).size().reset_index(name="n")
    have_ball = set(
        map(tuple, tracking[tracking["team"] == "ball"][["period", "frame"]].values.tolist())
    )
    # the modal complement is the normal 11v11 (22); max() would chase rare
    # substitution frames where both players are briefly on the pitch.
    full = int(counts["n"].mode().iloc[0])
    cand = counts[counts["n"] == full]
    cand = cand[[(p, f) in have_ball for p, f in zip(cand["period"], cand["frame"], strict=False)]]
    cand = cand.sort_values(["period", "frame"]).reset_index(drop=True)
    if len(cand) > n_scan:
        idx = np.unique(np.linspace(0, len(cand) - 1, n_scan).round().astype(int))
        cand = cand.iloc[idx]

    groups = tracking.groupby(["period", "frame"], sort=False)
    best_score, best = -np.inf, None
    for period, frame in cand[["period", "frame"]].itertuples(index=False):
        fr = groups.get_group((period, frame))
        ball = fr[fr["team"] == "ball"]
        players = fr[fr["team"].isin(["home", "away"])]
        if ball.empty or players.empty:
            continue
        bx, by = float(ball.iloc[0]["x"]), float(ball.iloc[0]["y"])
        d = np.hypot(players["x"].to_numpy() - bx, players["y"].to_numpy() - by)
        carrier = players.iloc[int(d.argmin())]
        att = carrier["team"]
        direction = directions.get((att, period))
        if direction is None:
            continue
        # possession is clear when the carrier owns the ball and the nearest
        # opponent is meaningfully further away.
        opp = players[players["team"] != att]
        d_opp = np.hypot(opp["x"] - bx, opp["y"] - by).min() if len(opp) else 0.0
        d_car = float(d.min())
        if d_car > 2.5 or (d_opp - d_car) < 1.0:
            continue
        mates = players[(players["team"] == att) & (players["player_id"] != carrier["player_id"])]
        if mates.empty:
            continue
        # keep the ball in the attacking third but off the byline: a build-up
        # reads clearly, a six-yard-box scramble is an unlabellable pile-up.
        adv = bx * direction
        if not (8.0 <= adv <= 38.0):
            continue
        # the option should be a genuine forward pass, ahead of the ball.
        if float((mates["x"] * direction).max()) <= adv:
            continue
        # reward spacing so labels don't collide; mild bonus for advancement.
        px, py = players["x"].to_numpy(), players["y"].to_numpy()
        dm = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
        np.fill_diagonal(dm, np.inf)
        spread = float(dm.min(axis=1).mean())
        score = spread + 0.15 * adv
        if score > best_score:
            best_score, best = score, (int(period), int(frame))

    if best is None:  # degenerate data: fall back to any full frame with a ball
        row = cand.iloc[0]
        return int(row["period"]), int(row["frame"])
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/raw/tracking/pff", help="PFF game directory")
    ap.add_argument("--game", type=int, default=10517)
    ap.add_argument("--out", default="docs/img/pff_wc_final_responsibility.png")
    args = ap.parse_args()

    cfg = load_config()
    paths = pff_game_paths(args.dir, args.game)
    if not paths["tracking"].exists():
        raise SystemExit(
            f"[pff_demo] no PFF game {args.game} in {args.dir}.\n"
            f"           Run:  python scripts/download_pff.py --sample\n"
            f"           (or point --dir at your PFF download; --game 9001 uses the synthetic fixture)."
        )
    md = load_pff_metadata(paths["metadata"])
    rosters = load_pff_rosters(paths["rosters"], md)
    tracking = load_pff_tracking(paths["tracking"], md, rosters)

    # velocities aren't needed for the positional responsibility prior — zero them
    # so graph_state_from_action has the columns it expects.
    for col in ("vx", "vy", "speed", "accel"):
        tracking[col] = 0.0

    names = dict(zip(rosters["player_id"].astype(str), rosters["name"], strict=False))
    directions = infer_playing_direction(tracking)
    goalkeepers = pff_goalkeepers(rosters)

    period, frame = _pick_frame(tracking, directions)
    fr = tracking[(tracking["period"] == period) & (tracking["frame"] == frame)]
    # guard: a real frame is one instant, so every player appears exactly once.
    dup = fr[fr["team"].isin(["home", "away"])].duplicated(["team", "player_id"]).sum()
    assert dup == 0, f"frame ({period},{frame}) has {dup} duplicate player rows — loader dedup regressed"
    ball = fr[fr["team"] == "ball"].iloc[0]

    # possession = team of the player nearest the ball; carrier = that player.
    players = fr[fr["team"].isin(["home", "away"])].copy()
    players["d_ball"] = np.hypot(players["x"] - ball["x"], players["y"] - ball["y"])
    carrier = players.sort_values("d_ball").iloc[0]
    att_team = carrier["team"]

    # option = the attacking team's most-advanced teammate (a plausible pass target)
    direction = directions[(att_team, period)]
    mates = players[(players["team"] == att_team) & (players["player_id"] != carrier["player_id"])]
    option_id = mates.assign(fwd=mates["x"] * direction).sort_values("fwd").iloc[-1]["player_id"]

    action = {
        "period": period, "frame": frame, "team": att_team,
        "player": carrier["player_id"], "inferred_receiver": option_id,
    }
    state = graph_state_from_action(tracking, action, directions, goalkeepers, cfg)
    assert state is not None, "could not build a GraphState for the chosen frame"

    engine = CreditEngine(epv_engine=None, cfg=cfg)
    option_idx = state.player_ids.index(str(option_id))
    resp = engine._geometric_responsibility(state, option_idx)  # {player_id: weight}
    weights = np.array([resp.get(pid, 0.0) for pid in state.player_ids])

    _render(state, weights, names, md, att_team, args.out, cfg)

    top = sorted(resp.items(), key=lambda kv: kv[1], reverse=True)[:3]
    att_name = md.home_team_name if att_team == "home" else md.away_team_name
    print(f"[demo] {md.home_team_name} vs {md.away_team_name} — {att_name} in possession")
    print(f"[demo] carrier: {names.get(str(carrier['player_id']),'?')}  "
          f"pass option: {names.get(str(option_id),'?')}")
    print("[demo] top responsibility:",
          ", ".join(f"{names.get(pid,pid)} {w:.2f}" for pid, w in top))
    print(f"[demo] saved {args.out}")


def _render(state, weights, names, md, att_team, out_path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ax = draw_pitch(cfg=cfg)
    w = np.asarray(weights, float)
    att = state.is_attacking == 1
    dfd_idx = np.flatnonzero(state.is_attacking == 0)
    att_name = md.home_team_name if att_team == "home" else md.away_team_name
    dfd_name = md.away_team_name if att_team == "home" else md.home_team_name

    # attackers
    ax.scatter(state.px[att], state.py[att], c="#2a78d6", s=210, edgecolors="white",
               linewidths=1.5, zorder=3)
    for i in np.flatnonzero(att):
        ax.text(state.px[i], state.py[i] - 2.6, _surname(names.get(state.player_ids[i], "")),
                color="#123a63", ha="center", va="top", fontsize=7.5, zorder=5,
                path_effects=halo)
    # carrier
    ci = state.carrier_idx
    ax.scatter([state.px[ci]], [state.py[ci]], c="#0d3a6b", s=320, edgecolors="white",
               linewidths=2, marker="*", zorder=4)

    # defenders shaded by responsibility
    wmax = max(w[dfd_idx].max(), 1e-6)
    wn = w / wmax
    colors = matplotlib.colormaps["Reds"](0.25 + 0.75 * wn[dfd_idx])
    sizes = 150 + 650 * wn[dfd_idx]
    ax.scatter(state.px[dfd_idx], state.py[dfd_idx], c=colors, s=sizes,
               edgecolors="#7f0000", linewidths=1.3, zorder=3)
    for i in dfd_idx:
        # push the name/value clear of the marker so big circles don't cover them.
        off = 2.6 + 4.0 * wn[i]
        ax.text(state.px[i], state.py[i] - off, _surname(names.get(state.player_ids[i], "")),
                color="#7f0000", ha="center", va="top", fontsize=7.5, fontweight="bold",
                zorder=5, path_effects=halo)
        if w[i] > 0.06:
            ax.text(state.px[i], state.py[i] + off, f"{w[i]:.2f}", color="#7f0000",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    zorder=5, path_effects=halo)

    ax.set_title(
        f"Defender responsibility: {md.home_team_name} vs {md.away_team_name}, 2022 World Cup Final\n"
        f"{att_name} in possession (blue); {dfd_name} defenders shaded by geometric responsibility",
        color="black", fontsize=11,
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
