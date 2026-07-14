#!/usr/bin/env python
"""Assemble EPV from the trained component models (Phase 4).

- Prints a Figure-2-style per-option table for one action (task 4.1).
- Validates that state EPV rises toward the opponent goal (task 4.2).
- Renders an EPV pitch surface by sweeping the ball-carrier over the pitch.

Usage: python scripts/compute_epv.py
"""

from __future__ import annotations

import copy
import warnings

import numpy as np

from defcon import load_config
from defcon.data.metrica import identify_goalkeepers, infer_playing_direction
from defcon.data.pipeline import process_metrica_match
from defcon.epv.epv import EPVEngine
from defcon.features.graph import X_COL
from defcon.features.state import graph_state_from_action
from defcon.viz.pitch import plot_epv_surface

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = load_config()
    gd = cfg.path("tracking_dir") / "metrica"
    actions, tracking = process_metrica_match(gd, 1, cfg)
    directions = infer_playing_direction(tracking)
    gks = identify_goalkeepers(tracking)
    engine = EPVEngine.from_checkpoints(cfg)

    # ---- 4.1 Figure-2 table for one attacking action ----
    passes = actions[actions.type == "pass"].reset_index(drop=True)
    row = passes.iloc[300]
    state = graph_state_from_action(tracking, row, directions, gks, cfg)
    tbl = engine.option_table(state)
    tbl = tbl.sort_values("p_select", ascending=False)
    print("=== 4.1 per-option EPV (Figure-2 style) ===")
    print(f"action by {row.player} ({row.team}), outcome={row.outcome}")
    with_fmt = tbl.head(8).copy()
    for c in ["p_select", "p_success", "epv_success", "epv_fail", "option_value"]:
        with_fmt[c] = with_fmt[c].map(lambda v: f"{v:+.3f}")
    print(with_fmt.to_string(index=False))
    print(f"state EPV = {engine.state_epv(state):.4f}")

    # ---- 4.2 monotonicity: EPV vs distance to goal, over many real actions ----
    print("\n=== 4.2 EPV rises toward goal (real actions) ===")
    epvs, xs = [], []
    for _, a in passes.iloc[::3].iterrows():
        s = graph_state_from_action(tracking, a, directions, gks, cfg)
        if s is None:
            continue
        epvs.append(engine.state_epv(s))
        xs.append(s.px[s.carrier_idx])  # +x = toward attacking goal
    epvs, xs = np.array(epvs), np.array(xs)
    corr = np.corrcoef(xs, epvs)[0, 1]
    print(f"n={len(epvs)} | EPV range [{epvs.min():.3f}, {epvs.max():.3f}] | mean {epvs.mean():.3f}")
    print(f"corr(carrier_x_toward_goal, EPV) = {corr:+.3f}  (positive => higher near goal)")
    own, opp = epvs[xs < 0].mean(), epvs[xs > 0].mean()
    print(f"mean EPV  own half {own:.3f}  vs  attacking half {opp:.3f}")

    # ---- EPV surface: sweep the carrier over the pitch on a frozen frame ----
    print("\n[epv] rendering EPV surface ...")
    base = graph_state_from_action(tracking, passes.iloc[300], directions, gks, cfg)
    L, W = cfg.pitch.length, cfg.pitch.width
    gx = np.linspace(-L / 2 + 3, L / 2 - 3, 34)
    gy = np.linspace(-W / 2 + 3, W / 2 - 3, 22)
    grid = np.zeros((len(gy), len(gx)))
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            s = copy.copy(base)
            s.px = base.px.copy(); s.py = base.py.copy()
            s.px[base.carrier_idx] = xx
            s.py[base.carrier_idx] = yy
            s.ball_x, s.ball_y = xx, yy
            grid[j, i] = engine.state_epv(s)
    out = plot_epv_surface(gx, gy, grid, "docs/img/epv_surface.png", cfg, state=base,
                           title="Expected Possession Value: carrier swept over pitch")
    print(f"[epv] saved {out}  (surface range [{grid.min():.3f}, {grid.max():.3f}])")


if __name__ == "__main__":
    main()
