"""Credit engine — team defensive value, outcome routing, aggregation (5.1, 5.2, 5.7)
and end-to-end wiring of the Phase 5 rules over a processed match.

D(s_k) = EPV(s_k) − EPV(s_{k+1}) is the team-level defensive value (Eqs 3-4),
measured from the perspective of the team that possessed the ball at k. Each
action is routed to one of seven scenarios (task 5.2) and dispatched to the
matching rule in :mod:`defcon.credit.rules`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from defcon.config import Config, load_config
from defcon.credit import rules
from defcon.credit.rules import CATEGORIES, Credit

__all__ = [
    "team_defensive_value",
    "route_action",
    "aggregate_by_category",
    "CreditEngine",
    "ROUTER_CASES",
]

ROUTER_CASES = (
    "pass_fail_defensive",   # (a) intercepted pass          -> Eq 6
    "pass_fail_no_action",   # (b) out / offside             -> Eq 7
    "pass_success_up",       # (c) completed, EPV rose       -> Eq 8 (penalty)
    "deter",                 # (d) completed, EPV dropped    -> Eqs 9-12
    "foul",                  # (e) foul                      -> foul rule
    "blocked_shot",          # (f) blocked shot              -> Eq 6 variant
    "unblocked_shot",        # (g) unblocked shot            -> Eqs 14-15
)


def team_defensive_value(epv_k: float, epv_next: float, possession_retained: bool) -> float:
    """D(s_k) = EPV(s_k) − EPV(s_{k+1}) from the k-possessor's perspective.

    On a turnover the next state's EPV belongs to the opponent, so in the
    original attacker's frame it enters with a flipped sign: D = EPV_k + EPV_next
    (winning the ball is valuable to the defenders). D > 0 ⇒ good defense.
    """
    if possession_retained:
        return epv_k - epv_next
    return epv_k + epv_next


def route_action(
    action_type: str,
    outcome: str,
    epv_k: float,
    epv_next: float,
    possession_retained: bool,
) -> str:
    """Classify an action into one of the seven credit scenarios (task 5.2)."""
    if action_type == "foul":
        return "foul"
    if action_type == "shot":
        return "unblocked_shot"  # blocked-shot detection needs b2 (deferred)
    if action_type == "pass":
        if outcome in ("out", "offside"):
            return "pass_fail_no_action"
        if outcome == "fail":
            return "pass_fail_defensive"
        if outcome == "success":
            # attacker's EPV change (their own frame)
            delta = (epv_next - epv_k) if possession_retained else (-epv_next - epv_k)
            return "pass_success_up" if delta > 0 else "deter"
    return "pass_fail_no_action"


def aggregate_by_category(credits: list[Credit], player_team: dict | None = None) -> pd.DataFrame:
    """Sum credits per player × category, with a Net column (task 5.7 / 6.1)."""
    rows: dict[str, dict[str, float]] = {}
    for c in credits:
        row = rows.setdefault(c.player, {k: 0.0 for k in CATEGORIES})
        row[c.category] += c.value
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="player")
    if df.empty:
        df = pd.DataFrame(columns=["player", *CATEGORIES])
    for k in CATEGORIES:
        if k not in df:
            df[k] = 0.0
    df["net"] = df[list(CATEGORIES)].sum(axis=1)
    if player_team:
        df["team"] = df["player"].map(player_team)
    return df.sort_values("net", ascending=False).reset_index(drop=True)


class CreditEngine:
    """Wires EPV + responsibility + rules to produce per-action defensive credits.

    ``responsibility_fn(state, option_idx) -> dict[player_id, weight]`` returns the
    defender responsibility distribution; it defaults to a geometric inverse-distance
    prior (robust) but the trained d1 model can be substituted.
    """

    def __init__(self, epv_engine, cfg: Config | None = None, responsibility_fn=None):
        self.epv = epv_engine
        self.cfg = cfg or load_config()
        self.responsibility_fn = responsibility_fn or self._geometric_responsibility

    # -- responsibility -------------------------------------------------------
    def _geometric_responsibility(self, state, option_idx: int, temperature: float = 4.0) -> dict:
        """Softmax of −distance(option, defender) over outfield defenders."""
        dfd = np.flatnonzero(state.is_attacking == 0)
        gk = self._goalkeeper_index(state)
        outfield = [i for i in dfd if i != gk]
        if not outfield:
            outfield = list(dfd)
        ox, oy = state.px[option_idx], state.py[option_idx]
        d = np.hypot(state.px[outfield] - ox, state.py[outfield] - oy)
        z = -d / temperature
        z -= z.max()
        e = np.exp(z)
        w = e / e.sum()
        return {state.player_ids[i]: float(wi) for i, wi in zip(outfield, w)}

    @staticmethod
    def _goalkeeper_index(state) -> int | None:
        gk = np.flatnonzero((state.is_attacking == 0) & (state.is_gk == 1))
        return int(gk[0]) if len(gk) else None

    # -- per-action dispatch --------------------------------------------------
    def credits_for_action(self, state, action, D, epv_k, epv_next, case) -> list[Credit]:
        from defcon.features.nodes import build_node_features

        ng = build_node_features(state, self.cfg)
        # option node = intended receiver (fallback: carrier's forward-most teammate)
        recv = action.get("inferred_receiver")
        option_idx = (
            state.player_ids.index(str(recv))
            if recv is not None and not pd.isna(recv) and str(recv) in state.player_ids
            else state.carrier_idx
        )
        resp = self.responsibility_fn(state, option_idx)

        if case == "pass_fail_defensive":
            interceptor = action.get("interceptor")
            if interceptor is None or pd.isna(interceptor) or str(interceptor) not in resp:
                # no known winner -> treat as positioning-only failure
                return rules.pass_fail_no_action(D, resp)
            p = self._pass_success(ng, option_idx)
            return rules.pass_fail_defensive(D, p, resp, str(interceptor))
        if case == "pass_fail_no_action":
            return rules.pass_fail_no_action(D, resp)
        if case == "pass_success_up":
            return rules.pass_success_epv_up(D, resp)
        if case == "deter":
            # Simplified single-option deter: distribute D by responsibility toward
            # the (deterred) intended receiver. Tagged 'deter'. (Full multi-option
            # Eq 9-12 needs per-option threats; approximated here.)
            return [Credit(v, w * D, "deter") for v, w in resp.items()]
        if case == "foul":
            return rules.foul(D, str(action.get("player")))
        if case == "unblocked_shot":
            gk = self._goalkeeper_index(state)
            gk_id = state.player_ids[gk] if gk is not None else "gk"
            uxg = self._uxg(state)
            on_target = action.get("outcome") == "success"  # goal is on target
            return rules.unblocked_shot(epv_k, uxg, resp, gk_id, on_target, max(epv_next, 0.0))
        return []

    def _pass_success(self, ng, option_idx) -> float:
        import torch

        from defcon.features.graph import to_pyg_data

        data = to_pyg_data(ng, self.cfg)
        with torch.no_grad():
            logit = self.epv.b1(data)[option_idx].item()
        return 1.0 / (1.0 + np.exp(-logit))

    def _uxg(self, state) -> float:
        cx = state.px[state.carrier_idx] + self.cfg.pitch.length / 2.0
        cy = state.py[state.carrier_idx] + self.cfg.pitch.width / 2.0
        return float(self.epv.uxg.score_location(cx, cy))

    # -- match processing -----------------------------------------------------
    def process_match(self, actions: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        """Return a long table of per-action credits (player, value, category, ...)."""
        from defcon.data.metrica import identify_goalkeepers, infer_playing_direction
        from defcon.features.state import graph_state_from_action

        directions = infer_playing_direction(tracking)
        gks = identify_goalkeepers(tracking)
        onball = actions[actions["type"].isin(["pass", "shot", "foul"])].reset_index(drop=True)

        # Precompute EPV + state per action.
        states, epvs = [], []
        for _, a in onball.iterrows():
            s = graph_state_from_action(tracking, a, directions, gks, self.cfg)
            states.append(s)
            epvs.append(self.epv.state_epv(s) if s is not None else np.nan)

        records = []
        for k in range(len(onball) - 1):
            s, a = states[k], onball.iloc[k]
            if s is None or np.isnan(epvs[k]) or np.isnan(epvs[k + 1]):
                continue
            retained = onball.iloc[k]["team"] == onball.iloc[k + 1]["team"]
            D = team_defensive_value(epvs[k], epvs[k + 1], retained)
            case = route_action(a["type"], a["outcome"], epvs[k], epvs[k + 1], retained)
            for c in self.credits_for_action(s, a, D, epvs[k], epvs[k + 1], case):
                records.append({
                    "action_id": int(a["action_id"]), "case": case,
                    "player": c.player, "value": c.value, "category": c.category,
                    "D": D, "epv_k": epvs[k], "epv_next": epvs[k + 1],
                })
        return pd.DataFrame(records)
