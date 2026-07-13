# Scope & data decisions

## Tracking dataset (task D.0)

**Primary tracking source: Metrica Sports sample** (3 open matches, 25 Hz,
cleanest format for building the pipeline quickly). Recorded in
`configs/default.yaml` as `dataset.tracking_source: metrica`.

The paper's private AFC Ajax Eredivisie tracking data is unobtainable. Metrica
is the fastest public substitute to get Phase 1–5 working end-to-end. For results
that need **real player identities** (the market-value validation, Phase 7), the
substitute is **PFF FC 2022 World Cup** (broadcast tracking + events, real
rosters). That loader is now built (`src/defcon/data/pff.py`) and verified on the
public PFF sample (the World Cup Final) — the pipeline is provider-agnostic, so
only the loader changes. The full 64-game bundle is free but gated behind a
request form; see `scripts/download_pff.py`.

**Why two tracking sources.** Metrica is anonymized (`Player1`–`Player28`), which
is fine for building and testing every model but blocks the Transfermarkt join.
PFF carries real names (Messi, Mbappé, …), so it is reserved for the one study
that needs identities. Everything else runs on Metrica.

Two datasets from the paper are public and used regardless of tracking source:

- **Wyscout open events** (Pappalardo et al., 2019) → UxG model (task 3.6).
- **Transfermarkt** market values → validation (Phase 7).

## MVP vs. full scope (task 0.3)

**MVP (six models + pass credit rules):** action-selection (a1), pass-success
(b1), goal-scoring (c1), goal-conceding (c2), UxG (c3), defender-responsibility
(d1), plus the pass/foul credit-assignment rules (Eqs 6–12). This covers the
"better prevent than tackle" headline: EPV, team defensive value, responsibility
weighting, and the Intercept/Disturb/Deter/Concede categories for passes.

**Full (add shots):** shot-blocking model (b2) and the shot credit rules
(Eqs 13–15, unblocked/blocked, goalkeeper save credit). Shots are scarce and
biased in tracking data and require the UxG-based proxy augmentation, so they
are **deferred** — build them only after the pass pipeline validates against
market values. Nothing else depends on b2, so deferring it is safe.

## Build order

Following the milestones in `tasks.md`: **M1 = Phase 0 setup + UxG (c3)** — the
only model that needs no tracking data, shipped first as an early win. Then the
tracking pipeline (Phase 1) on Metrica, graphs + GAT (Phase 2–3), EPV (Phase 4),
credit engine (Phase 5), and the market-value validation (Phase 7).
