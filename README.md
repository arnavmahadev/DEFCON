# DEFCON — Reproducing "Better Prevent than Tackle"

An independent, from-scratch **reproduction and study implementation** of the paper:

> **Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks**
> Hyunsung Kim, Sangwoo Seo, Hoyoung Choi, Tom Boomstra, Jinsung Yoon, and Chanyoung Park
> (KAIST · Fitogether Inc. · AFC Ajax)

> [!IMPORTANT]
> **This is an unofficial, personal project.** It is not the authors' code, is not affiliated
> with or endorsed by the authors, KAIST, Fitogether, or AFC Ajax, and reproduces the method
> for educational and portfolio purposes. All credit for the ideas, framework, and results
> belongs to the original authors — see [Credits](#credits). Any bugs or deviations from the
> paper are mine.

## What this project is

The paper introduces **DEFCON** (DEFensive CONtribution evaluator), a framework that quantifies
how much *individual defenders* contribute in soccer — not just through visible actions like
tackles and interceptions, but through positioning that **prevents** dangerous attacks before
they happen ("if I have to make a tackle, I've already made a mistake" — Paolo Maldini).

The central idea is that **defense is the zero-sum mirror of offense**: every attacking action
changes the attacking team's **Expected Possession Value (EPV)**, and the negation of that change
is the defending team's value, which DEFCON distributes onto individual defenders using
scenario-specific credit-assignment rules. Component quantities (action-selection probability,
success probability, outcome-conditioned EPV, and defender responsibility) are estimated with
**Graph Attention Networks** over a graph of all players and both goals.

**This repository is my end-to-end re-implementation of that pipeline**, built as a learning
project. It covers:

- a data pipeline (tracking + event parsing, event–tracking synchronization, feature engineering);
- seven component models (six GNNs + one logistic-regression xG model);
- EPV computation and the defensive credit-assignment engine;
- aggregation into interpretable categories (**Intercept / Disturb / Deter / Concede**);
- validation against player market values; and
- temporal, spatial, and pairwise visualizations.

> [!NOTE]
> **Status: work in progress.** This is a personal reproduction being built incrementally;
> results here are my own and will not exactly match the paper (different data — see below).

## Data

The original paper uses **proprietary optical tracking + event data** for 564 Dutch Eredivisie
matches (2023–24 and 2024–25 seasons), provided by AFC Ajax. **That dataset is private and is
not, and will never be, included in this repository.**

Because the original tracking data is unobtainable, this reproduction substitutes **public data**:

| Purpose | Source |
|---|---|
| Player/ball tracking + events | Public tracking dataset (e.g. Metrica Sports sample, PFF FC 2022 World Cup) — *not committed* |
| Unblocked-shot xG (UxG) model | [Wyscout open event data](https://figshare.com/collections/Soccer_match_event_dataset/4415000) (Pappalardo et al., 2019) |
| Validation labels | Player market values from [Transfermarkt](https://www.transfermarkt.com) |

All data is git-ignored. See `.gitignore`. No proprietary data is redistributed here.

## Method overview

For each on-ball attacking action `k`:

1. **Estimate EPV** via a learned decomposition — action selection × success probability ×
   outcome-conditioned value (following the EPV framework of Fernández et al.).
2. **Team-level defensive value** = EPV before − EPV after the action.
3. **Defender responsibility** = probability each defender would recover the ball if the action
   failed (a failure-conditioned receiver distribution).
4. **Credit assignment** — apply the rule matching the situation (pass intercepted, pass
   conceded, threat deterred, foul, shot blocked, shot unblocked, …) to split the team value
   onto individual players.
5. **Aggregate** per player into four categories (Intercept / Disturb / Deter / Concede),
   normalized per 90 minutes.

### Component models

| Model | Task | Type |
|---|---|---|
| Action selection | Which option the ball-carrier attempts | GAT (softmax over teammates + goal) |
| Pass success | P(pass/dribble succeeds) | GAT (node-wise) |
| Shot blocking | P(shot blocked by an outfield player) | GAT (graph-level) |
| Goal-scoring | Outcome-conditioned P(score soon) | GAT (node-wise) |
| Goal-conceding | Outcome-conditioned P(concede soon) | GAT (node-wise) |
| UxG | Expected goal for an unblocked shot | Logistic regression |
| Responsibility | Which defender recovers the ball if the action fails | GAT (softmax over opponents) |

## Tech stack

Python · PyTorch · PyTorch Geometric (GATs) · scikit-learn / XGBoost / CatBoost (baselines &
UxG) · pandas / NumPy · Plotly & Matplotlib (visualizations).

## Repository layout (planned)

```
src/defcon/
  data/      # tracking + event parsing, synchronization, labeling
  features/  # graph construction, node/edge features
  models/    # GAT backbone + component-model heads, UxG
  epv/        # EPV assembly
  credit/    # defensive credit-assignment rules & aggregation
  eval/      # metrics, baselines, market-value study
  viz/       # timeline, heatmaps, pairwise matrices
```

## Credits

**Original paper and method** — all core credit belongs to the authors:

> Hyunsung Kim, Sangwoo Seo, Hoyoung Choi, Tom Boomstra, Jinsung Yoon, Chanyoung Park.
> *Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks.*

This reproduction also builds directly on ideas and resources from prior work cited in the paper,
including (non-exhaustive):

- **Expected Possession Value** — Fernández, Bornn & Cervone.
- **Action valuation (VAEP)** — Decroos, Bransen, Van Haaren & Davis.
- **Graph Attention Networks** — Veličković et al.
- **Expected Threat (xT)** — Karun Singh.
- **Wyscout open dataset** — Pappalardo et al., *Scientific Data* (2019).
- **Market values** — Transfermarkt.

Please cite the original paper (not this repository) if you reference the DEFCON method.

## License & disclaimer

This is a personal educational/portfolio project. It reimplements a published method but contains
no code or data from the original authors. It is not affiliated with the paper's authors or
their institutions. No proprietary tracking data is included or redistributed.
