"""Market-value validation — the paper's headline study (task 7.3).

The paper's key finding ("better prevent than tackle") is that a defender's
per-90 **Concede** and **Net** credit correlate *positively* with log market value,
while **Intercept** (visible on-ball actions alone) correlates *negatively* — i.e.
the market rewards prevention, not tackling.

This module implements the correlation machinery (Pearson of each credit category
vs. log market value, overall and by position). It is model-agnostic: feed it a
per-player credit table and a market-value mapping.

    ⚠️  DATA BLOCKER: the Metrica sample data used here has **anonymized players**
    (Player1–28) with no real identities, so it cannot be joined to Transfermarkt
    values — the headline finding is *not* reproducible on this substitute. To run
    it for real, switch the tracking source to one with real identities (PFF FC
    2022 World Cup) and join scraped Transfermarkt values. The code below is ready
    for that; only the input data is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "correlate_value", "CREDIT_METRICS", "add_log_value", "attach_identities",
    "attach_transfermarkt", "transfermarkt_values",
]


def _norm_name(s: str) -> str:
    """Accent/punctuation-insensitive name key for cross-source joins."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def transfermarkt_values(
    players_csv: str, valuations_csv: str, as_of: str = "2022-11-20"
) -> pd.DataFrame:
    """Transfermarkt market value per player as of ``as_of`` (tournament time).

    Takes each player's most recent valuation on/before ``as_of`` (falling back to
    their earliest later one), joined to name/country/position. Returns columns
    ``player_id, name, norm, country, position_tm, market_value_in_eur``.
    """
    players = pd.read_csv(
        players_csv,
        usecols=["player_id", "name", "country_of_citizenship", "sub_position"],
    )
    vals = pd.read_csv(valuations_csv, usecols=["player_id", "date", "market_value_in_eur"])
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    cutoff = pd.Timestamp(as_of)

    on_before = (
        vals[vals["date"] <= cutoff].sort_values("date").groupby("player_id").tail(1)
    )
    missing = set(vals["player_id"]) - set(on_before["player_id"])
    after = (
        vals[vals["player_id"].isin(missing)].sort_values("date").groupby("player_id").head(1)
    )
    val = pd.concat([on_before, after])[["player_id", "market_value_in_eur"]]

    df = players.merge(val, on="player_id", how="inner")
    df["norm"] = df["name"].map(_norm_name)
    return df.rename(columns={"country_of_citizenship": "country", "sub_position": "position_tm"})


def attach_transfermarkt(
    table: pd.DataFrame,
    players_csv: str,
    valuations_csv: str,
    as_of: str = "2022-11-20",
    name_col: str = "name",
    team_col: str = "team",
) -> pd.DataFrame:
    """Attach a ``market_value`` column to a credit table by name-matching.

    Joins on a normalized (accent-stripped) name. When a name is ambiguous
    (shared across players), prefers the Transfermarkt row whose country matches
    the credit table's national team, then the highest value. Unmatched players
    keep NaN so :func:`correlate_value` simply excludes them.
    """
    tm = transfermarkt_values(players_csv, valuations_csv, as_of)
    out = table.copy()
    out["norm"] = out[name_col].map(_norm_name)

    merged = out.merge(
        tm[["norm", "market_value_in_eur", "country", "position_tm"]], on="norm", how="left"
    )
    if team_col in merged:
        merged["_country_match"] = (
            merged["country"].fillna("").str.lower() == merged[team_col].fillna("").str.lower()
        ).astype(int)
    else:
        merged["_country_match"] = 0
    merged = (
        merged.sort_values(["_country_match", "market_value_in_eur"], ascending=False)
        .drop_duplicates(subset=[c for c in ("player", name_col) if c in merged][0])
        .drop(columns=["_country_match", "norm"])
        .rename(columns={"market_value_in_eur": "market_value"})
    )
    return merged.sort_values("net_p90", ascending=False).reset_index(drop=True) \
        if "net_p90" in merged else merged

CREDIT_METRICS = ("intercept_p90", "disturb_p90", "deter_p90", "concede_p90", "net_p90")


def add_log_value(table: pd.DataFrame, value_col: str = "market_value") -> pd.DataFrame:
    """Add a ``log_value`` column (natural log of market value)."""
    out = table.copy()
    out["log_value"] = np.log(out[value_col].clip(lower=1.0))
    return out


def attach_identities(
    credit_table: pd.DataFrame,
    identities: pd.DataFrame,
    values: pd.DataFrame | dict | None = None,
    credit_key: str = "player",
    value_col: str = "market_value",
) -> pd.DataFrame:
    """Join a per-player credit table to real identities and market values.

    This is the bridge that makes the market-value study runnable: the credit
    aggregation is keyed by anonymous ``player_id``; ``identities`` (from
    :func:`defcon.data.pff.pff_identity_table`) attaches the real ``name``,
    ``position`` and ``team``; ``values`` attaches a Transfermarkt market value
    (a ``{name: value}`` dict or a DataFrame with ``name`` + ``value_col``).

    Rows whose player has no identity are dropped; rows with no market value keep
    NaN (so :func:`correlate_value` simply excludes them). Returns a table ready
    for :func:`correlate_value`.
    """
    ident = identities.rename(columns={"player_id": credit_key})
    keep = [c for c in [credit_key, "name", "position", "team_name"] if c in ident]
    out = credit_table.merge(ident[keep], on=credit_key, how="inner")

    if values is not None:
        if isinstance(values, dict):
            values = pd.DataFrame({"name": list(values), value_col: list(values.values())})
        out = out.merge(values[["name", value_col]], on="name", how="left")
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p)


def correlate_value(
    table: pd.DataFrame,
    value_col: str = "market_value",
    metrics: tuple[str, ...] = CREDIT_METRICS,
    position_col: str | None = "position",
) -> pd.DataFrame:
    """Pearson r of each credit metric vs. log market value (Table-4 style).

    Returns one row per group ('overall' plus each position, if ``position_col``
    is present) with the correlation of every metric and the group size.
    """
    tbl = add_log_value(table, value_col)

    def row(name: str, sub: pd.DataFrame) -> dict:
        lv = sub["log_value"].to_numpy()
        rec = {"group": name, "n": int(len(sub))}
        for m in metrics:
            if m in sub:
                r, p = _pearson(sub[m].to_numpy(), lv)
                rec[f"r_{m}"] = r
                rec[f"p_{m}"] = p
        return rec

    rows = [row("overall", tbl)]
    if position_col and position_col in tbl:
        for pos, sub in tbl.groupby(position_col):
            rows.append(row(str(pos), sub))
    return pd.DataFrame(rows)


def plot_value_scatter(table, metric="net_p90", value_col="market_value", out_path=None,
                       cfg=None, annotate=False, name_col="name", title=None):
    """Scatter of a credit metric vs. log market value with a fit line (Figs 8-9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = add_log_value(table, value_col)
    tbl = tbl[np.isfinite(tbl["log_value"]) & np.isfinite(tbl[metric])]
    x = tbl["log_value"].to_numpy()
    y = tbl[metric].to_numpy()
    r, _ = _pearson(y, x)

    color_by = {"DF": "#2a78d6", "MF": "#1baf7a", "FW": "#eda100", "GK": "#8a8f98"}
    colors = tbl["position"].map(color_by).fillna("#5c6f92") if "position" in tbl else "#5c6f92"

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.scatter(x, y, s=60, c=colors, edgecolors="white", linewidths=1.2, zorder=3)
    if np.isfinite(r):
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b0 + b1 * xs, color="#46587b", lw=2, zorder=2)
    if annotate and name_col in tbl:
        for xi, yi, nm in zip(x, y, tbl[name_col], strict=False):
            ax.annotate(str(nm).split()[-1], (xi, yi), fontsize=6.5, color="#333",
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log market value (Transfermarkt, Nov 2022)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs. market value   (r = {r:+.2f}, n = {len(tbl)})")
    ax.grid(color="#e5e0d3", lw=1)
    if "position" in tbl:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=p, markersize=8)
                   for p, c in color_by.items() if (tbl["position"] == p).any()]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return r
