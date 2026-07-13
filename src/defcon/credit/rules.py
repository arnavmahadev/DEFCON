"""Defensive credit-assignment rules (Phase 5, Eqs 6-15).

Pure functions that split a team-level defensive value ``D(s_k)`` onto individual
defenders according to the scenario. Each returns a list of :class:`Credit`, every
one carrying exactly one of the four categories (task 5.7):

    intercept — an on-ball defensive win (the ball-winner's p·D share, GK saves)
    disturb   — positioning share on a failure (the w·(1−p)·D / w·D shares)
    deter     — credit for threatening options the attacker was forced to avoid
    concede   — a penalty (defenders let EPV rise, conceded a shot, or fouled)

Sign convention: ``D > 0`` means the defense reduced the attacker's EPV (good);
``D < 0`` is a penalty. These functions are model-agnostic and are unit-tested
directly against the paper's worked examples (Figs 3 and 5).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Credit",
    "CATEGORIES",
    "pass_fail_defensive",
    "pass_fail_no_action",
    "pass_success_epv_up",
    "deter",
    "foul",
    "blocked_shot",
    "unblocked_shot",
]

CATEGORIES = ("intercept", "disturb", "deter", "concede")


@dataclass(frozen=True)
class Credit:
    player: str
    value: float
    category: str

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown category {self.category!r}; expected one of {CATEGORIES}")


def pass_fail_defensive(
    D: float, p: float, responsibilities: dict[str, float], interceptor: str
) -> list[Credit]:
    """Eq 6 — pass fails via a defensive action.

    The interceptor gets an on-ball share ``p·D`` (intercept) plus its positioning
    share ``w·(1−p)·D`` (disturb); every other defender gets only its positioning
    share ``w·(1−p)·D`` (disturb).
    """
    credits = [
        Credit(v, w * (1.0 - p) * D, "disturb") for v, w in responsibilities.items()
    ]
    credits.append(Credit(interceptor, p * D, "intercept"))
    return credits


def pass_fail_no_action(D: float, responsibilities: dict[str, float]) -> list[Credit]:
    """Eq 7 — pass fails with no defensive action (out / offside).

    The full defensive value is shared by responsibility (all positioning).
    """
    return [Credit(v, w * D, "disturb") for v, w in responsibilities.items()]


def pass_success_epv_up(D: float, responsibilities: dict[str, float]) -> list[Credit]:
    """Eq 8 — pass succeeds and the attacker's EPV rose (``D < 0``, a penalty).

    The penalty is shared by responsibility (defenders failed to prevent it).
    """
    return [Credit(v, w * D, "concede") for v, w in responsibilities.items()]


def deter(D: float, options: list[dict]) -> list[Credit]:
    """Eqs 9-12 — pass succeeds but EPV dropped: the defense *deterred* threats.

    ``options`` is a list of threatening options, each a dict with:
      ``threat``          = E[G | a, O=1] − E[G | a]   (how much a would raise EPV)
      ``responsibilities`` = {defender: w_v(s_k, a)}   (who covers option a)

    ``D`` is allocated across options ∝ their threat, then split within each option
    by responsibility, so the total distributed equals ``D`` (task 5.5).
    """
    threats = [max(float(o["threat"]), 0.0) for o in options]
    total = sum(threats)
    credits: list[Credit] = []
    if total <= 0:
        return credits
    for o, th in zip(options, threats):
        if th <= 0:
            continue
        D_a = D * th / total
        for v, w in o["responsibilities"].items():
            credits.append(Credit(v, w * D_a, "deter"))
    return credits


def foul(D: float, fouler: str) -> list[Credit]:
    """Foul (Sec 2.4e) — the entire penalty falls on the fouling player."""
    return [Credit(fouler, D, "concede")]


def blocked_shot(
    D: float, p_not_blocked: float, responsibilities: dict[str, float], blocker: str
) -> list[Credit]:
    """Blocked shot (Sec 2.5) — reuse Eq 6 with ``p = P(not blocked)``."""
    return pass_fail_defensive(D, p_not_blocked, responsibilities, blocker)


def unblocked_shot(
    epv: float,
    uxg: float,
    outfield_responsibilities: dict[str, float],
    goalkeeper: str,
    on_target: bool,
    epv_next: float,
) -> list[Credit]:
    """Unblocked shot (Eqs 14-15).

    Outfield defenders share the penalty ``w_v·(EPV − UxG)`` (concede; the GK has
    zero responsibility weight here). The goalkeeper is credited ``UxG − EPV_{k+1}``
    if the shot was on target (a save), else 0 (intercept — an on-ball save).
    """
    delta = epv - uxg  # negative when the shot was dangerous -> a penalty
    credits = [Credit(v, w * delta, "concede") for v, w in outfield_responsibilities.items()]
    save = (uxg - epv_next) if on_target else 0.0
    credits.append(Credit(goalkeeper, save, "intercept"))
    return credits
