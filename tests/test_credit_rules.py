"""Tests for the credit-assignment rules (Phase 5), incl. the paper's worked
examples in Figures 3 and 5."""

import pytest

from defcon.credit.rules import (
    CATEGORIES,
    Credit,
    blocked_shot,
    deter,
    foul,
    pass_fail_defensive,
    pass_fail_no_action,
    pass_success_epv_up,
    unblocked_shot,
)


def _by_player(credits):
    out = {}
    for c in credits:
        out[c.player] = out.get(c.player, 0.0) + c.value
    return out


# --------------------------------------------------------------------------- #
# Eq 6 — Figure 3 worked example
# --------------------------------------------------------------------------- #
def test_fig3_intercepted_pass():
    """Fig 3: p=0.548, D=0.190, interceptor responsibility 0.210.

    Interceptor total = p·D + w·(1−p)·D. (The task file prints '0.012', which
    corresponds to D≈0.019 — a decimal slip; the *formula* is what we reproduce.)
    """
    D, p = 0.190, 0.548
    resp = {"blue9": 0.210, "blue4": 0.500, "blue2": 0.290}
    credits = pass_fail_defensive(D, p, resp, interceptor="blue9")

    expected_interceptor = p * D + 0.210 * (1 - p) * D
    assert _by_player(credits)["blue9"] == pytest.approx(expected_interceptor)
    # A non-interceptor gets only its positioning (disturb) share.
    assert _by_player(credits)["blue4"] == pytest.approx(0.500 * (1 - p) * D)

    # Category split: exactly one intercept credit (the on-ball win) = p·D.
    intercepts = [c for c in credits if c.category == "intercept"]
    assert len(intercepts) == 1
    assert intercepts[0].player == "blue9"
    assert intercepts[0].value == pytest.approx(p * D)
    # Everyone (incl. interceptor) has a disturb share.
    assert sum(1 for c in credits if c.category == "disturb") == 3


def test_eq6_total_credit_conserved():
    """Total credit = D·[p + (1−p)·Σw]; with Σw=1 this is exactly D."""
    D, p = 0.20, 0.6
    resp = {"a": 0.5, "b": 0.3, "c": 0.2}  # sums to 1
    total = sum(c.value for c in pass_fail_defensive(D, p, resp, "a"))
    assert total == pytest.approx(D)


# --------------------------------------------------------------------------- #
# Eqs 7 & 8
# --------------------------------------------------------------------------- #
def test_eq7_pass_fail_no_action_is_disturb():
    resp = {"a": 0.6, "b": 0.4}
    credits = pass_fail_no_action(0.1, resp)
    assert all(c.category == "disturb" for c in credits)
    assert _by_player(credits)["a"] == pytest.approx(0.06)
    assert sum(c.value for c in credits) == pytest.approx(0.1)


def test_eq8_pass_success_epv_up_is_concede_penalty():
    resp = {"a": 0.7, "b": 0.3}
    D = -0.08  # EPV rose for the attacker -> penalty
    credits = pass_success_epv_up(D, resp)
    assert all(c.category == "concede" for c in credits)
    assert all(c.value < 0 for c in credits)
    assert sum(c.value for c in credits) == pytest.approx(D)


# --------------------------------------------------------------------------- #
# Eqs 9-12 — deter
# --------------------------------------------------------------------------- #
def test_deter_distributes_D_over_threatening_options():
    D = 0.12
    options = [
        {"threat": 0.3, "responsibilities": {"a": 0.5, "b": 0.5}},
        {"threat": 0.1, "responsibilities": {"a": 1.0}},
        {"threat": -0.2, "responsibilities": {"c": 1.0}},  # not threatening -> ignored
    ]
    credits = deter(D, options)
    assert all(c.category == "deter" for c in credits)
    # Total distributed equals D (only threatening options contribute).
    assert sum(c.value for c in credits) == pytest.approx(D)
    # 'c' (non-threatening option) receives nothing.
    assert "c" not in _by_player(credits)
    # Option 1 gets 0.3/0.4 of D; option 2 gets 0.1/0.4 of D.
    assert _by_player(credits)["b"] == pytest.approx(D * 0.75 * 0.5)


def test_deter_no_threats_returns_empty():
    assert deter(0.1, [{"threat": -0.1, "responsibilities": {"a": 1.0}}]) == []


# --------------------------------------------------------------------------- #
# Foul + shots (Eqs 13-15) — Figure 5 worked example
# --------------------------------------------------------------------------- #
def test_foul_penalizes_only_the_fouler():
    credits = foul(-0.25, "red3")
    assert credits == [Credit("red3", -0.25, "concede")]


def test_blocked_shot_reuses_eq6():
    D, p_nb = 0.15, 0.4
    resp = {"a": 0.6, "b": 0.4}
    credits = blocked_shot(D, p_nb, resp, blocker="a")
    # Blocker gets on-ball + positioning; total conserved with Σw=1.
    assert sum(c.value for c in credits) == pytest.approx(D)
    assert any(c.category == "intercept" and c.player == "a" for c in credits)


def test_fig5_unblocked_shot():
    """Fig 5: EPV=0.024, UxG=0.121 -> shared penalty −0.097; GK save +0.069."""
    epv, uxg = 0.024, 0.121
    epv_next = 0.052  # so that UxG − EPV_{k+1} = 0.069 (the stated save credit)
    outfield = {"d1": 0.6, "d2": 0.4}  # sums to 1
    credits = unblocked_shot(epv, uxg, outfield, goalkeeper="gk", on_target=True, epv_next=epv_next)

    penalty = sum(c.value for c in credits if c.category == "concede")
    assert penalty == pytest.approx(-0.097, abs=1e-6)
    gk = [c for c in credits if c.player == "gk"][0]
    assert gk.value == pytest.approx(0.069, abs=1e-6)
    assert gk.category == "intercept"


def test_off_target_shot_gives_gk_no_save_credit():
    credits = unblocked_shot(0.02, 0.12, {"d1": 1.0}, "gk", on_target=False, epv_next=0.0)
    gk = [c for c in credits if c.player == "gk"][0]
    assert gk.value == 0.0


def test_all_credits_have_valid_category():
    credits = pass_fail_defensive(0.1, 0.5, {"a": 0.5, "b": 0.5}, "a")
    assert all(c.category in CATEGORIES for c in credits)
