"""Tests for team defensive value, the outcome router, and aggregation (5.1, 5.2, 5.7)."""

import pytest

from defcon.credit.engine import (
    ROUTER_CASES,
    aggregate_by_category,
    route_action,
    team_defensive_value,
)
from defcon.credit.rules import Credit


def test_team_value_retained_possession():
    # EPV drops from 0.5 to 0.2 while keeping the ball -> good defense, D>0.
    assert team_defensive_value(0.5, 0.2, possession_retained=True) == pytest.approx(0.3)


def test_team_value_turnover_adds():
    # Turnover: next EPV belongs to the opponent -> D = epv_k + epv_next.
    assert team_defensive_value(0.3, 0.15, possession_retained=False) == pytest.approx(0.45)


def test_router_intercepted_pass():
    assert route_action("pass", "fail", 0.3, 0.1, possession_retained=False) == "pass_fail_defensive"


def test_router_out_of_play():
    assert route_action("pass", "out", 0.3, 0.0, possession_retained=False) == "pass_fail_no_action"


def test_router_completed_pass_epv_up_is_penalty():
    # Retained, EPV rose 0.2 -> 0.4 -> the attacker progressed -> penalty case.
    assert route_action("pass", "success", 0.2, 0.4, possession_retained=True) == "pass_success_up"


def test_router_completed_pass_epv_down_is_deter():
    # Retained, EPV dropped 0.4 -> 0.2 -> defense deterred -> deter case.
    assert route_action("pass", "success", 0.4, 0.2, possession_retained=True) == "deter"


def test_router_foul_and_shot():
    assert route_action("foul", "success", 0.1, 0.1, True) == "foul"
    assert route_action("shot", "fail", 0.1, 0.0, False) == "unblocked_shot"


def test_all_router_outputs_are_known_cases():
    cases = {
        route_action("pass", "fail", 0.3, 0.1, False),
        route_action("pass", "success", 0.2, 0.4, True),
        route_action("pass", "success", 0.4, 0.2, True),
        route_action("foul", "success", 0.1, 0.1, True),
        route_action("shot", "fail", 0.1, 0.0, False),
    }
    assert cases <= set(ROUTER_CASES)


def test_aggregate_by_category():
    credits = [
        Credit("a", 0.1, "intercept"),
        Credit("a", 0.02, "disturb"),
        Credit("a", -0.05, "concede"),
        Credit("b", 0.03, "deter"),
    ]
    df = aggregate_by_category(credits)
    a = df[df.player == "a"].iloc[0]
    assert a["intercept"] == pytest.approx(0.1)
    assert a["disturb"] == pytest.approx(0.02)
    assert a["concede"] == pytest.approx(-0.05)
    assert a["net"] == pytest.approx(0.07)
    # 'a' has higher net than 'b' -> sorted first.
    assert df.iloc[0]["player"] == "a"
