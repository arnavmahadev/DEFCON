"""Tests for EPV assembly (tasks 4.1, 4.2)."""

import numpy as np
import pandas as pd
import pytest

from defcon import load_config
from defcon.epv.epv import combine_state_epv
from defcon.epv.epv import _masked_softmax  # noqa: F401  (tested directly)
from defcon.features.state import GraphState

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from defcon.epv.epv import EPVEngine  # noqa: E402
from defcon.models.gat import GATNodeModel  # noqa: E402
from defcon.models.uxg import UxGModel  # noqa: E402


def test_masked_softmax():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    out = _masked_softmax(logits, mask)
    assert out[1] == 0 and out[3] == 0
    assert out.sum() == pytest.approx(1.0)
    assert out[2] > out[0]  # larger logit -> larger prob


def test_combine_state_epv_math():
    tbl = pd.DataFrame({"p_select": [0.5, 0.3, 0.2], "option_value": [0.4, -0.1, 0.9]})
    assert combine_state_epv(tbl) == pytest.approx(0.5 * 0.4 + 0.3 * -0.1 + 0.2 * 0.9)


def _fitted_uxg():
    rng = np.random.default_rng(0)
    n = 800
    x_m = rng.uniform(75, 105, n); y_m = rng.uniform(10, 58, n)
    dist = np.hypot(105 - x_m, y_m - 34)
    p = 1 / (1 + np.exp(-(2.0 - 0.15 * dist)))
    df = pd.DataFrame({"x_m": x_m, "y_m": y_m, "is_header": rng.integers(0, 2, n),
                       "is_set_piece": rng.integers(0, 2, n), "is_goal": rng.binomial(1, p)})
    return UxGModel().fit(df)


def _crafted_state(cfg):
    return GraphState(
        player_ids=["carrier", "m1", "m2", "d1", "d2"],
        px=np.array([0.0, 20.0, -10.0, 5.0, 25.0]),
        py=np.array([0.0, 5.0, -5.0, 2.0, 0.0]),
        pvx=np.zeros(5), pvy=np.zeros(5), pspeed=np.zeros(5), paccel=np.zeros(5),
        is_attacking=np.array([1.0, 1.0, 1.0, 0.0, 0.0]), is_gk=np.zeros(5),
        carrier_idx=0, ball_x=0.0, ball_y=0.0, ball_z=0.0, ball_vx=1.0, ball_vy=0.0,
        pitch_length=cfg.pitch.length, pitch_width=cfg.pitch.width, goal_width=cfg.pitch.goal_width,
    )


def _engine(cfg):
    torch.manual_seed(0)
    def m(out):  # noqa: E306
        return GATNodeModel(in_dim=cfg.features.node_dim, hidden_dim=16, heads=2,
                            mlp_hidden=16, out_dim=out).eval()
    return EPVEngine(a1=m(1), b1=m(1), c1=m(2), c2=m(2), uxg=_fitted_uxg(), cfg=cfg)


def test_option_table_structure_and_select_sums_to_one():
    cfg = load_config()
    engine = _engine(cfg)
    state = _crafted_state(cfg)
    tbl = engine.option_table(state)
    # 2 teammates + 1 shot option.
    assert set(tbl["type"]) == {"pass", "shot"}
    assert (tbl["type"] == "pass").sum() == 2
    assert (tbl["type"] == "shot").sum() == 1
    # Selection is a valid distribution over all options.
    assert tbl["p_select"].sum() == pytest.approx(1.0, abs=1e-5)


def test_option_values_bounded():
    cfg = load_config()
    engine = _engine(cfg)
    tbl = engine.option_table(_crafted_state(cfg))
    # EPV(a,o) is a probability difference in [-1, 1]; shot value = UxG in [0, 1].
    assert tbl["option_value"].between(-1, 1).all()
    assert tbl["epv_success"].between(-1, 1).all()
    shot = tbl[tbl.type == "shot"].iloc[0]
    assert 0.0 <= shot["option_value"] <= 1.0
    assert shot["p_success"] == 1.0  # unblocked (b2 deferred)


def test_state_epv_equals_weighted_sum():
    cfg = load_config()
    engine = _engine(cfg)
    state = _crafted_state(cfg)
    tbl = engine.option_table(state)
    assert engine.state_epv(state) == pytest.approx(combine_state_epv(tbl))
