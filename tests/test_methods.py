import numpy as np

from abtest import beta_binomial, cuped_adjust, msprt, segment_effects, ztest_proportions
from abtest.data import synthetic
from abtest.sequential import msprt_path


def test_ztest_detects_real_lift():
    r = ztest_proportions(5000, 100_000, 5600, 100_000)
    assert r.significant and r.ci_low > 0


def test_ztest_null_is_not_significant():
    r = ztest_proportions(5000, 100_000, 5020, 100_000)
    assert not r.significant


def test_cuped_reduces_variance_and_keeps_lift():
    df = synthetic(100_000)
    y, x = df["revenue"].to_numpy(), df["pre_spend"].to_numpy()
    res = cuped_adjust(y, x)
    assert res.variance_reduction > 0.05
    # adjustment is mean-zero → lift unchanged in expectation
    t = df["treatment"].to_numpy()
    raw = y[t == 1].mean() - y[t == 0].mean()
    adj = res.y_adj[t == 1].mean() - res.y_adj[t == 0].mean()
    assert abs(raw - adj) < 0.05 * abs(raw) + 0.5


def test_msprt_controls_false_positives_under_null():
    rng = np.random.default_rng(0)
    false_pos = 0
    for _ in range(100):
        c, t = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
        if any(r.stop for r in msprt_path(c, t, every=250)):
            false_pos += 1
    assert false_pos <= 10  # ≤ alpha * trials with margin; a naive peeking t-test would be ~30-40


def test_msprt_detects_true_effect():
    rng = np.random.default_rng(1)
    c, t = rng.normal(0, 1, 5000), rng.normal(0.15, 1, 5000)
    assert msprt(c, t).stop


def test_bayesian_agrees_with_frequentist_direction():
    b = beta_binomial(5000, 100_000, 5600, 100_000)
    assert b.prob_treatment_better > 0.99
    assert b.expected_loss_choose_treatment < b.expected_loss_choose_control


def test_segment_effects_orders_and_adjusts():
    df = synthetic(400_000)
    out = segment_effects(df, "segment")
    assert len(out) == 4
    assert "p_adj_bh" in out and (out["p_adj_bh"] >= out["p_value"]).all()
    assert out.iloc[0]["abs_lift"] >= out.iloc[-1]["abs_lift"]
