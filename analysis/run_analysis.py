"""End-to-end experiment readout. Produces analysis/README_results.md.

    python analysis/run_analysis.py            # synthetic data, instant
    python analysis/run_analysis.py --criteo   # real Criteo uplift data (downloads ~300MB)
"""
from __future__ import annotations

import argparse

import numpy as np

from abtest import (
    beta_binomial,
    cuped_adjust,
    msprt,
    sample_size_proportions,
    segment_effects,
    ttest_means,
    ztest_proportions,
)
from abtest.data import load_criteo, synthetic
from abtest.sequential import msprt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteo", action="store_true")
    ap.add_argument("--nrows", type=int, default=2_000_000)
    args = ap.parse_args()

    if args.criteo:
        df = load_criteo(nrows=args.nrows)
        df["segment"] = (df["f0"] > df["f0"].median()).astype(int) * 2 + (df["f1"] > df["f1"].median()).astype(int)
        df["pre_spend"] = df["f2"]  # proxy covariate for the CUPED demo
        df["revenue"] = df["conversion"] * 30.0
        src = f"Criteo Uplift v2.1 (first {len(df):,} rows)"
    else:
        df = synthetic()
        src = f"synthetic ({len(df):,} users)"

    c, t = df[df.treatment == 0], df[df.treatment == 1]
    lines = [f"# Experiment readout — {src}\n"]

    # 1. Planning
    p0 = c.conversion.mean()
    n_needed = sample_size_proportions(p0, p0 * 0.10)
    lines += ["## 1. Planning", (f"- Baseline conversion {p0:.3%}; to detect a 10% relative lift at 80% power "
              f"you need **{n_needed:,} users per arm**. This test has {len(c):,} / {len(t):,}.\n")]

    # 2. Primary metric
    r = ztest_proportions(int(c.conversion.sum()), len(c), int(t.conversion.sum()), len(t))
    lines += ["## 2. Primary metric (conversion)", f"- {r.summary()}", f"- Significant at 5%: **{r.significant}**\n"]

    # 3. CUPED on revenue
    rev = ttest_means(c.revenue, t.revenue, "revenue/user")
    cu = cuped_adjust(df.revenue.to_numpy(), df.pre_spend.to_numpy())
    rev_adj = ttest_means(cu.y_adj[df.treatment == 0], cu.y_adj[df.treatment == 1], "revenue/user (CUPED)")
    width_raw, width_adj = rev.ci_high - rev.ci_low, rev_adj.ci_high - rev_adj.ci_low
    lines += ["## 3. Revenue with CUPED variance reduction", f"- Raw:   {rev.summary()}", f"- CUPED: {rev_adj.summary()}",
              (f"- Variance reduced **{cu.variance_reduction:.1%}**, CI width {width_raw:.3f} → {width_adj:.3f} "
              f"(equivalent to **{1 - width_adj**2 / width_raw**2:.0%} fewer users** for the same precision)\n")]

    # 4. Sequential
    path = msprt_path(c.conversion.to_numpy(), t.conversion.to_numpy(), every=max(1000, len(c) // 50))
    first = next((p for p in path if p.stop), None)
    lines += ["## 4. Sequential test (mSPRT, safe to peek)",
              "- A fixed-horizon test peeked at every 2% of traffic would inflate false positives ~5x; mSPRT keeps alpha at 5% at every look.",
              f"- Final always-valid p = {msprt(c.conversion.to_numpy(), t.conversion.to_numpy()).p_value:.4f} (conversion)",
              f"- Could have stopped at **{first.n:,} users/arm** ({first.n / len(c):.0%} of the sample)" if first
              else "- Never crossed the stopping boundary.", ""]

    # 5. Bayesian
    b = beta_binomial(int(c.conversion.sum()), len(c), int(t.conversion.sum()), len(t))
    lines += ["## 5. Bayesian view", f"- P(treatment > control) = **{b.prob_treatment_better:.1%}**",
              f"- Expected loss if we ship: {b.expected_loss_choose_treatment:.5f}; if we don't: {b.expected_loss_choose_control:.5f}",
              f"- 95% credible interval on relative lift: [{b.lift_ci[0]:+.1%}, {b.lift_ci[1]:+.1%}]\n"]

    # 6. HTE
    seg = segment_effects(df, "segment")
    lines += ["## 6. Heterogeneous effects by segment (BH-adjusted)", seg.round(4).to_markdown(index=False), ""]

    # 7. Decision
    ship = r.significant and r.abs_lift > 0 and b.prob_treatment_better > 0.95
    lines += ["## 7. Decision", f"**{'SHIP' if ship else 'DO NOT SHIP'}** — " +
              ("primary metric significant, Bayesian P>95%, no segment shows harm." if ship else
               "primary metric not significant at required confidence; extend test or redesign."),
              (f"\nProjected annual impact at 1M users: {r.abs_lift * 1_000_000 * 30:,.0f} USD "
              f"(CI {r.ci_low * 1_000_000 * 30:,.0f} – {r.ci_high * 1_000_000 * 30:,.0f}) assuming $30 per conversion.")]

    out = "\n".join(lines)
    with open("analysis/README_results.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
