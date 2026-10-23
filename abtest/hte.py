"""Heterogeneous treatment effects: the lift per segment, with a multiple-comparison guard."""
from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from .frequentist import ztest_proportions


def segment_effects(df: pd.DataFrame, segment: str, treatment_col: str = "treatment",
                    outcome_col: str = "conversion", min_n: int = 1000) -> pd.DataFrame:
    """One row per segment: rates, lift, CI, raw p and Benjamini-Hochberg adjusted p."""
    rows = []
    for seg, g in df.groupby(segment):
        c, t = g[g[treatment_col] == 0], g[g[treatment_col] == 1]
        if len(c) < min_n or len(t) < min_n:
            continue
        r = ztest_proportions(int(c[outcome_col].sum()), len(c), int(t[outcome_col].sum()), len(t))
        rows.append({segment: seg, "n_control": r.n_control, "n_treatment": r.n_treatment,
                     "control_rate": r.control_mean, "treatment_rate": r.treatment_mean,
                     "abs_lift": r.abs_lift, "rel_lift": r.rel_lift,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out = out.sort_values("abs_lift", ascending=False).reset_index(drop=True)
    return out
