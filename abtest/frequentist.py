"""Classic two-sample tests with effect sizes and confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control_mean: float
    treatment_mean: float
    abs_lift: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (f"{self.metric}: {self.control_mean:.4f} → {self.treatment_mean:.4f} "
                f"({self.rel_lift:+.2%}), 95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.4f}")


def ztest_proportions(x_c: int, n_c: int, x_t: int, n_t: int, metric: str = "conversion") -> TestResult:
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pooled
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    return TestResult(metric, p_c, p_t, diff, diff / p_c if p_c else np.nan,
                      diff - 1.96 * se, diff + 1.96 * se, p, n_c, n_t)


def ttest_means(control: np.ndarray, treatment: np.ndarray, metric: str = "revenue") -> TestResult:
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    _t, p = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    return TestResult(metric, control.mean(), treatment.mean(), diff,
                      diff / control.mean() if control.mean() else np.nan,
                      diff - 1.96 * se, diff + 1.96 * se, p, len(control), len(treatment))
