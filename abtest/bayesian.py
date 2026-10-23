"""Beta-Binomial Bayesian A/B test for conversion rates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesResult:
    prob_treatment_better: float
    expected_loss_choose_treatment: float  # expected regret if T is wrong
    expected_loss_choose_control: float
    lift_ci: tuple[float, float]            # 95% credible interval on relative lift


def beta_binomial(x_c: int, n_c: int, x_t: int, n_t: int, prior=(1, 1), draws: int = 200_000,
                  seed: int = 0) -> BayesResult:
    rng = np.random.default_rng(seed)
    a, b = prior
    pc = rng.beta(a + x_c, b + n_c - x_c, draws)
    pt = rng.beta(a + x_t, b + n_t - x_t, draws)
    rel = (pt - pc) / pc
    return BayesResult(
        prob_treatment_better=float((pt > pc).mean()),
        expected_loss_choose_treatment=float(np.maximum(pc - pt, 0).mean()),
        expected_loss_choose_control=float(np.maximum(pt - pc, 0).mean()),
        lift_ci=(float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))),
    )


