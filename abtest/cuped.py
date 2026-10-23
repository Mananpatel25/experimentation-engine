"""CUPED: Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).

Y_adj = Y - theta * (X - mean(X)),  theta = cov(X, Y) / var(X)

X is a pre-experiment covariate (e.g. the user's spend in the 30 days before
the test). Because X is independent of assignment, the adjusted metric has the
same expected lift but lower variance → smaller sample size / tighter CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CupedResult:
    y_adj: np.ndarray
    theta: float
    variance_reduction: float  # 1 - var(Y_adj)/var(Y)


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> CupedResult:
    y, x = np.asarray(y, float), np.asarray(x, float)
    if len(y) != len(x):
        raise ValueError("y and x must be the same length")
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    vr = 1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1)
    return CupedResult(y_adj, float(theta), float(vr))
