"""Mixture Sequential Probability Ratio Test (mSPRT) for always-valid inference.

Peeking at a classic test inflates false positives. mSPRT (Johari, Pekelis,
Walsh 2017 — the method behind Optimizely Stats Engine) gives a p-value that is
valid at every look, so you can stop the moment it crosses alpha.

Normal-mean version with a Gaussian mixing prior of variance tau^2 on the
effect. Suitable for means and (via normal approximation) for proportions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequentialResult:
    n: int
    lambda_: float          # mixture likelihood ratio
    p_value: float          # always-valid p-value = min(1, 1/lambda)
    effect: float
    stop: bool


def msprt(control: np.ndarray, treatment: np.ndarray, tau: float | None = None,
          alpha: float = 0.05) -> SequentialResult:
    """Evaluate the mSPRT statistic on everything observed so far."""
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    n = min(len(c), len(t))
    if n < 2:
        return SequentialResult(n, 1.0, 1.0, 0.0, False)
    c, t = c[:n], t[:n]
    sigma2 = (c.var(ddof=1) + t.var(ddof=1)) / 2
    if tau is None:  # default prior scale: 10% of the noise level (effects are usually small)
        tau = 0.1 * np.sqrt(sigma2) if sigma2 > 0 else 0.1
    zn = t.mean() - c.mean()
    v = 2 * sigma2 / n
    lam = np.sqrt(v / (v + tau**2)) * np.exp(tau**2 * zn**2 / (2 * v * (v + tau**2)))
    p = float(min(1.0, 1.0 / lam))
    return SequentialResult(n, float(lam), p, float(zn), p < alpha)


def msprt_path(control: np.ndarray, treatment: np.ndarray, every: int = 100, **kw) -> list[SequentialResult]:
    """Replay a test as if peeking every `every` users; returns the p-value path."""
    n = min(len(control), len(treatment))
    return [msprt(control[:k], treatment[:k], **kw) for k in range(every, n + 1, every)]
