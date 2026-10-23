"""Sample size and minimum detectable effect (MDE)."""
from __future__ import annotations

import math

from scipy import stats


def sample_size_proportions(p0: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                            two_sided: bool = True) -> int:
    """Per-arm sample size to detect an absolute lift `mde` from baseline rate p0."""
    p1 = p0 + mde
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / mde**2)


def sample_size_means(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                      two_sided: bool = True) -> int:
    """Per-arm sample size to detect a mean difference `mde` given std `sigma`."""
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = stats.norm.ppf(power)
    return math.ceil(2 * ((z_a + z_b) * sigma / mde) ** 2)


def mde_proportions(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with n_per_arm users per arm (approximate)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * math.sqrt(2 * p0 * (1 - p0) / n_per_arm)
