"""abtest — a small, well-tested A/B experimentation toolkit.

    power      sample-size and MDE calculations
    frequentist two-sample tests (proportions and means), CIs
    cuped      variance reduction with pre-experiment covariates
    sequential always-valid p-values via mSPRT (safe to peek)
    bayesian   posterior P(treatment > control), expected loss
    hte        heterogeneous treatment effects by segment
"""
from .bayesian import BayesResult, beta_binomial
from .cuped import cuped_adjust
from .frequentist import TestResult, ttest_means, ztest_proportions
from .hte import segment_effects
from .power import mde_proportions, sample_size_means, sample_size_proportions
from .sequential import SequentialResult, msprt

__all__ = [
    "BayesResult",
    "SequentialResult",
    "TestResult",
    "beta_binomial",
    "cuped_adjust",
    "mde_proportions",
    "msprt",
    "sample_size_means",
    "sample_size_proportions",
    "segment_effects",
    "ttest_means",
    "ztest_proportions",
]
__version__ = "0.1.0"
