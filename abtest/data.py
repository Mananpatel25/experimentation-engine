"""Datasets: Criteo Uplift (real, 13.9M rows) and a synthetic generator for tests/demos."""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
CRITEO_PATH = os.path.join("data", "criteo-uplift-v2.1.csv.gz")


def load_criteo(nrows: int | None = None, path: str = CRITEO_PATH) -> pd.DataFrame:
    """Criteo Uplift Modeling dataset: 12 features, treatment, conversion, visit, exposure.
    ~300MB gzipped. Pass nrows for a quick sample."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {CRITEO_URL} → {path} (~300MB)")
        urllib.request.urlretrieve(CRITEO_URL, path)
    return pd.read_csv(path, nrows=nrows)


def synthetic(n: int = 200_000, base_rate: float = 0.04, lift: float = 0.004, seed: int = 42,
              covariate_corr: float = 0.6, n_segments: int = 4) -> pd.DataFrame:
    """Users with a pre-period covariate correlated with outcome (so CUPED has something to do),
    a segment with heterogeneous effects, and a true absolute lift of `lift`."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    segment = rng.integers(0, n_segments, n)
    pre_spend = rng.gamma(2.0, 20.0, n)                       # pre-experiment covariate
    latent = covariate_corr * (pre_spend - pre_spend.mean()) / pre_spend.std() + rng.normal(0, 1, n)
    seg_lift = lift * np.array([0.25, 0.75, 1.25, 1.75])[segment % 4]  # effect varies by segment
    p = base_rate + treatment * seg_lift + 0.02 * latent
    conversion = (rng.random(n) < np.clip(p, 0.001, 0.999)).astype(int)
    revenue = conversion * rng.gamma(2.0, 25.0, n) + 0.3 * pre_spend * rng.random(n)
    return pd.DataFrame({"user_id": np.arange(n), "treatment": treatment, "segment": segment,
                         "pre_spend": pre_spend, "conversion": conversion, "revenue": revenue})
