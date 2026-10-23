# Experimentation Engine

A small, tested A/B testing toolkit plus an end-to-end experiment readout on the Criteo Uplift dataset (13.9M real users).
Built because most "A/B test" portfolio projects stop at a t-test; real experimentation teams need the five things below.

| Module | What it does | Why it matters |
|---|---|---|
| `abtest.power` | Sample size & MDE for proportions and means | Know *before* launch whether the test can detect the lift you care about |
| `abtest.frequentist` | Two-sample z/t tests with effect sizes and CIs | The primary readout |
| `abtest.cuped` | CUPED variance reduction (Deng et al. 2013) | Same lift, tighter CI; 10–40% fewer users needed |
| `abtest.sequential` | mSPRT always-valid p-values (Johari et al. 2017) | Peek at results daily without inflating false positives — the Optimizely Stats Engine method |
| `abtest.bayesian` | Beta-Binomial posterior, P(T>C), expected loss | "What's the probability we're wrong if we ship?" |
| `abtest.hte` | Lift by segment with Benjamini-Hochberg correction | Find who the feature helps or hurts, without p-hacking |

## Quick start

```bash
pip install -e ".[dev,app]"
pytest -q                                  # 11 tests, including a false-positive-rate check on mSPRT
python analysis/run_analysis.py            # full readout on synthetic data (seconds)
python analysis/run_analysis.py --criteo   # same on real Criteo data (downloads ~300MB)
streamlit run app/streamlit_app.py         # interactive planner + readout
```

```python
from abtest import sample_size_proportions, ztest_proportions, cuped_adjust, msprt, beta_binomial

sample_size_proportions(p0=0.04, mde=0.004)          # → 46,000 per arm
ztest_proportions(x_c=4050, n_c=100_000, x_t=4410, n_t=100_000).summary()
# conversion: 0.0405 → 0.0441 (+8.9%), 95% CI [+0.0018, +0.0054], p=0.0001
```

## The readout (`analysis/run_analysis.py`)

Produces a decision memo in seven steps: planning → primary metric → CUPED on revenue → sequential replay ("could we have stopped early?") → Bayesian view → segment effects → **ship / don't ship** with projected $ impact and a confidence interval. Sample output on synthetic data:

> **SHIP** — primary metric significant (p=0.0001), P(T>C)=100%, CUPED cut variance 12%, mSPRT would have stopped at 64% of the sample, no segment shows harm. Projected impact at 1M users: $108K/yr (CI $55K–$161K).

## What I'd add next
- Stratified assignment + Sample Ratio Mismatch (SRM) check on ingest
- Delta method for ratio metrics (revenue per session)
- Uplift modeling (causalml) to go from "who converted" to "who was *caused* to convert"

## References
Deng, Xu, Kohavi, Walker (2013) *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data*.
Johari, Pekelis, Walsh (2017) *Always Valid Inference: Bringing Sequential Analysis to A/B Testing*.
Diemert et al. (2018) *A Large Scale Benchmark for Uplift Modeling* (Criteo dataset).
