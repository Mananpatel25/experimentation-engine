"""Experiment readout app. `streamlit run app/streamlit_app.py`"""
import pandas as pd
import plotly.express as px
import streamlit as st

from abtest import beta_binomial, cuped_adjust, sample_size_proportions, segment_effects, ttest_means, ztest_proportions
from abtest.data import synthetic
from abtest.sequential import msprt_path

st.set_page_config(page_title="Experimentation Engine", layout="wide")
st.title("Experimentation Engine")

tab_plan, tab_read, tab_seq, tab_hte = st.tabs(["Plan", "Readout", "Sequential", "Segments"])

with tab_plan:
    p0 = st.slider("Baseline conversion", 0.005, 0.30, 0.04, 0.005)
    rel = st.slider("Relative lift to detect", 0.01, 0.50, 0.10, 0.01)
    power = st.slider("Power", 0.5, 0.95, 0.8, 0.05)
    n = sample_size_proportions(p0, p0 * rel, power=power)
    st.metric("Users per arm", f"{n:,}")
    st.caption("Two-sided, alpha = 0.05")

uploaded = st.sidebar.file_uploader("Upload experiment CSV (treatment, conversion, revenue, pre_spend, segment)", type="csv")
df = pd.read_csv(uploaded) if uploaded else synthetic()
c, t = df[df.treatment == 0], df[df.treatment == 1]

with tab_read:
    r = ztest_proportions(int(c.conversion.sum()), len(c), int(t.conversion.sum()), len(t))
    b = beta_binomial(int(c.conversion.sum()), len(c), int(t.conversion.sum()), len(t))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Control CVR", f"{r.control_mean:.3%}")
    k2.metric("Treatment CVR", f"{r.treatment_mean:.3%}", f"{r.rel_lift:+.2%}")
    k3.metric("p-value", f"{r.p_value:.4f}")
    k4.metric("P(T > C)", f"{b.prob_treatment_better:.1%}")
    if "pre_spend" in df and "revenue" in df:
        rev = ttest_means(c.revenue, t.revenue)
        cu = cuped_adjust(df.revenue.to_numpy(), df.pre_spend.to_numpy())
        rev_adj = ttest_means(cu.y_adj[df.treatment == 0], cu.y_adj[df.treatment == 1])
        st.subheader("Revenue — raw vs CUPED")
        st.write(pd.DataFrame({"": ["raw", "CUPED"], "lift": [rev.abs_lift, rev_adj.abs_lift],
                               "CI low": [rev.ci_low, rev_adj.ci_low], "CI high": [rev.ci_high, rev_adj.ci_high],
                               "p": [rev.p_value, rev_adj.p_value]}).set_index(""))
        st.caption(f"Variance reduction {cu.variance_reduction:.1%}")

with tab_seq:
    path = msprt_path(c.conversion.to_numpy(), t.conversion.to_numpy(), every=max(500, len(c) // 100))
    pdf = pd.DataFrame({"n per arm": [p.n for p in path], "always-valid p": [p.p_value for p in path]})
    fig = px.line(pdf, x="n per arm", y="always-valid p", log_y=True, title="mSPRT always-valid p-value path (conversion)")
    fig.add_hline(y=0.05, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

with tab_hte:
    if "segment" in df:
        seg = segment_effects(df, "segment")
        fig = px.bar(seg, x="segment", y="abs_lift", error_y=seg.ci_high - seg.abs_lift,
                     error_y_minus=seg.abs_lift - seg.ci_low, title="Lift by segment (95% CI)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg.round(4))
