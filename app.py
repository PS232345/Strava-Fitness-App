"""Fitbit Recovery Lab — Streamlit dashboard: SQL analysis + statistics + ML."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analysis as an
from src.db import QueryError, build_db, run_query
from src.queries import QUERIES

st.set_page_config(page_title="Fitbit Recovery Lab", page_icon="🌙", layout="wide")
TEAL, SLATE = "#0F766E", "#334155"


@st.cache_data(show_spinner="Loading data…")
def get_data(exclude_nonwear: bool):
    build_db()
    return an.load_daily(exclude_nonwear)


@st.cache_data(show_spinner="Running SQL…")
def q(key: str):
    return run_query(QUERIES[key][2])


@st.cache_data(show_spinner="Cross-validating 6 models…")
def get_models(exclude_nonwear: bool):
    return an.compare_models(get_data(exclude_nonwear))


def bar(df, x, y, title, **kw):
    fig = px.bar(df, x=x, y=y, title=title, text_auto=".0f", color_discrete_sequence=[TEAL], **kw)
    fig.update_layout(height=340, margin=dict(t=50, b=10), title_font_size=15)
    return fig


st.title("🌙 Fitbit Recovery Lab")
st.caption("Does movement help or hurt recovery? SQL analysis, statistics and ML on 33 Fitbit users (12 Apr–12 May 2016).")

with st.sidebar:
    st.header("Filters")
    excl = st.toggle("Exclude non-wear days", True, help="Days with 0 steps and 1440 sedentary minutes = tracker not worn.")
    st.markdown("**Workflow**\n1. Merge & clean CSVs\n2. SQLite + SQL analyses\n3. EDA & stats tests\n4. ML comparison\n5. Insights")

try:
    df = get_data(excl)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load data: {e}")
    st.stop()

tabs = st.tabs(["🏠 Overview", "🗄️ SQL Lab", "🔍 EDA", "🔗 Relationships & Tests", "🤖 ML & Personas", "✅ Insights"])

# ---------------- Overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Users", df.Id.nunique())
    c[1].metric("User-days", f"{len(df):,}")
    c[2].metric("Avg steps/day", f"{df.TotalSteps.mean():,.0f}")
    c[3].metric("Avg sleep", f"{df.TotalMinutesAsleep.mean() / 60:.1f} h")
    c[4].metric("Nights < 7h", f"{(df.GoodSleep.dropna() == 0).mean() * 100:.0f}%")
    st.subheader("Missing values & how they were handled")
    m = an.missingness(df)
    l, r = st.columns([1, 2])
    l.plotly_chart(bar(m, "column", "missing_pct", "% missing"), width="stretch")
    r.dataframe(m, hide_index=True, width="stretch")
    st.subheader("Data completeness (SQL)")
    st.dataframe(q("completeness"), hide_index=True, width="stretch")

# ---------------- SQL Lab
with tabs[1]:
    st.write("Nine ready-made analyses (joins, CTEs, CASE, window functions) plus a safe read-only editor. Tables: `daily`, `hourly`.")
    key = st.selectbox("Analysis", list(QUERIES), format_func=lambda k: QUERIES[k][0])
    title, question, sql = QUERIES[key]
    st.info(question)
    sql_text = st.text_area("SQL (editable, SELECT/WITH only)", sql.strip(), height=220)
    if st.button("Run query", type="primary"):
        try:
            res = run_query(sql_text)
            st.dataframe(res, hide_index=True, width="stretch")
            st.download_button("Download CSV", res.to_csv(index=False), f"{key}.csv")
        except QueryError as e:
            st.error(str(e))

# ---------------- EDA
with tabs[2]:
    a, b = st.columns(2)
    seg = q("segments")
    a.plotly_chart(bar(seg, "segment", "users", "Users per activity segment"), width="stretch")
    b.plotly_chart(bar(seg, "segment", "avg_sleep_min", "Avg sleep (min) by segment"), width="stretch")
    a, b = st.columns(2)
    wd = q("weekday")
    a.plotly_chart(bar(wd, "weekday", "avg_steps", "Steps by weekday"), width="stretch")
    ss = q("sleep_vs_steps")
    b.plotly_chart(bar(ss, "steps_band", "avg_sleep_min", "More steps, less sleep?"), width="stretch")
    hr = q("hourly")
    fig = px.line(hr, x="hour_of_day", y=["avg_steps", "avg_hr"], markers=True, title="24-hour rhythm: steps and heart rate")
    fig.update_layout(height=340, margin=dict(t=50, b=10), legend_title="")
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="TotalSteps", nbins=30, title="Distribution of daily steps",
                                color_discrete_sequence=[TEAL]), width="stretch")
    s = df.dropna(subset=["TotalMinutesAsleep"])
    sc = px.scatter(s, x="TotalSteps", y="TotalMinutesAsleep", opacity=0.5, color_discrete_sequence=[SLATE],
                    title="Steps vs sleep minutes (each dot = one user-night)")
    k, i = np.polyfit(s.TotalSteps, s.TotalMinutesAsleep, 1)
    xs = np.array([s.TotalSteps.min(), s.TotalSteps.max()])
    sc.add_scatter(x=xs, y=k * xs + i, mode="lines", line=dict(color="crimson"), name="trend")
    b.plotly_chart(sc, width="stretch")
    st.subheader("User segments (behavioural personas, KMeans)")
    try:
        u, k_ = an.personas(df)
        st.caption(f"k={k_} chosen by silhouette score. Ordered by average steps.")
        st.plotly_chart(px.scatter(u, x="avg_steps", y="avg_sleep_min", color="persona", size="avg_very_active_min",
                                   hover_data=["Id"], title="Personas: steps vs sleep (bubble = very-active minutes)"), width="stretch")
        st.dataframe(u.groupby("persona").mean(numeric_only=True).round(0).assign(users=u.persona.value_counts()), width="stretch")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Personas unavailable: {e}")

# ---------------- Relationships
with tabs[3]:
    st.write("Stress and anxiety are not recorded by Fitbit, so **recovery proxies** are used: sleep duration/efficiency and minimum heart rate.")
    cm = an.corr_matrix(df)
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Spearman correlation matrix", height=560), width="stretch")
    st.subheader("Statistical tests")
    tests = an.stat_tests(df)
    st.dataframe(tests, hide_index=True, width="stretch")
    st.caption("Non-parametric tests (skewed data). User-days from the same person are pooled, so p-values are optimistic — treat as evidence, not proof.")

# ---------------- ML
with tabs[4]:
    st.write("**Task:** predict whether a user sleeps ≥ 7 h from that day's activity. Grouped 5-fold CV by user (no user in both train and test). "
             "`SedentaryMinutes` is excluded because it includes sleep time (data leakage).")
    try:
        res = get_models(excl)
        st.dataframe(res, hide_index=True, width="stretch")
        fig = px.bar(res, x="model", y="roc_auc", error_y="roc_auc_std", text_auto=".2f", title="ROC-AUC (0.5 = coin flip)",
                     color_discrete_sequence=[TEAL])
        fig.add_hline(y=0.5, line_dash="dash", line_color="crimson")
        st.plotly_chart(fig, width="stretch")
        best = res[res.model != "Baseline (majority)"].iloc[0]
        st.warning(f"Honest result: best model = {best.model} (AUC {best.roc_auc:.2f}). Daily activity alone is a weak predictor of sleep — "
                   "useful negative finding; sleep likely depends on factors Fitbit does not record.")
        st.subheader("🎛️ Try the sleep predictor (illustrative)")
        model = an.fit_best(df, best.model)
        c = st.columns(4)
        steps = c[0].slider("Steps", 0, 30000, 8000, 500)
        very = c[1].slider("Very-active min", 0, 180, 15)
        fair = c[2].slider("Fairly-active min", 0, 120, 10)
        light = c[3].slider("Lightly-active min", 0, 500, 200)
        wk = st.checkbox("Weekend")
        row = pd.DataFrame([{"TotalSteps": steps, "TotalDistance": steps * 0.00075, "VeryActiveMinutes": very,
                             "FairlyActiveMinutes": fair, "LightlyActiveMinutes": light,
                             "Calories": 1500 + steps * 0.06 + very * 8, "IsWeekend": int(wk)}])[an.FEATURES]
        st.metric("Probability of ≥ 7 h sleep", f"{model.predict_proba(row)[0, 1] * 100:.0f}%")
    except Exception as e:  # noqa: BLE001
        st.error(f"Modelling failed: {e}")

# ---------------- Insights
with tabs[5]:
    s = df.dropna(subset=["TotalMinutesAsleep"])
    hi, lo = s[s.TotalSteps >= 10000].TotalMinutesAsleep.mean(), s[s.TotalSteps < 5000].TotalMinutesAsleep.mean()
    st.markdown(f"""
### Key findings
- Users on **10k+ step days sleep ~{lo - hi:.0f} min less** than on <5k step days ({hi:.0f} vs {lo:.0f} min) — significant (Kruskal-Wallis), though the effect is small.
- **{(s.TotalMinutesAsleep < 420).mean() * 100:.0f}% of logged nights are under 7 h.**
- Higher very-active minutes go with a **lower minimum heart rate** (fitness signal).
- Weekend vs weekday steps: **no significant difference**.
- Activity features barely predict sleep (ML AUC ≈ 0.5) — sleep is driven by other factors.

### Recommendations
1. Prompt users with <7 h sleep for 3+ nights with a wind-down nudge (see *Sleep-debt watchlist* in SQL Lab).
2. Encourage **earlier-day** activity: the 24-hour rhythm shows the evening step peak that may delay sleep (test with timestamped workouts).
3. Collect more sleep, weight and heart-rate data — only 24, 8 and 14 of 33 users have them; conclusions are limited.
4. Add self-reported stress/mood and external data (weather, work schedule) to model recovery properly.

### Limitations
33 users, 30 days, pooled user-days, no stress/anxiety labels, sleep logged for a subset.
""")
