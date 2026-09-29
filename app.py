import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from src import analysis as an
from src.db import QueryError, build_db, run_query
from src.queries import QUERIES

st.set_page_config(page_title="Fitbit Recovery Lab", page_icon="🔥", layout="wide")
ORANGE, TEAL, LIME, GREY = "#FC4C02", "#14B8A6", "#A3E635", "#6B7280"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TRACK = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1400' height='900'><g fill='none' stroke='white' stroke-width='2'>"
         + "".join(f"<ellipse cx='700' cy='450' rx='{640 - i * 38}' ry='{380 - i * 38}'/>" for i in range(7))
         + "<line x1='700' y1='70' x2='700' y2='450'/></g></svg>")
BEAT = [(0, 40), (30, 40), (38, 40), (46, 10), (54, 70), (62, 26), (70, 40), (100, 40)]
ECG = " ".join(f"{b * 100 + x},{y}" for b in range(6) for x, y in BEAT)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root {{color-scheme:dark}}
.stApp {{background: radial-gradient(1100px 600px at 90% -10%, rgba(252,76,2,.28), transparent 60%),
  radial-gradient(900px 520px at -10% 35%, rgba(20,184,166,.16), transparent 55%), linear-gradient(180deg,#0B0F19,#04060B);
  font-family:'Inter',sans-serif;color:#F3F4F6}}
.stApp:before {{content:"";position:fixed;inset:0;background:url("{TRACK}") center/cover no-repeat;opacity:.04;pointer-events:none;z-index:0}}
[data-testid="stHeader"] {{background:transparent}}
.block-container {{padding-top:2.5rem;max-width:1250px}}

/* ---------- READABILITY: force light text on the dark theme ---------- */
.stApp p, .stApp li, .stApp label, .stApp span, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{color:#F3F4F6}}
[data-testid="stWidgetLabel"] p, .stApp label p {{color:#E5E7EB !important;font-weight:600;font-size:.92rem}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stApp small {{color:#AEB6C4 !important}}
h1,h2,h3,h4 {{font-family:'Oswald',sans-serif !important;letter-spacing:.6px;text-transform:uppercase;color:#FFFFFF !important}}
h3 {{border-left:5px solid {ORANGE};padding-left:12px}}
code {{background:rgba(252,76,2,.16) !important;color:#FFC9AD !important;border-radius:6px}}
a {{color:#5EEAD4 !important}}

/* ---------- HERO ---------- */
.hero {{position:relative;overflow:hidden;border-radius:24px;padding:38px 44px 96px;margin-bottom:20px;
  background:linear-gradient(115deg,#FC4C02 0%,#C2380A 45%,#3A1206 100%);
  box-shadow:0 14px 44px rgba(252,76,2,.30);border:1px solid rgba(255,255,255,.14)}}
.hero .kicker {{font-family:Oswald;letter-spacing:5px;font-size:.85rem;color:#FFE1D2 !important;font-weight:500}}
.hero h1 {{font-size:4rem;margin:.15rem 0 .4rem;line-height:1;color:#fff !important;border:0;padding:0;text-shadow:0 2px 12px rgba(0,0,0,.35)}}
.hero p {{max-width:720px;margin:0;color:#FFFFFF !important;font-size:1.05rem;line-height:1.55;text-shadow:0 1px 8px rgba(0,0,0,.45)}}
.hero svg {{position:absolute;left:0;right:0;bottom:8px;width:100%;height:64px;opacity:.55}}
.hero polyline {{fill:none;stroke:#fff;stroke-width:3;stroke-linejoin:round;stroke-dasharray:700;stroke-dashoffset:700;animation:ecg 3.2s linear infinite}}
@keyframes ecg {{to {{stroke-dashoffset:-700}}}}

/* ---------- KPI CARDS ---------- */
[data-testid="stMetric"] {{background:linear-gradient(160deg,rgba(255,255,255,.09),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.16);
  border-radius:16px;padding:16px 20px;border-top:3px solid {ORANGE}}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{font-family:Oswald;color:#FF7A33 !important;font-size:2.2rem}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{text-transform:uppercase;letter-spacing:1.2px;font-size:.76rem;color:#E5E7EB !important;opacity:1;font-weight:600}}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {{gap:8px;flex-wrap:wrap;margin:8px 0 14px}}
.stTabs [data-baseweb="tab"] {{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);border-radius:999px;padding:8px 18px;height:auto}}
.stTabs [data-baseweb="tab"] p {{color:#E5E7EB !important;font-weight:600}}
.stTabs [data-baseweb="tab"]:hover {{background:rgba(252,76,2,.25)}}
.stTabs [aria-selected="true"] {{background:{ORANGE} !important;border-color:{ORANGE}}}
.stTabs [aria-selected="true"] p {{color:#fff !important}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {{display:none}}

/* ---------- INPUTS ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextArea textarea, .stNumberInput input {{
  background:#111827 !important;color:#F9FAFB !important;border:1px solid rgba(255,255,255,.22) !important;border-radius:10px !important}}
[data-baseweb="select"] *, .stTextArea textarea {{color:#F9FAFB !important}}
[data-baseweb="select"] svg {{fill:#F9FAFB !important}}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{background:#111827 !important}}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {{color:#F9FAFB !important;background:#111827 !important}}
[data-baseweb="popover"] li:hover {{background:rgba(252,76,2,.30) !important}}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {{color:#AEB6C4 !important}}
[data-testid="stSlider"] [role="slider"] {{background:{ORANGE} !important}}
[data-testid="stThumbValue"] {{color:#FFB08A !important;font-weight:700}}
.stRadio label p, .stCheckbox label p, .stToggle label p {{color:#F3F4F6 !important}}
.stButton button, .stDownloadButton button {{border-radius:10px;font-weight:700;border:1px solid rgba(255,255,255,.25);background:#1F2937;color:#fff}}
.stButton button[kind="primary"] {{background:{ORANGE};border-color:{ORANGE};color:#fff}}
.stButton button:hover, .stDownloadButton button:hover {{border-color:{ORANGE};color:#fff}}

/* ---------- ALERTS, CARDS, CHIPS ---------- */
[data-testid="stAlert"] {{background:rgba(255,255,255,.07) !important;border:1px solid rgba(255,255,255,.16);border-radius:12px}}
[data-testid="stAlert"] p, [data-testid="stAlert"] div {{color:#F3F4F6 !important}}
[data-testid="stExpander"] {{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.14);border-radius:12px}}
[data-testid="stExpander"] summary p {{color:#F3F4F6 !important;font-weight:600}}
.card {{background:linear-gradient(160deg,rgba(255,255,255,.09),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.16);border-radius:16px;padding:16px 20px;color:#F3F4F6;line-height:1.6}}
.card b {{font-family:Oswald;letter-spacing:1.5px;color:#FF7A33}}
.chip {{display:inline-block;padding:6px 14px;border-radius:999px;margin:4px 6px 4px 0;font-weight:600;font-size:.85rem;border:1px solid}}
.chip.on {{background:rgba(252,76,2,.22);border-color:{ORANGE};color:#FFD2B8}}
.chip.off {{border-color:#4B5563;color:#9CA3AF;background:rgba(255,255,255,.03)}}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {{background:#0D1320;border-right:1px solid rgba(255,255,255,.12)}}
section[data-testid="stSidebar"] * {{color:#F3F4F6}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:#AEB6C4 !important}}
</style>
<div class="hero"><div class="kicker">STRAVA-STYLE FITNESS ANALYTICS</div><h1>Recovery Lab</h1>
<p>33 real Fitbit athletes, 30 days of steps, sleep and heart rate. Does more movement mean better recovery? Explore it with SQL, statistics and ML.</p>
<svg viewBox="0 0 600 80" preserveAspectRatio="none"><polyline points="{ECG}"/></svg></div>
""", unsafe_allow_html=True)


def style(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(t=48, b=10, l=10, r=10), font=dict(family="Inter", size=13, color="#F3F4F6"),
                      colorway=[ORANGE, TEAL, LIME, "#F59E0B", "#8B5CF6"], title_font=dict(family="Oswald", size=19, color="#FFFFFF"),
                      legend=dict(font=dict(color="#F3F4F6")))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.10)", tickfont=dict(color="#E5E7EB"), title_font=dict(color="#E5E7EB"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.10)", tickfont=dict(color="#E5E7EB"), title_font=dict(color="#E5E7EB"))
    return fig


def show(fig, h=340):
    st.plotly_chart(style(fig, h), width="stretch")


def bar(df, x, y, title, **kw):
    return px.bar(df, x=x, y=y, title=title, text_auto=".0f", color_discrete_sequence=[ORANGE], **kw)


def gauge(val, goal, title, suffix=""):
    f = go.Figure(go.Indicator(mode="gauge+number", value=val, number=dict(suffix=suffix, font=dict(family="Oswald", size=34)),
                               title=dict(text=title, font=dict(size=14)),
                               gauge=dict(axis=dict(range=[0, max(goal * 1.5, val * 1.1)]), bar=dict(color=ORANGE if val >= goal else TEAL),
                                          bgcolor="rgba(255,255,255,.06)", borderwidth=0,
                                          threshold=dict(line=dict(color="white", width=4), value=goal))))
    return f


def chip(txt, on):
    return f'<span class="chip {"on" if on else "off"}">{txt}</span>'


@st.cache_data(show_spinner="Warming up…")
def get_data(excl: bool):
    build_db()
    d = an.load_daily(excl)
    d["Weekday"] = d.Date.dt.day_name()
    return d


@st.cache_data(show_spinner="Running SQL…")
def q(key):
    return run_query(QUERIES[key][2])


@st.cache_data(show_spinner="Cross-validating 6 models…")
def get_models(excl):
    return an.compare_models(get_data(excl))


with st.sidebar:
    st.markdown("### 🔥 Controls")
    excl = st.toggle("Exclude non-wear days", True, help="0 steps + 1440 sedentary minutes = tracker not worn.")
    st.caption("Pipeline: raw CSVs → clean/merge → SQLite → SQL + stats + ML → this dashboard.")
try:
    df = get_data(excl)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load data: {e}")
    st.stop()

k = st.columns(5)
k[0].metric("Athletes", df.Id.nunique())
k[1].metric("User-days", f"{len(df):,}")
k[2].metric("Avg steps/day", f"{df.TotalSteps.mean():,.0f}")
k[3].metric("Avg sleep", f"{df.TotalMinutesAsleep.mean() / 60:.1f} h")
k[4].metric("Nights < 7h", f"{(df.GoodSleep.dropna() == 0).mean() * 100:.0f}%")

tabs = st.tabs(["🏃 Athlete Passport", "🏆 Leaderboard", "⏱️ Daily Rhythm", "🔬 Explorer", "🗄️ SQL Lab", "🤖 ML Lab", "📌 Insights"])

# ============ Athlete Passport
with tabs[0]:
    ids = [int(i) for i in sorted(df.Id.unique())]
    lab = {i: f"Athlete {n + 1:02d} · …{str(i)[-4:]}" for n, i in enumerate(ids)}
    c1, c2, c3 = st.columns([2, 1, 1])
    aid = c1.selectbox("Pick an athlete", ids, format_func=lambda i: lab[i])
    goal = c2.slider("Daily step goal", 5000, 20000, 10000, 500)
    sgoal = c3.slider("Sleep goal (h)", 6.0, 9.0, 7.0, 0.5)
    a = df[df.Id == aid].sort_values("Date")
    steps, very = a.TotalSteps.mean(), a.VeryActiveMinutes.mean()
    sl = a.TotalMinutesAsleep.dropna()
    has_sleep, sleep_h = len(sl) > 0, (sl.mean() / 60 if len(sl) else 0)
    kind = "🏅 Marathoner" if steps >= goal else "🚶 Steady Strider" if steps >= .75 * goal else "🌿 Casual Walker" if steps >= .5 * goal else "🪑 Desk Warrior"
    st.markdown(f"### {lab[aid]} — {kind}")
    g = st.columns(3)
    g[0].plotly_chart(style(gauge(steps, goal, "Avg steps / day"), 250), width="stretch")
    if has_sleep:
        g[1].plotly_chart(style(gauge(sleep_h, sgoal, "Avg sleep (h)", " h"), 250), width="stretch")
    else:
        g[1].info("No sleep logged by this athlete.")
    mix = go.Figure(go.Pie(labels=["Very active", "Fairly active", "Lightly active"],
                           values=[very, a.FairlyActiveMinutes.mean(), a.LightlyActiveMinutes.mean()], hole=.62,
                           marker=dict(colors=[ORANGE, LIME, TEAL]), textinfo="percent"))
    mix.update_layout(title="Activity mix (min/day)", showlegend=True)
    g[2].plotly_chart(style(mix, 250), width="stretch")

    n_goal = int((a.TotalSteps >= goal).sum())
    pct_sleep = (sl >= sgoal * 60).mean() if len(sl) else 0
    tip = (f"You average **{steps:,.0f}** steps, **{goal - steps:,.0f}** short of your goal — a brisk 15-minute walk adds roughly 1,500."
           if steps < goal else f"Goal smashed on average (**{steps:,.0f}** steps).")
    if has_sleep and sleep_h < sgoal:
        tip += f" Sleep is **{sleep_h:.1f} h** vs {sgoal:g} h: in this dataset 10k+ step days go with ~58 min less sleep, so try earlier-day workouts."
    b1, b2 = st.columns([3, 2])
    b1.markdown('<div class="card"><b>BADGES</b><br>' + chip(f"🔥 Goal crusher · {n_goal} days", n_goal >= 3)
                + chip("🌙 Sleep champion", has_sleep and pct_sleep >= .6) + chip("⚡ Intensity beast", very >= 20)
                + chip("📅 Consistent · 28+ days", len(a) >= 28) + chip("❤️ Heart-tracked", a.HR_mean.notna().any()) + "</div>",
                unsafe_allow_html=True)
    b2.markdown(f'<div class="card"><b>COACH SAYS</b><br>{tip}</div>', unsafe_allow_html=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=a.Date, y=a.TotalSteps, name="Steps", marker_color=[ORANGE if s >= goal else GREY for s in a.TotalSteps], secondary_y=False)
    if has_sleep:
        fig.add_scatter(x=a.Date, y=a.TotalMinutesAsleep / 60, name="Sleep (h)", mode="lines+markers", line=dict(color=TEAL, width=3), secondary_y=True)
    fig.add_hline(y=goal, line_dash="dash", line_color="white", annotation_text="goal", secondary_y=False)
    fig.update_layout(title="Daily steps (orange = goal hit) vs sleep", legend=dict(orientation="h", y=1.12))
    show(fig, 360)

    c1, c2 = st.columns(2)
    a2 = a.assign(wk=(a.Date - pd.to_timedelta(a.Date.dt.dayofweek, unit="D")).dt.normalize(), dow=a.Date.dt.dayofweek)
    pv = a2.pivot_table(index="dow", columns="wk", values="TotalSteps", aggfunc="first").reindex(range(7))
    heat = go.Figure(go.Heatmap(z=pv.values, x=[d.strftime("%d %b") for d in pv.columns], y=DAYS, xgap=4, ygap=4,
                                colorscale=[[0, "#1F2937"], [.5, "#B45309"], [1, ORANGE]],
                                hovertemplate="%{y}, week of %{x}<br>%{z:,.0f} steps<extra></extra>"))
    heat.update_layout(title="Training calendar", yaxis=dict(autorange="reversed"))
    c1.plotly_chart(style(heat, 300), width="stretch")
    coh = {"Steps": df.TotalSteps.mean(), "Very-active min": df.VeryActiveMinutes.mean(), "Calories": df.Calories.mean()}
    me = {"Steps": steps, "Very-active min": very, "Calories": a.Calories.mean()}
    cmp_ = pd.DataFrame({"metric": list(coh), "pct_of_cohort": [me[m] / coh[m] * 100 for m in coh]})
    f = bar(cmp_, "metric", "pct_of_cohort", "You vs cohort average (100% = average)")
    f.add_hline(y=100, line_dash="dash", line_color="white")
    c2.plotly_chart(style(f, 300), width="stretch")
    if a.HR_mean.notna().any():
        hr = go.Figure()
        for c, col in [("HR_max", ORANGE), ("HR_mean", LIME), ("HR_min", TEAL)]:
            hr.add_scatter(x=a.Date, y=a[c], name=c, mode="lines+markers", line=dict(color=col))
        hr.update_layout(title="Heart rate (bpm): max, mean, min")
        show(hr, 300)

# ============ Leaderboard
with tabs[1]:
    opts = {"Avg steps": "TotalSteps", "Very-active min": "VeryActiveMinutes", "Calories": "Calories", "Sleep (min)": "TotalMinutesAsleep"}
    c1, c2 = st.columns([2, 1])
    m = c1.radio("Rank athletes by", list(opts), horizontal=True)
    n = c2.slider("Show top", 5, 15, 10)
    lb = df.groupby("Id")[opts[m]].mean().dropna().sort_values(ascending=False).head(n).reset_index()
    lb["Athlete"] = lb.Id.map(lambda i: f"…{str(i)[-4:]}")
    f = px.bar(lb.iloc[::-1], x=opts[m], y="Athlete", orientation="h", text_auto=".0f", title=f"Top {n} by {m.lower()}", color_discrete_sequence=[ORANGE])
    show(f, 90 + 34 * n)
    seg = q("segments")
    c1, c2 = st.columns(2)
    c1.plotly_chart(style(bar(seg, "segment", "users", "Athletes per activity segment"), 320), width="stretch")
    c2.plotly_chart(style(bar(seg, "segment", "avg_sleep_min", "Avg sleep (min) by segment"), 320), width="stretch")
    try:
        u, kk = an.personas(df)
        f = px.scatter(u, x="avg_steps", y="avg_sleep_min", color="persona", size="avg_very_active_min", hover_data=["Id"],
                       title=f"Behavioural personas (KMeans, k={kk}) — bubble = very-active min")
        show(f, 380)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Personas unavailable: {e}")

# ============ Daily Rhythm
with tabs[2]:
    mcol = st.selectbox("Metric", ["StepTotal", "Calories", "TotalIntensity", "HR_mean"], format_func=lambda c: {"StepTotal": "Steps", "Calories": "Calories", "TotalIntensity": "Intensity", "HR_mean": "Heart rate"}[c])
    hm = run_query(f"SELECT CAST(strftime('%w',Hour) AS INTEGER) AS dow, CAST(strftime('%H',Hour) AS INTEGER) AS hr, AVG({mcol}) AS v FROM hourly GROUP BY dow, hr")
    hm["dow"] = (hm.dow - 1) % 7
    pv = hm.pivot(index="dow", columns="hr", values="v").reindex(range(7))
    f = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=DAYS, colorscale="YlOrRd", xgap=2, ygap=2, colorbar=dict(title="avg")))
    f.update_layout(title="When do athletes move? (weekday × hour)", xaxis_title="hour of day", yaxis=dict(autorange="reversed"))
    show(f, 380)
    hr_ = q("hourly")
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_bar(x=hr_.hour_of_day, y=hr_.avg_steps, name="Steps", marker_color=ORANGE)
    f.add_scatter(x=hr_.hour_of_day, y=hr_.avg_hr, name="Heart rate", mode="lines+markers", line=dict(color=TEAL, width=3), secondary_y=True)
    f.update_layout(title="24-hour rhythm: steps vs heart rate", legend=dict(orientation="h", y=1.12))
    show(f, 340)

# ============ Explorer
with tabs[3]:
    vars_ = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", "Calories",
             "METs_mean", "HR_mean", "HR_min", "TotalMinutesAsleep", "SleepEfficiency"]
    c = st.columns(3)
    x = c[0].selectbox("X axis", vars_, 0)
    y = c[1].selectbox("Y axis", vars_, 9)
    col = c[2].selectbox("Colour by", ["StepsBand", "Weekday", "IsWeekend", "None"])
    d = df[[x, y] + ([col] if col != "None" else [])].dropna()
    if len(d) > 10 and x != y:
        rho, p = stats.spearmanr(d[x], d[y])
        f = px.scatter(d, x=x, y=y, color=(d[col].astype(str) if col != "None" else None), opacity=.6, title=f"{y} vs {x}  (Spearman ρ = {rho:.2f}, p = {p:.4f}, n = {len(d)})")
        kk, ii = np.polyfit(d[x], d[y], 1)
        xs = np.array([d[x].min(), d[x].max()])
        f.add_scatter(x=xs, y=kk * xs + ii, mode="lines", line=dict(color="white", dash="dash"), name="trend")
        show(f, 420)
        if "SedentaryMinutes" in (x, y) and "TotalMinutesAsleep" in (x, y):
            st.caption("⚠️ Sedentary minutes include time asleep, so this link is partly by construction.")
    else:
        st.info("Pick two different variables with enough data.")
    cm = an.corr_matrix(df)
    show(px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Spearman correlation matrix"), 560)
    st.subheader("Statistical tests")
    st.dataframe(an.stat_tests(df), hide_index=True, width="stretch")
    st.caption("Non-parametric tests. User-days are pooled, so p-values are optimistic. Fitbit has no stress/anxiety data, so sleep and minimum heart rate are recovery proxies.")
    with st.expander("Missing values & how they were handled"):
        st.dataframe(an.missingness(df), hide_index=True, width="stretch")
        st.dataframe(q("completeness"), hide_index=True, width="stretch")

# ============ SQL Lab
with tabs[4]:
    st.write("Nine ready-made analyses (CTEs, CASE, LAG, RANK) plus a safe read-only editor. Tables: `daily`, `hourly`.")
    key = st.selectbox("Analysis", list(QUERIES), format_func=lambda k_: QUERIES[k_][0])
    st.info(QUERIES[key][1])
    sql_text = st.text_area("SQL (SELECT / WITH only)", QUERIES[key][2].strip(), height=220)
    if st.button("▶ Run query", type="primary"):
        try:
            res = run_query(sql_text)
            st.dataframe(res, hide_index=True, width="stretch")
            st.download_button("Download CSV", res.to_csv(index=False), f"{key}.csv")
        except QueryError as e:
            st.error(str(e))

# ============ ML Lab
with tabs[5]:
    st.write("**Task:** will an athlete sleep ≥ 7 h given that day's activity? Grouped 5-fold CV by athlete (no athlete in both train and test). `SedentaryMinutes` is excluded — it contains sleep time (leakage).")
    try:
        res = get_models(excl)
        st.dataframe(res, hide_index=True, width="stretch")
        f = px.bar(res, x="model", y="roc_auc", error_y="roc_auc_std", text_auto=".2f", title="ROC-AUC (0.5 = coin flip)", color_discrete_sequence=[ORANGE])
        f.add_hline(y=.5, line_dash="dash", line_color="white")
        show(f, 340)
        best = res[res.model != "Baseline (majority)"].iloc[0]
        st.warning(f"Honest result: best model = {best.model} (AUC {best.roc_auc:.2f}). Daily activity alone barely predicts sleep — a useful negative finding.")
        st.subheader("🎛️ Sleep predictor (illustrative)")
        model = an.fit_best(df, best.model)
        c = st.columns(4)
        steps_ = c[0].slider("Steps", 0, 30000, 8000, 500)
        very_ = c[1].slider("Very-active min", 0, 180, 15)
        fair_ = c[2].slider("Fairly-active min", 0, 120, 10)
        light_ = c[3].slider("Lightly-active min", 0, 500, 200)
        wk = st.checkbox("Weekend")
        row = pd.DataFrame([{"TotalSteps": steps_, "TotalDistance": steps_ * 0.00075, "VeryActiveMinutes": very_, "FairlyActiveMinutes": fair_,
                             "LightlyActiveMinutes": light_, "Calories": 1500 + steps_ * 0.06 + very_ * 8, "IsWeekend": int(wk)}])[an.FEATURES]
        pr = model.predict_proba(row)[0, 1]
        st.plotly_chart(style(gauge(pr * 100, 50, "Chance of ≥ 7 h sleep", "%"), 250), width="stretch")
    except Exception as e:  # noqa: BLE001
        st.error(f"Modelling failed: {e}")

# ============ Insights
with tabs[6]:
    s = df.dropna(subset=["TotalMinutesAsleep"])
    hi, lo = s[s.TotalSteps >= 10000].TotalMinutesAsleep.mean(), s[s.TotalSteps < 5000].TotalMinutesAsleep.mean()
    st.markdown(f"""
### Key findings
- **10k+ step days → ~{lo - hi:.0f} min less sleep** than <5k days ({hi:.0f} vs {lo:.0f} min) — significant, small effect.
- **{(s.TotalMinutesAsleep < 420).mean() * 100:.0f}% of logged nights are under 7 h.**
- More very-active minutes go with a **lower minimum heart rate** (fitness signal).
- **No weekend effect** on steps. Activity alone barely predicts sleep (AUC ≈ 0.5).

### Recommendations (Bellabeat-style, for the marketing team)
1. **Wear detection:** flag non-wear days so metrics aren't distorted (7.7% of days here).
2. **Smart nudges:** notify inactive users; suggest earlier-day workouts to protect sleep.
3. **Sleep-debt alerts** after 3+ nights under 7 h (see SQL Lab watchlist).
4. **Collect more data:** only 24 / 14 / 8 of 33 athletes have sleep / heart-rate / weight data; add self-reported stress and mood.

*Limitations: 33 users, 30 days, pooled user-days, observational data (no causality).*
""")