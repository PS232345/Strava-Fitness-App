import time
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

.stTextArea textarea {{font-family:'JetBrains Mono',Consolas,monospace !important;font-size:.92rem;line-height:1.55;background:#0A0F1A !important;border-left:4px solid {ORANGE} !important}}
/* ---------- BADGE PILLS (clickable chips) ---------- */
.st-key-badgebox {{background:linear-gradient(160deg,rgba(255,255,255,.09),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.16);border-radius:16px;padding:16px 20px}}
.st-key-badgebox button {{border-radius:999px !important;font-weight:600;font-size:.85rem;background:rgba(255,255,255,.03) !important;border:1px solid #4B5563 !important;color:#9CA3AF !important}}
.st-key-badgebox button p {{color:inherit !important}}
.st-key-badgebox button:hover {{border-color:{ORANGE} !important;color:#FFD2B8 !important}}
.st-key-badgebox button[kind="pillsActive"], .st-key-badgebox [data-testid="stBaseButton-pillsActive"] {{background:rgba(252,76,2,.22) !important;border-color:{ORANGE} !important;color:#FFD2B8 !important}}
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


def set_aid(i):
    st.session_state["aid"] = i


def sleep_gap(d):
    s_ = d.dropna(subset=["TotalMinutesAsleep"])
    return (s_[s_.TotalSteps >= 10000].TotalMinutesAsleep.mean(), s_[s_.TotalSteps < 5000].TotalMinutesAsleep.mean())


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
    lab_inv = {v: k_ for k_, v in lab.items()}
    if st.session_state.get("aid") not in ids:
        st.session_state["aid"] = ids[0]
    aid = c1.selectbox("Pick an athlete", ids, format_func=lambda i: lab[i], key="aid")
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
    run_ = best_run = 0
    for v_ in (a.TotalSteps >= goal):
        run_ = run_ + 1 if v_ else 0
        best_run = max(best_run, run_)
    hr_on = bool(a.HR_mean.notna().any())

    hi_, lo_ = sleep_gap(df)
    tip = (f"You average <b>{steps:,.0f}</b> steps, <b>{goal - steps:,.0f}</b> short of your goal — a brisk 15-minute walk adds roughly 1,500."
           if steps < goal else f"Goal smashed on average (<b>{steps:,.0f}</b> steps).")
    if has_sleep and sleep_h < sgoal:
        tip += f" Sleep is <b>{sleep_h:.1f} h</b> vs {sgoal:g} h."
        if lo_ - hi_ > 0:
            tip += f" Across the cohort, 10k+ step days go with ~{lo_ - hi_:.0f} min less sleep than &lt;5k days, so try earlier-day workouts."

    BADGES = {
        "🔥 Goal crusher": (n_goal >= 3, n_goal / 3, f"Hit your step goal on 3+ days ({n_goal} so far)."),
        "🌙 Sleep champion": (has_sleep and pct_sleep >= .6, pct_sleep / .6, f"Reach your sleep goal on 60% of logged nights ({pct_sleep * 100:.0f}% so far)."),
        "⚡ Intensity beast": (very >= 20, very / 20, f"Average 20+ very-active minutes a day ({very:.0f} so far)."),
        "📅 Consistent": (len(a) >= 28, len(a) / 28, f"Log 28+ days ({len(a)} so far)."),
        "❤️ Heart-tracked": (hr_on, 1.0 if hr_on else 0.0, "Have heart-rate data recorded."),
    }
    b1, b2 = st.columns([3, 2])
    LBL = {"🔥 Goal crusher": f"🔥 Goal crusher · {n_goal} days", "🌙 Sleep champion": "🌙 Sleep champion",
           "⚡ Intensity beast": "⚡ Intensity beast", "📅 Consistent": "📅 Consistent · 28+ days", "❤️ Heart-tracked": "❤️ Heart-tracked"}
    with b1.container(key="badgebox"):
        st.markdown('<b style="font-family:Oswald;letter-spacing:1.5px;color:#FF7A33">BADGES</b> <span style="color:#AEB6C4;font-size:.85rem">· tap one to see how to earn it</span>', unsafe_allow_html=True)
        pick = st.pills("Badges", list(BADGES), format_func=lambda n_: ("" if BADGES[n_][0] else "🔒 ") + LBL[n_],
                        label_visibility="collapsed", key=f"badge_{aid}")
        if pick:
            ok_, prog_, rule_ = BADGES[pick]
            st.progress(float(min(max(prog_, 0), 1)), text=("Earned. " if ok_ else "Not yet. ") + rule_)
    b2.markdown(f'<div class="card"><b>COACH SAYS</b><br>{tip}</div>', unsafe_allow_html=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=a.Date, y=a.TotalSteps, name="Steps", marker_color=[ORANGE if s >= goal else GREY for s in a.TotalSteps], secondary_y=False,
                hovertemplate="%{x|%a %d %b}<br>%{y:,.0f} steps<extra></extra>")
    if has_sleep:
        fig.add_scatter(x=a.Date, y=a.TotalMinutesAsleep / 60, name="Sleep (h)", mode="lines+markers", connectgaps=True,
                        line=dict(color=TEAL, width=3), secondary_y=True)
        fig.add_hline(y=sgoal, line_dash="dot", line_color=TEAL, annotation_text="sleep goal", annotation_position="top left", secondary_y=True)
    fig.add_hline(y=goal, line_dash="dash", line_color="white", annotation_text="step goal", secondary_y=False)
    fig.update_layout(title="Daily steps (orange = goal hit) vs sleep",
                      legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0, xanchor="left"))
    fig.update_yaxes(title_text="Steps", secondary_y=False)
    fig.update_yaxes(title_text="Sleep (h)", range=[0, 12], dtick=2, showgrid=False, secondary_y=True)
    ev = st.plotly_chart(style(fig, 380).update_layout(margin=dict(t=90, b=10, l=10, r=10)), width="stretch",
                         on_select="rerun", selection_mode="points", key=f"day_{aid}")
    pts = [p for p in (ev.selection.points if ev and ev.selection else []) if p.get("x") is not None]
    if pts:
        d0 = pd.to_datetime(pts[0]["x"]).normalize()
        r_ = a[a.Date.dt.normalize() == d0]
        if len(r_):
            r_ = r_.iloc[0]
            dc = st.columns(5)
            dc[0].metric(d0.strftime("%a %d %b"), f"{r_.TotalSteps:,.0f} steps", f"{r_.TotalSteps - goal:+,.0f} vs goal")
            dc[1].metric("Very active", f"{r_.VeryActiveMinutes:.0f} min")
            dc[2].metric("Calories", f"{r_.Calories:,.0f}")
            dc[3].metric("Sleep", f"{r_.TotalMinutesAsleep / 60:.1f} h" if pd.notna(r_.TotalMinutesAsleep) else "not logged")
            dc[4].metric("Avg heart rate", f"{r_.HR_mean:.0f} bpm" if pd.notna(r_.HR_mean) else "n/a")
    else:
        st.caption("👆 Click any bar to open that day's breakdown.")

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
    lb["Athlete"] = lb.Id.map(lab)
    f = px.bar(lb.iloc[::-1], x=opts[m], y="Athlete", orientation="h", text_auto=".0f", title=f"Top {n} by {m.lower()}", color_discrete_sequence=[ORANGE])
    evl = st.plotly_chart(style(f, 90 + 34 * n), width="stretch", on_select="rerun", selection_mode="points", key="lb_chart")
    lp = [p for p in (evl.selection.points if evl and evl.selection else []) if p.get("y") in lab_inv]
    if lp:
        sid = lab_inv[lp[0]["y"]]
        st.button(f"🏃 Open {lp[0]['y']} in Athlete Passport", on_click=set_aid, args=(sid,), type="primary")
        st.caption("Loaded. Now switch to the Athlete Passport tab.")
    else:
        st.caption("👆 Click a bar to pick an athlete, then open their passport.")
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

# ============ SQL Lab  (playground)
CHALLENGES = [
    ("⭐ Warm-up: top steppers", "Return each athlete's `Id` and their average `TotalSteps` (call it `avg_steps`), highest first. Top 5 only.",
     "GROUP BY Id, AVG(...), ORDER BY ... DESC, LIMIT 5",
     "SELECT Id, AVG(TotalSteps) AS avg_steps FROM daily GROUP BY Id ORDER BY avg_steps DESC LIMIT 5"),
    ("⭐ Sleep debt", "How many logged athlete-days had **less than 7 hours** (420 min) of sleep? One number, call it `nights`.",
     "COUNT(*) with WHERE TotalMinutesAsleep < 420",
     "SELECT COUNT(*) AS nights FROM daily WHERE TotalMinutesAsleep < 420"),
    ("⭐⭐ Heavy movers", "Which athletes average **more than 20 very-active minutes** a day? Return `Id` and `v` (the average), highest first.",
     "GROUP BY Id, then HAVING (not WHERE) to filter on an aggregate",
     "SELECT Id, AVG(VeryActiveMinutes) AS v FROM daily GROUP BY Id HAVING v > 20 ORDER BY v DESC"),
    ("⭐⭐ Peak hour", "Using `hourly`, find the hour of day (0-23) with the highest average `StepTotal`. Return `hr` and `avg_steps`, one row.",
     "CAST(strftime('%H', Hour) AS INTEGER) gives the hour",
     "SELECT CAST(strftime('%H',Hour) AS INTEGER) AS hr, AVG(StepTotal) AS avg_steps FROM hourly GROUP BY hr ORDER BY avg_steps DESC LIMIT 1"),
    ("⭐⭐⭐ Calorie podium", "Rank athletes by average `Calories` with a window function. Return `Id`, `avg_cal`, `rnk` for ranks 1-3.",
     "RANK() OVER (ORDER BY AVG(Calories) DESC) inside a GROUP BY query",
     "SELECT Id, AVG(Calories) AS avg_cal, RANK() OVER (ORDER BY AVG(Calories) DESC) AS rnk FROM daily GROUP BY Id ORDER BY rnk LIMIT 3"),
]
SNIPPETS = {
    "Peek at the data": "SELECT *\nFROM daily\nLIMIT 10;",
    "Filter + sort": "SELECT Id, TotalSteps, Calories\nFROM daily\nWHERE TotalSteps > 15000\nORDER BY TotalSteps DESC\nLIMIT 20;",
    "GROUP BY + HAVING": "SELECT Id, COUNT(*) AS days, ROUND(AVG(TotalSteps)) AS avg_steps\nFROM daily\nGROUP BY Id\nHAVING days >= 20\nORDER BY avg_steps DESC;",
    "CASE bucketing": "SELECT CASE WHEN TotalSteps >= 10000 THEN '1 Active'\n            WHEN TotalSteps >= 5000  THEN '2 Moderate'\n            ELSE '3 Low' END AS band,\n       COUNT(*) AS days,\n       ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep\nFROM daily\nWHERE TotalMinutesAsleep IS NOT NULL\nGROUP BY band\nORDER BY band;",
    "CTE": "WITH per_user AS (\n  SELECT Id, AVG(TotalSteps) AS steps, AVG(TotalMinutesAsleep) AS sleep\n  FROM daily GROUP BY Id\n)\nSELECT * FROM per_user\nWHERE sleep IS NOT NULL\nORDER BY steps DESC;",
    "Window: RANK": "SELECT Id, ROUND(AVG(Calories)) AS avg_cal,\n       RANK() OVER (ORDER BY AVG(Calories) DESC) AS rnk\nFROM daily\nGROUP BY Id\nORDER BY rnk;",
}


def set_sql(text):
    st.session_state["sql_text"] = text


def same_result(a, b):
    if a.shape != b.shape:
        return False
    for i in range(a.shape[1]):
        x, y = a.iloc[:, i], b.iloc[:, i]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.astype(float), y.astype(float), atol=0.6, equal_nan=True):
                return False
        elif not (x.astype(str).values == y.astype(str).values).all():
            return False
    return True


with tabs[4]:
    ss = st.session_state
    ss.setdefault("hist", [])
    ss.setdefault("solved", set())
    ss.setdefault("sql_text", SNIPPETS["Peek at the data"])
    ss.setdefault("last", None)
    st.markdown("### 🧪 SQL Playground")
    left, right = st.columns([1, 3])

    with right:
        mode = st.radio("Mode", ["🧪 Free play", "🎯 Challenges", "📚 Query library"], horizontal=True, label_visibility="collapsed")
        ch = None
        if mode.startswith("🎯"):
            st.progress(len(ss.solved) / len(CHALLENGES), text=f"{len(ss.solved)} / {len(CHALLENGES)} challenges solved")
            ch = st.selectbox("Challenge", range(len(CHALLENGES)), format_func=lambda i: f"{'✅' if i in ss.solved else '⬜'} {CHALLENGES[i][0]}")
            st.info(CHALLENGES[ch][1])
            with st.expander("💡 Hint"):
                st.code(CHALLENGES[ch][2], language="text")
            with st.expander("🔓 Reveal solution"):
                st.code(CHALLENGES[ch][3], language="sql")
        elif mode.startswith("📚"):
            key = st.selectbox("Analysis", list(QUERIES), format_func=lambda k_: QUERIES[k_][0])
            st.info(QUERIES[key][1])
            st.button("⬇ Load into editor", on_click=set_sql, args=(QUERIES[key][2].strip(),))

        st.text_area("SQL editor (SELECT / WITH only)", key="sql_text", height=210)
        b = st.columns([1, 1, 4])
        run = b[0].button("▶ Run", type="primary")
        check = b[1].button("✔ Check") if ch is not None else False

        if run or check:
            t0 = time.perf_counter()
            try:
                res = run_query(ss.sql_text)
                ss.last = (res, (time.perf_counter() - t0) * 1000)
                ss.hist = ([ss.sql_text] + [h for h in ss.hist if h != ss.sql_text])[:6]
                if check:
                    sol = run_query(CHALLENGES[ch][3])
                    if same_result(res, sol):
                        if ch not in ss.solved:
                            ss.solved.add(ch)
                            st.balloons()
                        st.success("✅ Correct! Your result matches the expected answer.")
                    else:
                        st.error(f"Not quite: you returned {res.shape[0]} rows × {res.shape[1]} columns; the answer has {sol.shape[0]} × {sol.shape[1]}. Check the hint.")
            except QueryError as e:
                ss.last = None
                st.error(str(e))

        if ss.last is not None:
            res, ms = ss.last
            m = st.columns(3)
            m[0].metric("Rows", f"{len(res):,}")
            m[1].metric("Columns", res.shape[1])
            m[2].metric("Runtime", f"{ms:.0f} ms")
            if res.empty:
                st.warning("Query ran but returned no rows.")
            else:
                view = st.radio("View as", ["Table", "Bar", "Line", "Scatter"], horizontal=True)
                if view == "Table":
                    st.dataframe(res, hide_index=True, width="stretch")
                else:
                    nums = list(res.select_dtypes("number").columns)
                    if not nums:
                        st.info("Need at least one numeric column to chart.")
                    else:
                        cx, cy = st.columns(2)
                        xc = cx.selectbox("X", list(res.columns))
                        yc = cy.selectbox("Y", nums, index=min(1, len(nums) - 1) if nums[0] == xc else 0)
                        plot = res.assign(**{xc: res[xc].astype(str)}) if view == "Bar" else res
                        fig = {"Bar": px.bar, "Line": px.line, "Scatter": px.scatter}[view](plot, x=xc, y=yc, color_discrete_sequence=[ORANGE])
                        show(fig, 360)
                st.download_button("⬇ Download CSV", res.to_csv(index=False), "query_result.csv")

    with left:
        st.markdown("#### 🗂️ Schema")
        for t in ("daily", "hourly"):
            try:
                cols = list(run_query(f"SELECT * FROM {t} LIMIT 1").columns)
                n_ = int(run_query(f"SELECT COUNT(*) AS n FROM {t}").n[0])
                with st.expander(f"📋 {t} · {n_:,} rows", expanded=(t == "daily")):
                    st.markdown(" ".join(f"`{c}`" for c in cols))
            except Exception as e:  # noqa: BLE001
                st.caption(f"{t}: {e}")
        st.markdown("#### 🧰 Snippets")
        sn = st.selectbox("Pattern", list(SNIPPETS), label_visibility="collapsed")
        st.button("Insert pattern", on_click=set_sql, args=(SNIPPETS[sn],))
        if ss.hist:
            st.markdown("#### 🕘 History")
            for i, h in enumerate(ss.hist):
                st.button(" ".join(h.split())[:34] + "…", key=f"hist{i}", on_click=set_sql, args=(h,))

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
        if best.roc_auc < 0.6:
            st.warning(f"Honest result: best model = {best.model} (AUC {best.roc_auc:.2f}). Daily activity alone barely predicts sleep — a useful negative finding.")
        else:
            st.success(f"Best model = {best.model} (AUC {best.roc_auc:.2f}). Activity carries some signal about sleep, but it is far from a reliable predictor.")
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
    hi, lo = sleep_gap(df)
    raw = get_data(False)
    ndays = (df.Date.max() - df.Date.min()).days + 1
    nonwear = (1 - len(get_data(True)) / len(raw)) * 100
    n_all = raw.Id.nunique()
    n_sleep, n_hr = raw.dropna(subset=["TotalMinutesAsleep"]).Id.nunique(), raw.dropna(subset=["HR_mean"]).Id.nunique()
    wcol = next((c_ for c_ in raw.columns if "weight" in c_.lower()), None)
    n_w = raw.dropna(subset=[wcol]).Id.nunique() if wcol else None
    try:
        wk_p = stats.mannwhitneyu(df[df.IsWeekend == 1].TotalSteps, df[df.IsWeekend == 0].TotalSteps).pvalue
        wk_txt = "**No weekend effect** on steps." if wk_p >= .05 else "**Weekends differ** in steps (p < 0.05)."
    except Exception:  # noqa: BLE001
        wk_txt = ""
    gap = lo - hi
    gap_txt = (f"**10k+ step days → ~{gap:.0f} min less sleep** than <5k days ({hi:.0f} vs {lo:.0f} min)" if gap > 0
               else f"**10k+ step days → ~{-gap:.0f} min more sleep** than <5k days ({hi:.0f} vs {lo:.0f} min)")
    st.markdown(f"""
### Key findings
- {gap_txt} — small effect, pooled user-days.
- **{(s.TotalMinutesAsleep < 420).mean() * 100:.0f}% of logged nights are under 7 h.**
- More very-active minutes go with a **lower minimum heart rate** (fitness signal).
- {wk_txt} Activity alone is a weak predictor of sleep (see ML Lab).

### Recommendations (Bellabeat-style, for the marketing team)
1. **Wear detection:** flag non-wear days so metrics aren't distorted ({nonwear:.1f}% of days here).
2. **Smart nudges:** notify inactive users; suggest earlier-day workouts to protect sleep.
3. **Sleep-debt alerts** after 3+ nights under 7 h (see SQL Lab watchlist).
4. **Collect more data:** only {n_sleep} / {n_hr}{f' / {n_w}' if n_w is not None else ''} of {n_all} athletes have sleep / heart-rate{' / weight' if n_w is not None else ''} data; add self-reported stress and mood.

*Limitations: {n_all} users, {ndays} days, pooled user-days, observational data (no causality).*
""")