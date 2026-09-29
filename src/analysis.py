"""Statistics + ML. Pure functions, no Streamlit imports (reusable and testable)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .db import run_query

SLEEP_GOAL_MIN = 420  # 7 hours
FEATURES = ["TotalSteps", "TotalDistance", "VeryActiveMinutes", "FairlyActiveMinutes",
            "LightlyActiveMinutes", "Calories", "IsWeekend"]
# SedentaryMinutes is deliberately excluded: it counts minutes asleep, so it would leak the target.
CORR_COLS = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes",
             "Calories", "METs_mean", "HR_mean", "HR_min", "TotalMinutesAsleep", "TotalTimeInBed", "SleepEfficiency"]


def load_daily(exclude_nonwear: bool = True) -> pd.DataFrame:
    df = run_query("SELECT * FROM daily", limit=10**6)
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsWeekend"] = (df.Date.dt.dayofweek >= 5).astype(int)
    df["SleepEfficiency"] = 100 * df.TotalMinutesAsleep / df.TotalTimeInBed
    df["GoodSleep"] = np.where(df.TotalMinutesAsleep.isna(), np.nan, (df.TotalMinutesAsleep >= SLEEP_GOAL_MIN).astype(float))
    df["StepsBand"] = pd.cut(df.TotalSteps, [-1, 5000, 10000, 10**6], labels=["<5k", "5-10k", "10k+"])
    return df[df.IsNonWearDay == 0].copy() if exclude_nonwear else df


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["TotalMinutesAsleep", "HR_mean", "WeightKg", "METs_mean"]
    out = pd.DataFrame({"missing_pct": (df[cols].isna().mean() * 100).round(1)})
    out["handling"] = ["Not imputed for stats (rows without sleep dropped); median-imputed inside ML pipeline only if a feature is missing",
                       "Not imputed; HR analyses use only days with HR (14 users)",
                       "Excluded from modelling (only 8 users); shown descriptively",
                       "Only 6 days missing; left as NaN"]
    return out.reset_index(names="column")


def _mwu(a, b, label, what):
    a, b = a.dropna(), b.dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1  # rank-biserial effect size
    return dict(test="Mann-Whitney U", comparison=label, n=f"{len(a)} vs {len(b)}", statistic=round(u, 1), p_value=p,
                effect_size=round(r, 3), median_diff=f"{a.median():.0f} vs {b.median():.0f} {what}")


def stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Non-parametric tests (data are skewed). Caveat: user-days are pooled, so p-values are optimistic."""
    rows = []
    rows.append(_mwu(df[df.IsWeekend == 1].TotalSteps, df[df.IsWeekend == 0].TotalSteps, "Steps: weekend vs weekday", "steps"))
    s = df.dropna(subset=["TotalMinutesAsleep"])
    rows.append(_mwu(s[s.TotalSteps >= 10000].TotalMinutesAsleep, s[s.TotalSteps < 10000].TotalMinutesAsleep,
                     "Sleep: 10k+ step days vs others", "min"))
    groups = [g.TotalMinutesAsleep.values for _, g in s.groupby("StepsBand", observed=True) if len(g) > 2]
    h, p = stats.kruskal(*groups)
    rows.append(dict(test="Kruskal-Wallis", comparison="Sleep across step bands (<5k / 5-10k / 10k+)", n=str(len(s)),
                     statistic=round(h, 2), p_value=p, effect_size=round(h * (len(s) + 1) / (len(s) ** 2 - 1), 3), median_diff="epsilon-squared"))
    for x, y, lab in [("TotalSteps", "TotalMinutesAsleep", "Steps vs sleep minutes"),
                      ("VeryActiveMinutes", "SleepEfficiency", "Very-active min vs sleep efficiency"),
                      ("VeryActiveMinutes", "HR_min", "Very-active min vs min heart rate"),
                      ("TotalSteps", "HR_mean", "Steps vs mean heart rate")]:
        d = df[[x, y]].dropna()
        rho, p = stats.spearmanr(d[x], d[y])
        rows.append(dict(test="Spearman", comparison=lab, n=str(len(d)), statistic=round(rho, 3), p_value=p,
                         effect_size=round(rho, 3), median_diff="rho"))
    out = pd.DataFrame(rows)
    out["significant_at_5pct"] = np.where(out.p_value < 0.05, "Yes", "No")
    out["p_value"] = out.p_value.map(lambda v: f"{v:.4f}" if v >= 0.0001 else "<0.0001")
    return out


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in CORR_COLS if c in df]].corr(method="spearman").round(2)


def _models():
    def pipe(m, scale=True):
        steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else []) + [("m", m)]
        return Pipeline(steps)
    return {"Baseline (majority)": pipe(DummyClassifier(strategy="most_frequent")),
            "Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
            "KNN": pipe(KNeighborsClassifier(n_neighbors=15)),
            "SVM (RBF)": pipe(SVC(probability=True, random_state=42)),
            "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42), False),
            "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=42), False)}


def sleep_dataset(df: pd.DataFrame):
    d = df.dropna(subset=["GoodSleep"]).copy()
    if d.GoodSleep.nunique() < 2 or len(d) < 50:
        raise ValueError("Not enough labelled sleep nights to model.")
    return d[FEATURES], d.GoodSleep.astype(int), d.Id


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Predict 'slept >= 7h' from that day's activity. Grouped CV by user so no user appears in both train and test."""
    X, y, g = sleep_dataset(df)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, m in _models().items():
        r = cross_validate(m, X, y, groups=g, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        rows.append(dict(model=name, accuracy=r["test_accuracy"].mean(), f1=r["test_f1"].mean(),
                         roc_auc=r["test_roc_auc"].mean(), roc_auc_std=r["test_roc_auc"].std()))
    return pd.DataFrame(rows).round(3).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def fit_best(df: pd.DataFrame, name: str):
    X, y, _ = sleep_dataset(df)
    return _models()[name].fit(X, y)


def personas(df: pd.DataFrame):
    """KMeans on per-user behaviour; k chosen by silhouette score (2-5)."""
    u = df.groupby("Id").agg(avg_steps=("TotalSteps", "mean"), avg_very_active_min=("VeryActiveMinutes", "mean"),
                             avg_calories=("Calories", "mean"), avg_sleep_min=("TotalMinutesAsleep", "mean")).reset_index()
    u["avg_sleep_min"] = u.avg_sleep_min.fillna(u.avg_sleep_min.median())
    Z = StandardScaler().fit_transform(u.drop(columns="Id"))
    best = max(range(2, 6), key=lambda k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z)))
    lab = KMeans(best, n_init=10, random_state=42).fit_predict(Z)
    u["cluster"] = lab
    order = u.groupby("cluster").avg_steps.mean().sort_values().index
    names = ["Low movers", "Steady movers", "Power movers", "Elite movers", "Ultra movers"]
    u["persona"] = u.cluster.map({c: names[i] for i, c in enumerate(order)})
    return u.drop(columns="cluster"), best
