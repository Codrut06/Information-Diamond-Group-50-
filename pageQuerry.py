"""
pageQuerry.py — SQL Lab (Speed Dating)
Requirements: streamlit, pandas, numpy, pandasql, matplotlib
"""

from __future__ import annotations

import os
import re
import time
import sqlite3
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter
from pandasql import sqldf

# ============================== PAGE & THEME ==============================

st.set_page_config(page_title="SQL Lab — Speed Dating", layout="wide")

# ------------------------------ GLOBAL CSS -------------------------------

st.markdown(
    """
<style>
:root{
  --bg: #0e1117;
  --panel: rgba(255,255,255,.05);
  --ring: rgba(255,255,255,.12);
  --text: #e6e6e6;
  --muted: rgba(230,230,230,.65);
  --accent: #21c4b8;
  --accent2: #ffb020;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
.block-container { padding-top: 3.25rem; padding-bottom: 1rem; }

/* Headings */
h1, h2, h3, h4 { letter-spacing:.2px; }

/* Cards */
.app-card{
  background: var(--panel);
  border:1px solid var(--ring);
  border-radius:16px;
  padding:16px;
  box-shadow: 0 6px 24px rgba(0,0,0,.35);
}

/* KPI */
.kpi{
  display:flex;flex-direction:column;gap:.2rem;
  background: var(--panel);
  border:1px solid var(--ring); border-radius:14px; padding:12px 14px;
}
.kpi .label{font-size:.78rem; color:var(--muted)}
.kpi .val{font-weight:700; font-size:1.15rem}

/* Badges */
.badge{
  display:inline-flex;align-items:center;gap:.35rem;
  font-size:.78rem;background:var(--panel);padding:.25rem .55rem;border-radius:999px;border:1px solid var(--ring);
}
.hr-soft{height:1px;background:var(--ring);border:0;margin:.5rem 0 1rem}

/* Buttons & Inputs */
.stButton>button, .stDownloadButton>button{
  border-radius:10px; padding:.55rem .9rem; border:1px solid var(--ring);
}
.stButton>button:hover{ border-color:var(--accent); }
div[data-baseweb="select"] > div { border-radius: 10px !important; }

/* Editor */
.stTextArea textarea{
  background: rgba(255,255,255,.04) !important; color:var(--text) !important;
  border-radius:12px !important; border:1px solid var(--ring) !important;
  font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.95rem; line-height:1.35;
}

/* Dataframe */
[data-testid="stDataFrame"] div[role="table"] { border-radius:12px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] button{
  background: var(--panel); border:1px solid var(--ring); margin-right:.4rem; border-radius:10px;
}
.stTabs [data-baseweb="tab"] { color: var(--text); }

kbd{
  background: rgba(255,255,255,.08); color:#fff; padding:2px 6px; border-radius:6px;
  font-size:.78rem; border:1px solid var(--ring);
}
.small-muted{ color: var(--muted); font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ================================ DATA ===================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load CSV from ./proj/speeddating.csv or ./speeddating.csv."""
    path = os.path.join("proj", "speeddating.csv")
    if not os.path.exists(path):
        path = "speeddating.csv"
    return pd.read_csv(path)

df = load_data()

# Normalize placeholder strings to NaN
for _col in df.select_dtypes(include=["object"]).columns:
    try:
        df[_col] = df[_col].astype(str).str.strip().replace({"": np.nan, "?": np.nan})
    except (ValueError, TypeError):
        pass

# Global KPI (benchmark)
OVERALL_MATCH_RATE: Optional[float] = None
if "match" in df.columns:
    try:
        OVERALL_MATCH_RATE = float(pd.to_numeric(df["match"], errors="coerce").mean())
    except (ValueError, TypeError):
        OVERALL_MATCH_RATE = None

# =============================== PRESETS =================================

PRESETS: Dict[str, str] = {
    "Average age by gender": """
        SELECT
          CASE WHEN gender IS NULL THEN 'Unknown' ELSE gender END AS gender,
          AVG(age) AS avg_age
        FROM df
        WHERE age IS NOT NULL
        GROUP BY gender;
    """,
    "Match rate by gender": """
        SELECT
          CASE WHEN gender IS NULL THEN 'Unknown' ELSE gender END AS gender,
          COUNT(*) AS n_dates,
          AVG(match*1.0) AS match_rate
        FROM df
        WHERE match IS NOT NULL
        GROUP BY gender
        ORDER BY match_rate DESC;
    """,
    "Match rate by age bucket (5y)": """
        SELECT
          CAST((age/5)*5 AS INT) AS age_bucket,
          COUNT(*) AS n_dates,
          AVG(match*1.0) AS match_rate
        FROM df
        WHERE age IS NOT NULL AND match IS NOT NULL
        GROUP BY age_bucket
        ORDER BY age_bucket;
    """,
    "Match rate by race": """
        SELECT
          CASE
            WHEN race IS NULL OR TRIM(race) = '' THEN 'Unknown'
            ELSE race
          END AS race,
          COUNT(*) AS n_dates,
          AVG(match*1.0) AS match_rate
        FROM df
        WHERE match IS NOT NULL
        GROUP BY race
        ORDER BY match_rate DESC, n_dates DESC;
    """,
    "Top 10 fields by match rate (min 50 dates)": """
        SELECT
          CASE
            WHEN field IS NULL OR TRIM(field) = '' THEN 'Unknown'
            ELSE field
          END AS field,
          COUNT(*) AS n_dates,
          AVG(match*1.0) AS match_rate
        FROM df
        WHERE match IS NOT NULL
        GROUP BY field
        HAVING COUNT(*) >= 50
        ORDER BY match_rate DESC, n_dates DESC
        LIMIT 10;
    """,
    "'Like' by match outcome (0/1)": """
        SELECT match, AVG(like) AS avg_like, COUNT(*) AS n
        FROM df
        WHERE like IS NOT NULL AND match IS NOT NULL
        GROUP BY match
        ORDER BY match;
    """,
    "Perceived reciprocity by outcome (guess_prob_liked)": """
        SELECT match, AVG(guess_prob_liked) AS avg_guess_prob, COUNT(*) AS n
        FROM df
        WHERE guess_prob_liked IS NOT NULL AND match IS NOT NULL
        GROUP BY match
        ORDER BY match;
    """,
    "Average traits by match outcome": """
        SELECT
          match,
          AVG(attractive)   AS avg_attr,
          AVG(sincere)      AS avg_sincere,
          AVG(intelligence) AS avg_intel,
          AVG(funny)        AS avg_funny,
          AVG(ambition)     AS avg_ambition,
          COUNT(*)          AS n
        FROM df
        WHERE match IS NOT NULL
        GROUP BY match
        ORDER BY match;
    """,
    "Overall match rate": """
        SELECT AVG(match*1.0) AS match_rate
        FROM df
        WHERE match IS NOT NULL;
    """,
    "Match rate by wave/session": """
        SELECT wave, COUNT(*) AS n, AVG(match*1.0) AS match_rate
        FROM df
        WHERE wave IS NOT NULL AND match IS NOT NULL
        GROUP BY wave
        ORDER BY wave;
    """,
    "Like → Match correlation (bucketed)": """
        SELECT ROUND(like,0) AS like_bucket, COUNT(*) AS n, AVG(match*1.0) AS match_rate
        FROM df
        WHERE like IS NOT NULL AND match IS NOT NULL
        GROUP BY like_bucket
        ORDER BY like_bucket;
    """,
    "Guess reciprocity accuracy (bucketed)": """
        SELECT ROUND(guess_prob_liked,0) AS guess_bucket, COUNT(*) AS n, AVG(match*1.0) AS match_rate
        FROM df
        WHERE guess_prob_liked IS NOT NULL AND match IS NOT NULL
        GROUP BY guess_bucket
        ORDER BY guess_bucket;
    """,
}

DESCRIPTIONS: Dict[str, str] = {
    "Average age by gender": "Average age by gender.",
    "Match rate by gender": "Match rate across genders.",
    "Match rate by age bucket (5y)": "Five-year age buckets vs. match rate.",
    "Match rate by race": "Match rate by race.",
    "Top 10 fields by match rate (min 50 dates)": "Fields with the highest match rate (n ≥ 50).",
    "'Like' by match outcome (0/1)": "Average 'like' by match outcome.",
    "Perceived reciprocity by outcome (guess_prob_liked)": "Perceived reciprocity by match outcome.",
    "Average traits by match outcome": "Trait averages by match outcome.",
    "Overall match rate": "Overall match rate.",
    "Match rate by wave/session": "Match rate by session (wave).",
    "Like → Match correlation (bucketed)": "Match rate by rounded 'like' score.",
    "Guess reciprocity accuracy (bucketed)": "Match rate by rounded perceived reciprocity.",
}

# ============================== UTILITIES ================================

def _now() -> float:
    return time.perf_counter()

def _fmt_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "—"

def _runtime_badge(start_t: float) -> int:
    """Render a small runtime badge and return duration (ms)."""
    duration = int((_now() - start_t) * 1000)
    st.markdown(f"<span class='badge'>Runtime: {duration} ms</span>", unsafe_allow_html=True)
    return duration

def _small_fig() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=130)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    return fig, ax

def _percent_axis(ax: Axes) -> None:
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x * 100:.0f}%"))

BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)

def validate_sql(sql: str) -> Tuple[bool, str]:
    """Allow only simple SELECT statements against table `df`."""
    statement = (sql or "").strip().rstrip(";")
    if not statement:
        return False, "Enter a SQL query or choose a preset."
    if not statement.lower().startswith("select"):
        return False, "Only SELECT queries are allowed."
    if BLOCKED.search(statement):
        return False, "Statement blocked. Only simple SELECT queries are allowed."
    if re.search(r"from\s+(?!df\b)[a-zA-Z_]\w*", statement, re.IGNORECASE):
        return False, "Only the in-memory table `df` is available."
    return True, statement

def _format_table(df_res: pd.DataFrame) -> pd.DataFrame:
    """Light post-processing for numeric counts and *_rate columns."""
    for col in df_res.columns:
        lc = col.lower()
        if lc in {"n", "n_dates", "count", "num"}:
            with pd.option_context("mode.use_inf_as_na", True):
                try:
                    df_res[col] = pd.to_numeric(df_res[col], errors="coerce").round().astype("Int64")
                except (ValueError, TypeError):
                    pass
        if lc.endswith("_rate"):
            try:
                df_res[col] = pd.to_numeric(df_res[col], errors="coerce")
            except (ValueError, TypeError):
                pass
    return df_res

def _analytical_insights(df_res: pd.DataFrame) -> List[str]:
    """Generate short insights based on the result set."""
    lines: List[str] = []
    if df_res is None or df_res.empty:
        return lines

    cols_lower = [c.lower() for c in df_res.columns]

    # Weighted average for match_rate + lift vs. global
    if "match_rate" in cols_lower:
        mr_col = df_res.columns[cols_lower.index("match_rate")]
        mr = pd.to_numeric(df_res[mr_col], errors="coerce")

        if mr.notna().any():
            weights = None
            for wname in ("n", "n_dates", "count", "num"):
                if wname in df_res.columns:
                    weights = pd.to_numeric(df_res[wname], errors="coerce").fillna(0.0)
                    break

            if weights is not None and (weights > 0).any():
                denom = weights.sum() if weights.sum() != 0 else 1
                wa = float((mr.fillna(0) * weights).sum() / denom)
            else:
                wa = float(mr.mean())

            lines.append(f"Weighted average match_rate: { _fmt_percent(wa) }.")

            if OVERALL_MATCH_RATE:
                lift = (wa - OVERALL_MATCH_RATE) / OVERALL_MATCH_RATE
                lines.append(
                    f"Lift vs. overall: {lift * 100:.0f}% "
                    f"(overall { _fmt_percent(OVERALL_MATCH_RATE) })."
                )

            num_cols = df_res.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = [c for c in df_res.columns if c not in num_cols]
            rank_key = cat_cols[0] if cat_cols else df_res.columns[0]

            try:
                ranked = df_res[[rank_key, mr_col]].dropna().sort_values(mr_col, ascending=False)
                if len(ranked):
                    head = ranked.head(3)
                    tail = ranked.tail(3).sort_values(mr_col, ascending=True)

                    lines.append("Top groups by match_rate:")
                    for _, r in head.iterrows():
                        lines.append(f"• {r[rank_key]} → { _fmt_percent(r[mr_col]) }")

                    if len(ranked) >= 6:
                        lines.append("Bottom groups by match_rate:")
                        for _, r in tail.iterrows():
                            lines.append(f"• {r[rank_key]} → { _fmt_percent(r[mr_col]) }")
            except (KeyError, ValueError, TypeError):
                pass

            for cname in ("n", "n_dates"):
                if cname in df_res.columns:
                    try:
                        if pd.to_numeric(df_res[cname], errors="coerce").min() < 30:
                            lines.append("Some groups have a small sample size (n < 30).")
                    except (ValueError, TypeError):
                        pass
                    break

    # Raw match rate if a binary `match` column is present
    if "match" in cols_lower:
        mcol = df_res.columns[cols_lower.index("match")]
        s = pd.to_numeric(df_res[mcol], errors="coerce")
        uniq = set(s.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            mean_m = float(s.mean())
            lines.append(f"Raw match rate in the returned rows: { _fmt_percent(mean_m) }.")

    # Strongest numeric correlation (if any)
    num_cols_all = df_res.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols_all) >= 2:
        try:
            corr = df_res[num_cols_all].corr(numeric_only=True)
            np.fill_diagonal(corr.values, np.nan)
            idx = np.nanargmax(np.abs(corr.values))
            i, j = divmod(idx, corr.shape[1])
            cval = corr.values[i, j]
            if cval is not None and not np.isnan(cval) and abs(cval) >= 0.4:
                lines.append(
                    f"Strongest correlation: {num_cols_all[i]} vs {num_cols_all[j]} = {cval:.2f}."
                )
        except (ValueError, TypeError):
            pass

    # Simple outlier signal
    for cname in num_cols_all[:6]:
        s = pd.to_numeric(df_res[cname], errors="coerce").dropna()
        if len(s) >= 20:
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                hi = q3 + 1.5 * iqr
                lo = q1 - 1.5 * iqr
                out_hi = int((s > hi).sum())
                out_lo = int((s < lo).sum())
                if (out_hi + out_lo) >= 1:
                    lines.append(
                        f"Possible outliers detected in {cname} "
                        f"(~{out_hi + out_lo} outside 1.5×IQR)."
                    )

    return lines

# =============================== STATE ===================================

if "sql_text" not in st.session_state:
    st.session_state.sql_text = ""

if "auto_run" not in st.session_state:
    st.session_state.auto_run = False
if "run_flag" not in st.session_state:
    st.session_state.run_flag = False

# Clear editor via control flag
if st.session_state.get("_clear_text", False):
    st.session_state.pop("sql_text", None)
    st.session_state["_clear_text"] = False
    st.session_state["preset_select"] = "— select —"
    st.session_state.auto_run = False
    st.session_state.run_flag = False
    st.rerun()

# =============================== HEADER ==================================

st.title("SQL Lab — Speed Dating")
st.caption("Run SELECT queries against the in-memory table df. The tool formats tables, suggests a chart, and extracts brief insights.")

# KPI row
k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1], gap="small")

with k1:
    st.markdown(
        f"<div class='kpi'><span class='label'>Rows</span>"
        f"<span class='val'>{len(df):,}</span></div>",
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"<div class='kpi'><span class='label'>Columns</span>"
        f"<span class='val'>{df.shape[1]}</span></div>",
        unsafe_allow_html=True,
    )

with k3:
    missing = round(100 * df.isna().mean().mean(), 1)
    st.markdown(
        f"<div class='kpi'><span class='label'>Average Missing</span>"
        f"<span class='val'>{missing}%</span></div>",
        unsafe_allow_html=True,
    )

with k4:
    if OVERALL_MATCH_RATE is not None:
        st.markdown(
            f"<div class='kpi'><span class='label'>Overall Match Rate</span>"
            f"<span class='val'>{_fmt_percent(OVERALL_MATCH_RATE)}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='kpi'><span class='label'>Overall Match Rate</span>"
            "<span class='val'>—</span></div>",
            unsafe_allow_html=True,
        )

with k5:
    st.markdown(
        "<div class='kpi'><span class='label'>Shortcut</span>"
        "<span class='val'><kbd>Ctrl</kbd>+<kbd>Enter</kbd> to run</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

# ================================ LAYOUT =================================

left, right = st.columns([4, 2.2], gap="large")

# --------- RIGHT: PRESETS PANEL -----------------------------------------

with right:
    st.markdown("### Presets")
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)

    chosen = st.selectbox(
        "Preset",
        ["— select —"] + list(PRESETS.keys()),
        index=0,
        key="preset_select",
    )

    disabled = chosen == "— select —"

    if disabled:
        st.caption("Select a preset to enable the actions.")
    else:
        st.caption(DESCRIPTIONS.get(chosen, "—"))

    cpa, cpb = st.columns(2)

    with cpa:
        if st.button("Insert into editor", use_container_width=True, disabled=disabled):
            st.session_state.sql_text = PRESETS[chosen].strip()

    with cpb:
        if st.button("Replace and run", use_container_width=True, disabled=disabled):
            st.session_state.sql_text = PRESETS[chosen].strip()
            st.session_state.auto_run = True

    st.markdown("</div>", unsafe_allow_html=True)

# --------- LEFT: EDITOR + OUTPUT ----------------------------------------

with left:
    st.markdown("### SQL Editor")
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)

    with st.form("sql_form", clear_on_submit=False):
        _ = st.text_area(
            "SQL",
            key="sql_text",
            height=480,
            placeholder="Type a SELECT over df (or choose a preset)…",
        )

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            if st.form_submit_button("Run", use_container_width=True):
                st.session_state.run_flag = True

        with c2:
            if st.form_submit_button("Clear", use_container_width=True):
                st.session_state["_clear_text"] = True
                st.rerun()

        with c3:
            if st.form_submit_button("Copy SQL", use_container_width=True):
                st.code(st.session_state.sql_text or "-- (empty)", language="sql")
                st.toast("Copy from the code block. Direct clipboard access may be blocked by the browser.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== RUNNER ===================================

@st.cache_data(ttl=60, show_spinner=False)
def run_sql_cached(query: str, df_source: pd.DataFrame) -> pd.DataFrame:
    """Execute SQL against the in-memory DataFrame using pandasql."""
    return sqldf(query, {"df": df_source})

def plot_bar_percent(
    df_res: pd.DataFrame,
    x: str,
    y: str,
    top_n: int = 12,
    title: Optional[str] = None,
) -> None:
    """Bar chart for percentage-like y values."""
    grouped = df_res.groupby(x, dropna=False, as_index=False)[y].mean()
    grouped = grouped.sort_values(y, ascending=False).head(top_n)
    fig, ax = _small_fig()
    ax.bar(grouped[x].astype(str), grouped[y], width=0.55)
    _percent_axis(ax)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(f"{y} (%)", fontsize=9)
    ax.set_title(title or f"{y} by {x}", fontsize=10, pad=6)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=False)

def plot_line_percent(
    df_res: pd.DataFrame, x: str, y: str, title: Optional[str] = None
) -> None:
    """Line chart for percentage-like y values across ordered x."""
    fig, ax = _small_fig()
    ax.plot(df_res[x], df_res[y], marker="o", linewidth=1.25)
    _percent_axis(ax)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(f"{y} (%)", fontsize=9)
    ax.set_title(title or f"{y} by {x}", fontsize=10, pad=6)
    st.pyplot(fig, use_container_width=False)

def plot_bar_value(
    df_res: pd.DataFrame,
    x: str,
    y: str,
    top_n: int = 12,
    title: Optional[str] = None,
) -> None:
    """Bar chart for numeric y values."""
    grouped = df_res.groupby(x, dropna=False, as_index=False)[y].mean()
    grouped = grouped.sort_values(y, ascending=False).head(top_n)
    fig, ax = _small_fig()
    ax.bar(grouped[x].astype(str), grouped[y], width=0.55)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(y, fontsize=9)
    ax.set_title(title or f"{y} by {x}", fontsize=10, pad=6)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=False)

def plot_one_good_chart(res: pd.DataFrame) -> None:
    """Auto-pick a reasonable visualization based on the result data."""
    if res is None or res.empty:
        st.info("No data to chart.")
        return

    dfp = res.copy()
    num_cols = dfp.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in dfp.columns if c not in num_cols]
    cols_lower = [c.lower() for c in dfp.columns]

    if "match_rate" in cols_lower and len(cat_cols) >= 1:
        y_col = dfp.columns[cols_lower.index("match_rate")]
        x_col = cat_cols[0]
        plot_bar_percent(dfp, x_col, y_col)
        return

    for xname in ["age_bucket", "age_group", "age5", "like_bucket", "guess_bucket", "wave"]:
        if xname in cols_lower and "match_rate" in cols_lower:
            x_col = dfp.columns[cols_lower.index(xname)]
            y_col = dfp.columns[cols_lower.index("match_rate")]
            try:
                ordered = dfp.sort_values(x_col)
            except (TypeError, KeyError):
                ordered = dfp
            plot_line_percent(ordered, x_col, y_col)
            return

    if "match" in cols_lower:
        avg_cols = [c for c in dfp.columns if c.lower().startswith("avg")]
        if avg_cols:
            x_col = dfp.columns[cols_lower.index("match")]
            y_col = avg_cols[0]
            plot_bar_value(dfp, x_col, y_col, title=f"{y_col} by match (0/1)")
            return

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        x_col = cat_cols[0]
        pref = None
        for k in num_cols:
            if any(t in k.lower() for t in ["rate", "avg", "mean"]):
                pref = k
                break
        y_col = pref or num_cols[0]
        if "rate" in y_col.lower():
            plot_bar_percent(dfp, x_col, y_col)
        else:
            plot_bar_value(dfp, x_col, y_col)
        return

    if len(num_cols) >= 2:
        x_col, y_col = num_cols[:2]
        fig, ax = _small_fig()
        ax.scatter(dfp[x_col], dfp[y_col], s=16, alpha=0.7, edgecolor="none")
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        st.pyplot(fig, use_container_width=False)
        return

    if len(num_cols) == 1:
        col = num_cols[0]
        fig, ax = _small_fig()
        ax.hist(dfp[col].dropna(), bins=12, edgecolor="black", linewidth=0.35)
        ax.set_title(f"Distribution of {col}", fontsize=10, pad=6)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig, use_container_width=False)
        return

    st.info("Add a rate (e.g., match_rate) or an average to enable a chart suggestion.")

def render_output_tabs(
    query: str,
    res: pd.DataFrame,
    runtime_ms: int,
    preset_label: Optional[str],
) -> None:
    """Render table, chart, insights, and SQL tabs."""
    tabs = st.tabs(["Table", "Chart", "Insights", "SQL"])

    # Table
    with tabs[0]:
        if res is None or res.empty:
            st.info("No rows returned.")
        else:
            res_fmt = _format_table(res.copy())
            df_show = res_fmt.copy()

            percent_cols = {c for c in df_show.columns if c.lower().endswith("_rate")}
            column_config: Dict[str, st.column_config.Column] = {}

            for col in df_show.columns:
                if col in percent_cols:
                    try:
                        df_show[col] = pd.to_numeric(df_show[col], errors="coerce") * 100.0
                    except (ValueError, TypeError):
                        pass
                    column_config[col] = st.column_config.NumberColumn(format="%.2f%%")
                else:
                    if pd.api.types.is_numeric_dtype(df_show[col]):
                        column_config[col] = st.column_config.NumberColumn(format="%.2f")
                    else:
                        column_config[col] = st.column_config.TextColumn()

            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=2, key="pg_size")
            total = len(df_show)
            pages = max(1, (total + page_size - 1) // page_size)
            page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
            start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

            st.dataframe(
                df_show.iloc[start:end],
                hide_index=True,
                use_container_width=True,
                column_config=column_config,
            )

            cta, ctb, ctc = st.columns([1, 1, 2])
            with cta:
                st.markdown(f"<span class='badge'>Rows: {len(df_show):,}</span>", unsafe_allow_html=True)
            with ctb:
                st.markdown(f"<span class='badge'>Runtime: {int(runtime_ms)} ms</span>", unsafe_allow_html=True)
            with ctc:
                csv = df_show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="result.csv",
                    mime="text/csv",
                )

    # Chart
    with tabs[1]:
        plot_one_good_chart(res)

    # Insights
    with tabs[2]:
        if preset_label and preset_label in DESCRIPTIONS:
            st.markdown(f"Preset context — {DESCRIPTIONS[preset_label]}")
        insights = _analytical_insights(res)
        if insights:
            st.markdown("---")
            for line in insights:
                st.markdown(f"- {line}")
        if not (preset_label or insights):
            st.info("Run a preset or include a rate/average to view insights.")

    # SQL
    with tabs[3]:
        st.code(query, language="sql")

# ============================== EXECUTION ================================

want_run = bool(st.session_state.get("run_flag")) or bool(st.session_state.get("auto_run"))
if want_run:
    st.session_state.run_flag = False
    st.session_state.auto_run = False

    ok, clean_sql = validate_sql(st.session_state.sql_text)
    if not ok:
        st.error(clean_sql)
    else:
        t0 = _now()
        try:
            res_df = run_sql_cached(clean_sql, df)
        except (sqlite3.Error, ValueError, TypeError) as exc:
            st.error(f"SQL error: {exc}")
            res_df = None
        ms = _runtime_badge(t0)

        preset_name: Optional[str] = None
        for key, val in PRESETS.items():
            if val.strip().rstrip(";") == clean_sql.strip().rstrip(";"):
                preset_name = key
                break

        render_output_tabs(clean_sql, res_df, ms, preset_name)


