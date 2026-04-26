"""
TemporalEdge — Streamlit Dashboard  (Phase 5)
Three sections: Execution Plan / Simulator / Evidence
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TemporalEdge",
    page_icon="⏱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0D0F14;
  --bg-card:   #141720;
  --bg-hover:  #1A1E2A;
  --border:    #252A38;
  --border-hi: #323848;
  --text-1:    #E8EAF0;
  --text-2:    #9AA0B4;
  --text-3:    #5C6378;
  --green:     #2DB37A;
  --green-bg:  rgba(45,179,122,0.08);
  --green-brd: rgba(45,179,122,0.25);
  --amber:     #E8A020;
  --amber-bg:  rgba(232,160,32,0.08);
  --amber-brd: rgba(232,160,32,0.25);
  --grey:      #6B7280;
  --grey-bg:   rgba(107,114,128,0.06);
  --grey-brd:  rgba(107,114,128,0.2);
  --blue:      #4D9EF5;
  --red:       #E05C5C;
  --mono:      'IBM Plex Mono', monospace;
  --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
}

/* ── Sidebar: open state ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0D0F14 !important;
    border-right: 1px solid #1E2330 !important;
    min-width: 260px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: #141720 !important;
    border: 1px solid #252A38 !important;
    color: #E8EAF0 !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    padding: 0.3rem 0.5rem !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] .stNumberInput input {
    background: #141720 !important;
    border: 1px solid #252A38 !important;
    color: #E8EAF0 !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
section[data-testid="stSidebar"] button {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1E3A2B !important;
    border: 1px solid #2DB37A !important;
    color: #2DB37A !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #2DB37A !important;
    color: #0D0F14 !important;
}

/* ── Main content: center when sidebar is collapsed ─────────────────────── */
/* When sidebar is closed, Streamlit adds data-collapsed="true" to the     */
/* AppView container. We use this to re-center the block-container.        */
[data-testid="stAppViewContainer"] > section:last-child {
    transition: margin-left 0.3s ease;
}
/* Remove left padding offset Streamlit applies for sidebar — let the      */
/* block-container max-width + auto margins do the centering.              */
.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1100px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Sidebar toggle button — make it visible and styled ────────────────── */
button[data-testid="collapsedControl"] {
    background: #141720 !important;
    border: 1px solid #252A38 !important;
    border-radius: 0 6px 6px 0 !important;
    color: #9AA0B4 !important;
    top: 1.5rem !important;
}
button[data-testid="collapsedControl"]:hover {
    background: #1A1E2A !important;
    border-color: #2DB37A !important;
    color: #2DB37A !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }

/* Top nav */
.te-header {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 0 0 1.5rem 0; border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.te-logo {
  font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  letter-spacing: 0.08em; color: var(--text-1);
}
.te-logo span { color: var(--green); }
.te-meta { font-family: var(--mono); font-size: 0.72rem; color: var(--text-3); }

/* Ticker cards */
.ticker-card {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 1.1rem 1.25rem; margin-bottom: 0.75rem;
  background: var(--bg-card); transition: border-color 0.15s;
}
.ticker-card:hover { border-color: var(--border-hi); }
.ticker-card.green { border-left: 3px solid var(--green); background: linear-gradient(90deg, rgba(45,179,122,0.04) 0%, var(--bg-card) 40%); }
.ticker-card.amber { border-left: 3px solid var(--amber); background: linear-gradient(90deg, rgba(232,160,32,0.04) 0%, var(--bg-card) 40%); }
.ticker-card.grey  { border-left: 3px solid var(--grey);  background: var(--bg-card); }

.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem; }
.card-ticker { font-family: var(--mono); font-size: 1.05rem; font-weight: 600; }
.card-name   { font-size: 0.78rem; color: var(--text-2); margin-top: 2px; }
.card-badge  {
  font-family: var(--mono); font-size: 0.65rem; font-weight: 500;
  padding: 3px 8px; border-radius: 4px; letter-spacing: 0.06em; white-space: nowrap;
}
.badge-green { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-brd); }
.badge-amber { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-brd); }
.badge-grey  { background: var(--grey-bg);  color: var(--grey);  border: 1px solid var(--grey-brd); }

.card-rec { display: flex; gap: 2rem; margin-bottom: 0.6rem; align-items: baseline; }
.rec-day  { font-family: var(--mono); font-size: 1.4rem; font-weight: 600; }
.rec-label{ font-size: 0.72rem; color: var(--text-3); margin-top: 3px; }
.rec-conf { font-family: var(--mono); font-size: 0.78rem; color: var(--text-2); }

/* Confidence bar */
.conf-bar-wrap { background: var(--border); border-radius: 2px; height: 4px; margin: 0.4rem 0; }
.conf-bar { height: 4px; border-radius: 2px; }
.conf-green { background: var(--green); }
.conf-amber { background: var(--amber); }
.conf-grey  { background: var(--grey); }

.card-advisory {
  font-size: 0.80rem; color: var(--text-2); line-height: 1.55;
  border-top: 1px solid var(--border); padding-top: 0.6rem; margin-top: 0.4rem;
}
.key-factor { display: inline-block; font-family: var(--mono); font-size: 0.68rem;
              color: var(--text-3); background: var(--border); padding: 2px 7px;
              border-radius: 3px; margin-top: 0.4rem; }

/* Section headers */
.section-label {
  font-family: var(--mono); font-size: 0.68rem; font-weight: 500;
  letter-spacing: 0.12em; color: var(--text-3); text-transform: uppercase;
  margin: 1.5rem 0 0.75rem 0;
}

/* Tier legend */
.tier-legend { display: flex; gap: 1.5rem; margin-bottom: 1.5rem; }
.legend-item { display: flex; align-items: center; gap: 0.4rem;
               font-size: 0.75rem; color: var(--text-2); }
.legend-dot  { width: 8px; height: 8px; border-radius: 50%; }

/* Stat boxes */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.stat-box {
  flex: 1; background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.9rem 1.1rem;
}
.stat-val  { font-family: var(--mono); font-size: 1.35rem; font-weight: 600; }
.stat-lbl  { font-size: 0.72rem; color: var(--text-3); margin-top: 3px; }
.stat-green { color: var(--green); }
.stat-amber { color: var(--amber); }
.stat-white { color: var(--text-1); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important; font-size: 0.78rem !important;
  letter-spacing: 0.06em !important; color: var(--text-3) !important;
  background: transparent !important; border: none !important;
  padding: 0.6rem 1.25rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text-1) !important; border-bottom: 2px solid var(--green) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  color: var(--text-1) !important; font-family: var(--mono) !important;
  font-size: 0.85rem !important; border-radius: 6px !important;
}
.stSlider > div > div { background: var(--border) !important; }

/* Buttons */
.stButton > button {
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--text-2) !important; font-family: var(--mono) !important;
  font-size: 0.78rem !important; letter-spacing: 0.06em !important;
  border-radius: 6px !important; transition: all 0.15s !important;
}
.stButton > button:hover {
  border-color: var(--green) !important; color: var(--green) !important;
}

/* Plotly charts */
.js-plotly-plot { border-radius: 8px; }

/* Table */
.te-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.78rem; }
.te-table th { color: var(--text-3); font-weight: 500; padding: 0.4rem 0.75rem;
               border-bottom: 1px solid var(--border); text-align: left; letter-spacing: 0.06em; }
.te-table td { color: var(--text-1); padding: 0.45rem 0.75rem;
               border-bottom: 1px solid rgba(37,42,56,0.5); }
.te-table tr:hover td { background: var(--bg-hover); }
.td-green { color: var(--green) !important; }
.td-amber { color: var(--amber) !important; }
.td-red   { color: var(--red)   !important; }
.td-grey  { color: var(--grey)  !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# ── Default portfolio (pre-filled example) ───────────────────────────────────
_DEFAULT_PORTFOLIO = {
    "VOO":  {"name": "Vanguard S&P 500 ETF",           "monthly_usd": 15},
    "VTI":  {"name": "Vanguard Total Stock Market",     "monthly_usd": 10},
    "NVDA": {"name": "NVIDIA",                          "monthly_usd":  5},
    "AAPL": {"name": "Apple",                           "monthly_usd":  5},
    "SCHD": {"name": "Schwab Dividend ETF",             "monthly_usd":  4},
    "VXUS": {"name": "Vanguard Total International",    "monthly_usd":  5},
    "TSLA": {"name": "Tesla",                           "monthly_usd":  2},
    "BND":  {"name": "Vanguard Total Bond Market",      "monthly_usd":  3},
    "VYM":  {"name": "Vanguard High Dividend Yield",    "monthly_usd":  2},
    "VEA":  {"name": "Vanguard FTSE Developed Markets", "monthly_usd":  2},
    "VWO":  {"name": "Vanguard FTSE Emerging Markets",  "monthly_usd":  2},
}

def get_active_portfolio() -> dict:
    """
    Return the merged portfolio for the current session.
    Base = _DEFAULT_PORTFOLIO, overridden/extended by sidebar inputs.
    """
    if "user_portfolio" not in st.session_state:
        return _DEFAULT_PORTFOLIO
    return st.session_state["user_portfolio"]

# Module-level alias so existing code that uses PORTFOLIO still works
# (render_simulator uses PORTFOLIO.get(ticker_input, {}))
PORTFOLIO = _DEFAULT_PORTFOLIO

CACHE_FILE = Path(__file__).parent / "cache" / "execution_plan.json"
RESULTS_FILE = Path(__file__).parent.parent / "results" / "portfolio_simulation.json"

# Tickers whose model weights come from a proxy ticker (not their own training).
# For these, we display "proxy (SOURCE)" instead of a confidence % on the card —
# because the confidence belongs to the source model, not the ticker itself.
MODEL_PROXY = {"VTI": "VOO"}


def _format_confidence(ticker: str, conf_pct: int) -> str:
    """Return display string for confidence — 'proxy (VOO)' for proxy tickers."""
    if ticker in MODEL_PROXY:
        return f"proxy ({MODEL_PROXY[ticker]})"
    return f"{conf_pct}%"


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#141720",
    plot_bgcolor="#141720",
    font=dict(family="IBM Plex Mono, monospace", color="#9AA0B4", size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#252A38"),
)
# Axis defaults — applied per chart to avoid duplicate keyword errors
AXIS_DEFAULTS = dict(gridcolor="#252A38", linecolor="#252A38", showgrid=True)

TIER_COLORS = {"green": "#2DB37A", "amber": "#E8A020", "grey": "#6B7280"}
TIER_LABELS = {
    "green": "ACT ON MODEL",
    "amber": "USE LOOSELY",
    "grey":  "SKIP TIMING",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_execution_plan():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600)
def load_simulation_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None


# ── Header ────────────────────────────────────────────────────────────────────

def render_header(active_tab: str):
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    active = get_active_portfolio()
    n_tickers  = len(active)
    total_mo   = sum(v["monthly_usd"] for v in active.values())

    st.markdown(f"""
    <div class="te-header">
        <div>
            <div class="te-logo">TEMPORAL<span>EDGE</span></div>
            <div class="te-meta">portfolio-aware DCA timing system</div>
        </div>
        <div style="display:flex;align-items:center;gap:1.5rem;">
            <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                        text-align:right;line-height:1.6;">
                {now} UTC<br>
                <span style="color:#9AA0B4">{n_tickers} tickers · <span style="color:#2DB37A">${total_mo}/mo</span></span>
            </div>
            <button onclick="
                var btn = window.parent.document.querySelector('button[data-testid=\"collapsedControl\"]');
                var sidebar = window.parent.document.querySelector('section[data-testid=\"stSidebar\"]');
                if (btn) btn.click();
            " style="
                background:#141720;border:1px solid #252A38;border-radius:5px;
                color:#9AA0B4;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                padding:0.35rem 0.75rem;cursor:pointer;letter-spacing:0.04em;
                transition:all 0.15s;
            "
            onmouseover="this.style.borderColor='#2DB37A';this.style.color='#2DB37A'"
            onmouseout="this.style.borderColor='#252A38';this.style.color='#9AA0B4'"
            >⚙ Portfolio</button>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page 1: Execution Plan ────────────────────────────────────────────────────

def render_execution_plan():
    plan_data = load_execution_plan()

    if plan_data is None:
        st.markdown("""
        <div style="text-align:center; padding:4rem; color:#5C6378;">
            <div style="font-family:var(--mono); font-size:0.85rem; margin-bottom:1rem;">
                NO EXECUTION PLAN CACHED
            </div>
            <div style="font-size:0.78rem;">
                Run <code>python scripts/monthly_refresh.py</code> to generate this month's plan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⟳  Generate Plan Now"):
            with st.spinner("Running models and fetching context..."):
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from scripts.monthly_refresh import run_monthly_refresh
                    run_monthly_refresh()
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
        return

    # ── Macro bar ──────────────────────────────────────────────────────────
    macro = plan_data.get("macro_context", {})
    vix   = macro.get("vix", 0)
    vix_color = (
        "#E05C5C" if vix >= 30 else
        "#E8A020" if vix >= 20 else
        "#9AA0B4" if vix >= 15 else "#2DB37A"
    )

    summary = plan_data.get("summary", {})
    g, a, gr = summary.get("green_count",0), summary.get("amber_count",0), summary.get("grey_count",0)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-val" style="color:{vix_color}">{vix:.1f}</div>
        <div class="stat-lbl">VIX · {macro.get('vix_regime','')}</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('sp500_5d_chg',0):+.1f}%</div>
        <div class="stat-lbl">S&P 500 · 5-day return</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('yield_curve','—')}</div>
        <div class="stat-lbl">Yield curve · 10Y–3M</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('usd_index_5d_chg',0):+.1f}%</div>
        <div class="stat-lbl">USD · 5-day change</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">
          <span style="color:#2DB37A">{g}</span>
          <span style="color:#5C6378; font-size:1rem"> / </span>
          <span style="color:#E8A020">{a}</span>
          <span style="color:#5C6378; font-size:1rem"> / </span>
          <span style="color:#6B7280">{gr}</span>
        </div>
        <div class="stat-lbl">Act · Consider · Skip</div>
      </div>
    </div>
    <div class="te-meta" style="margin-bottom:1.5rem;">
      Plan generated: {plan_data.get('generated_at','')[:16]}
    </div>
    """, unsafe_allow_html=True)

    # ── Tier legend ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="tier-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#2DB37A"></div>ACT ON MODEL — high confidence, stable signal</div>
      <div class="legend-item"><div class="legend-dot" style="background:#E8A020"></div>USE LOOSELY — review context before acting</div>
      <div class="legend-item"><div class="legend-dot" style="background:#6B7280"></div>SKIP TIMING — keep day 27 schedule</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Ticker cards ───────────────────────────────────────────────────────
    ep = plan_data.get("execution_plan", {})
    active_portfolio = get_active_portfolio()

    # Show only tickers in the user's active portfolio.
    # Unknown tickers (not in cache) get a "queued" card.
    cached_tickers  = set(ep.keys())
    user_tickers    = set(active_portfolio.keys())
    known_tickers   = user_tickers & cached_tickers
    unknown_tickers = user_tickers - cached_tickers

    # Sort known: green first, then amber, then grey; within tier by score desc
    tier_order = {"green": 0, "amber": 1, "grey": 2}
    sorted_tickers = sorted(
        known_tickers,
        key=lambda t: (tier_order.get(ep[t].get("exec_tier","grey"), 2),
                       -ep[t].get("exec_score", 0))
    )

    # Show warning cards for unknown tickers
    if unknown_tickers:
        unknown_list = ", ".join(sorted(unknown_tickers))
        st.markdown(f"""
        <div style="background:rgba(232,160,32,0.07);border:1px solid rgba(232,160,32,0.2);
                    border-radius:6px;padding:0.75rem 1rem;margin-bottom:1rem;
                    font-family:var(--mono);font-size:0.75rem;color:#E8A020;">
            ⏳ <strong>NOT YET ANALYSED:</strong> {unknown_list}<br>
            <span style="color:#9AA0B4;font-size:0.7rem;">
            Go to the Simulator tab → run each ticker once → come back here.
            Results are cached for future visits.
            </span>
        </div>
        """, unsafe_allow_html=True)

    for ticker in sorted_tickers:
        card = ep[ticker]
        tier        = card.get("exec_tier", "grey")
        conf        = card.get("confidence", 0.5)
        rec_day     = card.get("recommended_day", 27)
        top3        = card.get("top_3_days", [rec_day])
        advisory    = card.get("advisory", "")
        window_str  = card.get("suggested_window", "")
        key_factor  = card.get("key_factor", "")
        pred_save   = card.get("predicted_saving", 0.0)
        monthly     = card.get("monthly_usd", 0)
        score       = card.get("exec_score", 0)
        win_rate_pct = card.get("win_rate_pct", None)  # Phase 4 MC win rate

        # Bug 3 fix: use win rate as tiebreaker within each integer band so
        # green tickers don't all score exactly 7/10.
        # Band floors: A→7.0, B→5.0, C→3.0.  Win rate adds 0.0–0.9 within band.
        _TIER_BASE  = {"green": 7.0, "amber": 5.0, "grey": 3.0}
        _TIER_RANGE = {"green": (0.60, 0.85), "amber": (0.50, 0.75), "grey": (0.40, 0.70)}
        if win_rate_pct is not None and tier in _TIER_BASE:
            wr   = win_rate_pct / 100
            lo, hi = _TIER_RANGE[tier]
            clamped = max(lo, min(hi, wr))
            tiebreaker = 0.9 * (clamped - lo) / (hi - lo)
            score = round(_TIER_BASE[tier] + tiebreaker, 1)

        badge_cls = f"badge-{tier}"
        card_cls  = tier
        color     = TIER_COLORS[tier]
        label     = TIER_LABELS[tier]
        conf_pct  = int(conf * 100)
        conf_cls  = f"conf-{tier}"
        conf_display = _format_confidence(ticker, conf_pct)

        top3_str = "  ".join([
            f"<span style='color:{color};font-weight:600'>{d}</span>" if d == rec_day
            else f"<span style='color:#5C6378'>{d}</span>"
            for d in top3
        ])

        st.markdown(f"""
        <div class="ticker-card {card_cls}">
          <div class="card-header">
            <div>
              <div class="card-ticker">{ticker}
                <span style="font-weight:300;font-size:0.85rem;color:#5C6378;margin-left:0.5rem">${monthly}/mo</span>
              </div>
              <div class="card-name">{card.get('name','')}</div>
            </div>
            <div style="display:flex;gap:0.5rem;align-items:center">
              <span style="font-family:var(--mono);font-size:0.72rem;color:#5C6378">score {score}/10</span>
              <span class="card-badge {badge_cls}">{label}</span>
            </div>
          </div>

          <div class="card-rec">
            <div>
              <div class="rec-label">RECOMMENDED DAY</div>
              <div class="rec-day" style="color:{color}">{rec_day}</div>
            </div>
            <div>
              <div class="rec-label">TOP 3 DAYS</div>
              <div style="font-family:var(--mono);font-size:0.95rem;margin-top:4px">{top3_str}</div>
            </div>
            <div>
              <div class="rec-label">CONFIDENCE</div>
              <div class="rec-conf" style="margin-top:4px">{conf_display}</div>
              <div class="conf-bar-wrap" style="width:80px">
                <div class="conf-bar {conf_cls}" style="width:{conf_pct}%"></div>
              </div>
            </div>
            <div>
              <div class="rec-label">PREDICTED SAVING</div>
              <div style="font-family:var(--mono);font-size:0.88rem;color:{color};margin-top:4px">{pred_save:+.2f}%</div>
            </div>
            {"<div><div class='rec-label'>WINDOW</div><div style='font-family:var(--mono);font-size:0.82rem;color:#9AA0B4;margin-top:4px'>" + window_str + "</div></div>" if window_str else ""}
          </div>

          {"<div class='card-advisory'>" + advisory + ("<div class='key-factor'>↳ " + key_factor + "</div>" if key_factor else "") + "</div>" if advisory else ""}
        </div>
        """, unsafe_allow_html=True)

    # Refresh button
    st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
    if st.button("⟳  Refresh Plan"):
        with st.spinner("Refreshing..."):
            try:
                from scripts.monthly_refresh import run_monthly_refresh
                run_monthly_refresh()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page 2: Portfolio Simulator ───────────────────────────────────────────────

def render_simulator():
    st.markdown('<div class="section-label">portfolio simulator</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker", value="VOO",
            help="Any ticker on Yahoo Finance — portfolio or any stock/ETF",
            placeholder="e.g. VOO, ARKK, MSFT, GME"
        ).upper().strip()
    with col2:
        default_mo = get_active_portfolio().get(ticker_input, {}).get("monthly_usd", 50)
        monthly_usd = st.number_input("Monthly $", min_value=1, max_value=50000,
                                       value=default_mo, step=5)
    with col3:
        horizon = st.slider("Years", min_value=1, max_value=30, value=10)

    is_portfolio = ticker_input in get_active_portfolio()
    run_label    = "▶  Run Simulation" if is_portfolio else f"▶  Download + Train + Simulate {ticker_input}"
    run_sim      = st.button(run_label)

    if not is_portfolio and ticker_input:
        st.markdown(f"""
        <div style="background:rgba(77,158,245,0.07);border:1px solid rgba(77,158,245,0.2);
                    border-radius:6px;padding:0.65rem 1rem;margin-bottom:0.75rem;
                    font-size:0.78rem;color:#4D9EF5;font-family:var(--mono)">
            {ticker_input} is not in your portfolio.
            Running will: download history → build 121 features → train LightGBM
            → 1,000 Monte Carlo runs → LLM advisory. ~1-3 min first time, cached after.
        </div>
        """, unsafe_allow_html=True)

    # Check portfolio cache first
    sim_results  = load_simulation_results()
    cached_result = None

    if is_portfolio and sim_results and ticker_input in sim_results:
        cached_result = sim_results[ticker_input]
    else:
        # Check on-demand cache
        from src.pipeline.on_demand import _cache_valid, _load_cache
        if _cache_valid(ticker_input):
            cached_result = _load_cache(ticker_input)

    # Auto-show cached results without needing to click Run
    if cached_result and not run_sim:
        st.markdown(f"""
        <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;margin-bottom:0.5rem">
            Showing cached result — click Run to refresh
        </div>
        """, unsafe_allow_html=True)

    if run_sim or cached_result:
        result = cached_result

        if run_sim:
            if is_portfolio:
                with st.spinner(f"Running Monte Carlo for {ticker_input}..."):
                    try:
                        from src.simulation.monte_carlo import run_simulation
                        r = run_simulation(ticker_input, float(monthly_usd))
                        result = r
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        return
            else:
                # Full on-demand pipeline with progress bar
                progress_bar = st.progress(0)
                status_text  = st.empty()

                def update_progress(fraction: float, message: str):
                    progress_bar.progress(min(fraction, 1.0))
                    status_text.markdown(
                        f'<div style="font-family:var(--mono);font-size:0.75rem;'
                        f'color:#9AA0B4">{message}</div>',
                        unsafe_allow_html=True,
                    )

                try:
                    from src.pipeline.on_demand import run_on_demand
                    result = run_on_demand(
                        ticker_input,
                        monthly_usd=float(monthly_usd),
                        horizon_years=horizon,
                        n_runs=1000,
                        progress_cb=update_progress,
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    return

        if result is None:
            st.info("No results. Click Run Simulation.")
            return

        if "error" in result and result["error"]:
            st.error(f"❌ {result['error']}")
            return

        summary = result.get("summary", {})
        if not summary:
            st.error("Invalid result structure.")
            return

        win    = summary.get("win_rate_pct", 0)
        save   = summary.get("avg_saving_pct", 0)
        p      = summary.get("percentiles", {})
        tier   = summary.get("tier", "B")
        proj   = summary.get("projections", {})
        regime = summary.get("regime_stats", {})
        meta   = summary.get("ticker_meta", {})

        # Trust score fields
        trust_score    = summary.get("trust_score", None)
        tier_label     = summary.get("tier_label", "")
        trust_reasons  = summary.get("trust_reasons", [])
        trust_penalties= summary.get("trust_penalties", [])
        trust_summary  = summary.get("trust_summary", "")
        proj_note      = summary.get("projection_note", "")
        show_optimistic= summary.get("show_optimistic", True)
        proj_scenarios = summary.get("projections_by_scenario", {})
        clip_bounds    = summary.get("clip_bounds", {})
        n_clipped      = summary.get("clipped_values", 0)

        # Info banner for non-portfolio tickers
        if not is_portfolio and meta:
            beta   = float(meta.get("beta", 1.0) or 1.0)
            sector = meta.get("sector", "Unknown")
            name   = meta.get("name", ticker_input)
            elapsed = result.get("elapsed_s", 0)
            st.markdown(f"""
            <div style="background:rgba(45,179,122,0.07);border:1px solid rgba(45,179,122,0.2);
                        border-radius:6px;padding:0.65rem 1rem;margin-bottom:0.75rem;
                        font-size:0.78rem;color:#2DB37A;font-family:var(--mono)">
                ✓ {name} | sector: {sector} | beta: {beta:.2f}
                | trained on real {ticker_input} data in {elapsed:.0f}s
            </div>
            """, unsafe_allow_html=True)

        # Trust score panel
        if trust_score is not None:
            tier_color = {"A":"#2DB37A","B":"#E8A020","C":"#E05C5C"}.get(tier,"#9AA0B4")
            trust_color = "#2DB37A" if trust_score>=65 else "#E8A020" if trust_score>=40 else "#E05C5C"
            reasons_html = "".join([
                f'<span style="color:#2DB37A;margin-right:0.75rem">✓ {r}</span>'
                for r in trust_reasons
            ])
            penalties_html = "".join([
                f'<span style="color:#E05C5C;margin-right:0.75rem">✗ {r}</span>'
                for r in trust_penalties
            ])
            clip_note = ""
            if n_clipped > 0 and clip_bounds:
                clip_note = (f'<div style="margin-top:0.4rem;font-size:0.70rem;color:#5C6378">'
                             f'{n_clipped} extreme values clipped to '
                             f'[{clip_bounds.get("low",0):.1f}%, '
                             f'+{clip_bounds.get("high",0):.1f}%] — realistic bounds for this asset type'
                             f'</div>')
            st.markdown(f"""
            <div style="background:#141720;border:1px solid #252A38;border-radius:8px;
                        padding:1rem 1.25rem;margin-bottom:1rem">
              <div style="display:flex;justify-content:space-between;align-items:baseline;
                          margin-bottom:0.6rem">
                <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                            letter-spacing:0.1em">CAN I TRUST THIS RESULT?</div>
                <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378">
                  Tier <span style="color:{tier_color};font-weight:600">{tier}</span>
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:0.5rem">
                <div>
                  <span style="font-family:var(--mono);font-size:2rem;font-weight:600;
                               color:{trust_color}">{trust_score}</span>
                  <span style="font-family:var(--mono);font-size:0.78rem;color:#5C6378">/100</span>
                </div>
                <div>
                  <div style="background:#252A38;border-radius:4px;height:8px;width:200px">
                    <div style="background:{trust_color};border-radius:4px;height:8px;
                                width:{trust_score*2}px"></div>
                  </div>
                  <div style="font-size:0.75rem;color:#9AA0B4;margin-top:4px">{tier_label}</div>
                </div>
              </div>
              <div style="font-size:0.75rem;line-height:1.6">
                {reasons_html}
              </div>
              {f'<div style="font-size:0.75rem;line-height:1.6;margin-top:0.3rem">{penalties_html}</div>' if penalties_html else ""}
              {clip_note}
            </div>
            """, unsafe_allow_html=True)

        # Projection note for Tier B/C
        if proj_note:
            note_color = "#E05C5C" if tier == "C" else "#E8A020"
            st.markdown(f"""
            <div style="background:rgba(224,92,92,0.06) if tier=='C' else rgba(232,160,32,0.06);
                        border:1px solid {note_color}40;border-radius:6px;
                        padding:0.6rem 1rem;margin-bottom:0.75rem;
                        font-size:0.75rem;color:{note_color}">
                {proj_note}
            </div>
            """, unsafe_allow_html=True)

        tier_color = {"A": "#2DB37A", "B": "#E8A020", "C": "#6B7280"}.get(tier, "#9AA0B4")
        tier_label = {"A": "High precision", "B": "Medium precision", "C": "Low precision"}.get(tier, "")

        # ── Summary stats ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-val" style="color:{'#2DB37A' if win>=65 else '#E8A020' if win>=55 else '#6B7280'}">{win:.1f}%</div>
            <div class="stat-lbl">Win rate vs day 27</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" style="color:{'#2DB37A' if save>0 else '#E05C5C'}">{save:+.3f}%</div>
            <div class="stat-lbl">Avg saving per month</div>
          </div>
          <div class="stat-box">
            <div class="stat-val stat-white" style="font-size:1rem">
              {p.get('p10',0):+.2f}% / {p.get('p50',0):+.2f}% / {p.get('p90',0):+.2f}%
            </div>
            <div class="stat-lbl">P10 / P50 / P90</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" style="color:{tier_color}">Tier {tier}</div>
            <div class="stat-lbl">{tier_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column charts ──────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            # Compounding fan chart
            base_rate = 0.15
            opt_rate  = base_rate + (save/100 * 0.45 * 12 * 0.3)
            months    = np.arange(1, horizon*12+1)
            r_b = base_rate/12
            r_o = opt_rate/12
            fv_b = monthly_usd * ((1+r_b)**months - 1) / r_b
            fv_o = monthly_usd * ((1+r_o)**months - 1) / r_o

            # Bug 4 fix: Tier C suppresses the optimistic rate in the chart
            # title to match the table (which already hides the Optimistic row).
            conservative_rate = proj_scenarios.get("Conservative", {}).get("opt_rate", base_rate*100) if proj_scenarios else base_rate*100
            if tier == "C":
                chart_rate_label = f"{conservative_rate:.1f}% effective rate (conservative)"
            else:
                chart_rate_label = f"{opt_rate*100:.1f}% effective rate"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months/12, y=fv_b,
                name=f"Fixed day 27 ({base_rate*100:.0f}%)",
                line=dict(color="#5C6378", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=months/12, y=fv_o,
                name=f"AI-optimised ({opt_rate*100:.1f}%)",
                line=dict(color="#2DB37A", width=2.5),
                fill="tonexty", fillcolor="rgba(45,179,122,0.08)",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"{ticker_input} — {horizon}yr compounding ({chart_rate_label})", font=dict(size=12)),
                xaxis=dict(**AXIS_DEFAULTS, title="Years"),
                yaxis=dict(**AXIS_DEFAULTS, tickprefix="$", tickformat=",.0f"),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Regime breakdown
            if regime:
                r_names  = list(regime.keys())
                r_wins   = [regime[r]["win_rate"] for r in r_names]
                r_saves  = [regime[r]["avg_saving"] for r in r_names]
                r_colors = {"fear":"#E05C5C","elevated":"#E8A020","normal":"#9AA0B4","calm":"#2DB37A"}
                bar_colors = [r_colors.get(r, "#5C6378") for r in r_names]

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Bar(
                    x=r_names, y=r_wins, name="Win %",
                    marker_color=bar_colors, opacity=0.8,
                ), secondary_y=False)
                fig2.add_trace(go.Scatter(
                    x=r_names, y=r_saves, name="Avg save %",
                    line=dict(color="#4D9EF5", width=2), mode="lines+markers",
                    marker=dict(size=7),
                ), secondary_y=True)
                fig2.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text="Performance by VIX regime", font=dict(size=12)),
                    height=300, showlegend=True,
                )
                fig2.update_yaxes(title_text="Win rate %", secondary_y=False,
                                  gridcolor="#252A38", linecolor="#252A38", showgrid=True)
                fig2.update_yaxes(title_text="Avg saving %", secondary_y=True,
                                  gridcolor="rgba(0,0,0,0)", showgrid=False)
                st.plotly_chart(fig2, use_container_width=True)

        # ── Projection table — tier-aware scenarios ───────────────────
        if proj_scenarios:
            scenario_names = list(proj_scenarios.keys())
            if not show_optimistic and "Optimistic" in scenario_names:
                scenario_names = [s for s in scenario_names if s != "Optimistic"]

            label_str = " · ".join([
                f"{s} ({proj_scenarios[s].get('base_rate',0):.0f}% base, "
                f"{proj_scenarios[s].get('capture_rate',0):.0f}% capture)"
                for s in scenario_names
            ])
            st.markdown(f'<div class="section-label">compounding projections — {label_str}</div>',
                        unsafe_allow_html=True)

            rows = []
            for yr in [1, 3, 5, 10, 20]:
                row = {"Years": f"{yr}yr"}
                for s_name in scenario_names:
                    s_data = proj_scenarios[s_name].get("projections", {})
                    if yr in s_data:
                        extra = s_data[yr].get("extra", 0)
                        row[f"{s_name} extra"] = f"+${extra:,.0f}"
                        row[f"{s_name} lift"]  = f"+{s_data[yr].get('lift_pct',0):.1f}%"
                rows.append(row)

            df_proj = pd.DataFrame(rows)
            st.dataframe(df_proj, use_container_width=True, hide_index=True)

        elif proj:
            # Fallback for portfolio tickers using old format
            st.markdown('<div class="section-label">compounding projections — base case</div>',
                        unsafe_allow_html=True)
            rows = []
            for yr, p_data in sorted(proj.items()):
                extra = p_data.get("extra", 0)
                rows.append({
                    "Years": f"{yr}yr",
                    "Fixed day 27": f"${p_data.get('fixed',0):,.0f}",
                    "AI-optimised": f"${p_data.get('optimised',0):,.0f}",
                    "Extra wealth":  f"+${extra:,.0f}",
                    "Lift": f"+{extra/max(p_data.get('fixed',1),1)*100:.1f}%",
                })
            df_proj = pd.DataFrame(rows)
            st.dataframe(df_proj, use_container_width=True, hide_index=True)


# ── Page 3: Historical Evidence ───────────────────────────────────────────────

def render_evidence():
    st.markdown('<div class="section-label">historical evidence — walk-forward backtest + monte carlo</div>',
                unsafe_allow_html=True)

    sim = load_simulation_results()
    if sim is None:
        st.info("Run Phase 4 first: `python run_phase4.py`")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    phase3 = {
        "VOO":77.5,"VTI":76.3,"NVDA":71.0,"AAPL":79.0,"SCHD":77.5,
        "VXUS":70.1,"TSLA":71.4,"BND":74.5,"VYM":78.2,"VEA":76.7,"VWO":77.0,
    }
    rows = []
    for ticker, data in sim.items():
        s    = data.get("summary", {})
        p3   = phase3.get(ticker, 0)
        p4   = s.get("win_rate_pct", 0)
        delta = p4 - p3
        save  = s.get("avg_saving_pct", 0)
        p     = s.get("percentiles", {})
        tier  = s.get("tier", "B")
        ann   = s.get("avg_dollar_per_year", 0)
        rows.append({
            "Ticker": ticker,
            "Tier": tier,
            "P3 Win%": f"{p3:.1f}%",
            "P4 Win%": f"{p4:.1f}%",
            "Δ": f"{delta:+.1f}pp",
            "Avg Save": f"{save:+.3f}%",
            "P10": f"{p.get('p10',0):+.2f}%",
            "P50": f"{p.get('p50',0):+.2f}%",
            "P90": f"{p.get('p90',0):+.2f}%",
            "$/yr": f"+${ann:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Phase 3 vs Phase 4 consistency chart ──────────────────────────────
    st.markdown('<div class="section-label">phase 3 vs phase 4 consistency — key reliability test</div>',
                unsafe_allow_html=True)

    tickers  = list(sim.keys())
    p3_wins  = [phase3.get(t, 0) for t in tickers]
    p4_wins  = [sim[t]["summary"].get("win_rate_pct", 0) for t in tickers]
    deltas   = [p4-p3 for p4, p3 in zip(p4_wins, p3_wins)]
    colors   = ["#2DB37A" if abs(d)<=5 else "#E8A020" if abs(d)<=10 else "#E05C5C" for d in deltas]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=tickers, y=deltas, marker_color=colors, name="P4 - P3 delta"))
    fig3.add_hline(y=5,  line_dash="dot", line_color="#2DB37A", opacity=0.4)
    fig3.add_hline(y=-5, line_dash="dot", line_color="#2DB37A", opacity=0.4)
    fig3.add_hrect(y0=-5, y1=5, fillcolor="rgba(45,179,122,0.04)", line_width=0)
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Win rate delta: Phase 4 vs Phase 3 (green band = consistent signal)",
                   font=dict(size=12)),
        xaxis=dict(**AXIS_DEFAULTS),
        yaxis=dict(**AXIS_DEFAULTS, title="Δ percentage points"),
        height=320,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── 3-scenario compounding chart ──────────────────────────────────────
    st.markdown('<div class="section-label">20-year compounding — 3 scenarios ($55/month)</div>',
                unsafe_allow_html=True)

    avg_save = np.mean([sim[t]["summary"].get("avg_saving_pct", 0) for t in sim])
    base     = 0.15
    total_mo = 55
    scenarios = [
        ("Conservative", 0.10, 0.25, "#5C6378"),
        ("Base case",    0.15, 0.45, "#2DB37A"),
        ("Optimistic",   0.15, 1.00, "#4D9EF5"),
    ]

    fig4 = go.Figure()
    months = np.arange(1, 241)
    for name, base_r, cap, color in scenarios:
        opt_r = base_r + (avg_save/100 * cap * 12 * 0.3)
        r     = opt_r/12
        fv    = total_mo * ((1+r)**months - 1) / r
        fig4.add_trace(go.Scatter(
            x=months/12, y=fv, name=f"{name} ({opt_r*100:.1f}%)",
            line=dict(color=color, width=2, dash="dash" if name=="Conservative" else "solid"),
        ))
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="20-year wealth projection — fixed day 27 baseline (grey dashed)",
                   font=dict(size=12)),
        xaxis=dict(**AXIS_DEFAULTS, title="Years"),
        yaxis=dict(**AXIS_DEFAULTS, tickprefix="$", tickformat=",.0f"),
        height=340,
    )
    # Add fixed-day-27 baseline
    r_b  = 0.15/12
    fv_b = total_mo * ((1+r_b)**months - 1) / r_b
    fig4.add_trace(go.Scatter(
        x=months/12, y=fv_b, name="Fixed day 27 (15%)",
        line=dict(color="#3A3F52", width=1.5, dash="dash"),
    ))
    st.plotly_chart(fig4, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────


# ── Sidebar: Portfolio Setup ──────────────────────────────────────────────────

def render_sidebar():
    """
    Sidebar portfolio editor.
    Users can customise their tickers + monthly amounts.
    Everything stays in st.session_state — no data ever leaves their browser.
    """
    with st.sidebar:
        st.markdown("""
        <div style="font-family:var(--mono);font-size:0.9rem;font-weight:600;
                    color:#E8EAF0;letter-spacing:0.06em;margin-bottom:0.25rem;">
            MY PORTFOLIO
        </div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                    margin-bottom:1rem;line-height:1.5;">
            Your data never leaves this browser session.<br>
            Resets on refresh — no account needed.
        </div>
        """, unsafe_allow_html=True)

        # ── Initialise from defaults on first load ────────────────────────
        if "user_portfolio" not in st.session_state:
            st.session_state["user_portfolio"] = dict(_DEFAULT_PORTFOLIO)
        if "sidebar_rows" not in st.session_state:
            st.session_state["sidebar_rows"] = [
                {"ticker": t, "amount": v["monthly_usd"]}
                for t, v in _DEFAULT_PORTFOLIO.items()
            ]

        rows = st.session_state["sidebar_rows"]

        # ── Editable rows ─────────────────────────────────────────────────
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.68rem;'
            'color:#9AA0B4;margin-bottom:0.4rem;">TICKER · $/MONTH</div>',
            unsafe_allow_html=True
        )

        updated_rows = []
        for i, row in enumerate(rows):
            c1, c2, c3 = st.columns([2, 1.5, 0.6])
            with c1:
                t = st.text_input(
                    f"t_{i}", value=row["ticker"],
                    label_visibility="collapsed",
                    key=f"ticker_input_{i}",
                    placeholder="e.g. VOO"
                ).upper().strip()
            with c2:
                a = st.number_input(
                    f"a_{i}", value=float(row["amount"]),
                    min_value=1.0, max_value=50000.0, step=1.0,
                    label_visibility="collapsed",
                    key=f"amount_input_{i}",
                    format="%.0f"
                )
            with c3:
                if st.button("✕", key=f"del_{i}", help="Remove"):
                    # Remove this row — rerun will rebuild
                    st.session_state["sidebar_rows"].pop(i)
                    st.rerun()
            if t:
                updated_rows.append({"ticker": t, "amount": int(a)})

        st.session_state["sidebar_rows"] = updated_rows

        # ── Add ticker button ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        if len(rows) < 15:
            if st.button("＋  Add ticker", use_container_width=True):
                st.session_state["sidebar_rows"].append({"ticker": "", "amount": 10})
                st.rerun()
        else:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.68rem;'
                'color:#5C6378;">Max 15 tickers</div>',
                unsafe_allow_html=True
            )

        st.markdown("<hr style='border-color:#252A38;margin:1rem 0'>", unsafe_allow_html=True)

        # ── Apply / Reset buttons ─────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✓  Apply", use_container_width=True, type="primary"):
                new_portfolio = {}
                for row in st.session_state["sidebar_rows"]:
                    t = row["ticker"].upper().strip()
                    if t:
                        new_portfolio[t] = {
                            "name":        t,
                            "monthly_usd": int(row["amount"]),
                        }
                if new_portfolio:
                    st.session_state["user_portfolio"] = new_portfolio
                    st.success(f"Portfolio updated — {len(new_portfolio)} tickers")
                    # Clear old execution plan cache so it re-renders
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Add at least one ticker")

        with col_b:
            if st.button("↺  Reset", use_container_width=True):
                st.session_state["user_portfolio"] = dict(_DEFAULT_PORTFOLIO)
                st.session_state["sidebar_rows"] = [
                    {"ticker": t, "amount": v["monthly_usd"]}
                    for t, v in _DEFAULT_PORTFOLIO.items()
                ]
                st.rerun()

        # ── Portfolio summary ─────────────────────────────────────────────
        active = get_active_portfolio()
        total_mo = sum(v["monthly_usd"] for v in active.values())
        n        = len(active)
        st.markdown(f"""
        <div style="background:#141720;border:1px solid #252A38;border-radius:6px;
                    padding:0.6rem 0.8rem;margin-top:0.75rem;">
            <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                        margin-bottom:0.4rem;">ACTIVE PORTFOLIO</div>
            <div style="font-family:var(--mono);font-size:0.85rem;color:#E8EAF0;">
                {n} tickers · <span style="color:#2DB37A">${total_mo}/mo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Privacy note
        st.markdown("""
        <div style="font-family:var(--mono);font-size:0.63rem;color:#3C4258;
                    margin-top:1rem;line-height:1.6;">
            🔒 Session only. No data stored or transmitted.<br>
            Resets when you close the tab.
        </div>
        """, unsafe_allow_html=True)


def main():
    render_sidebar()
    render_header("plan")

    tab1, tab2, tab3 = st.tabs([
        "⬡  EXECUTION PLAN",
        "◎  SIMULATOR",
        "▦  EVIDENCE",
    ])

    with tab1:
        render_execution_plan()

    with tab2:
        render_simulator()

    with tab3:
        render_evidence()


if __name__ == "__main__":
    main()
