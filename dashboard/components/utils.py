"""
TemporalEdge — Shared utilities and constants for dashboard components.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.config import ROOT

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

# ── Display constants ─────────────────────────────────────────────────────────
TIER_COLORS = {"green": "#2DB37A", "amber": "#E8A020", "grey": "#6B7280"}
TIER_LABELS = {
    "green": "ACT ON MODEL",
    "amber": "USE LOOSELY",
    "grey":  "SKIP TIMING",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#141720",
    plot_bgcolor="#141720",
    font=dict(family="IBM Plex Mono, monospace", color="#9AA0B4", size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#252A38"),
)
AXIS_DEFAULTS = dict(gridcolor="#252A38", linecolor="#252A38", showgrid=True)


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

CACHE_FILE       = Path(__file__).parent.parent / "cache" / "execution_plan.json"
RESULTS_FILE     = Path(__file__).parent.parent.parent / "results" / "portfolio_simulation.json"
PRECOMPUTED_DIR  = Path(__file__).parent.parent.parent / "data" / "precomputed"
ONDEMAND_DIR     = Path(__file__).parent.parent / "cache" / "on_demand"

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

def load_ondemand_result(ticker: str) -> dict | None:
    """
    Load cached result for a ticker. Priority order:
    1. On-demand cache (fresh user run) — always preferred
    2. Precomputed cache (Phase 8 batch run) — fallback
    Returns None if neither exists.
    """
    from datetime import datetime, timedelta

    def _valid(path: Path, max_days: int = 35) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=max_days)

    # Check on-demand first (fresher, user-specific)
    od_path = ONDEMAND_DIR / f"{ticker}_result.json"
    if _valid(od_path):
        try:
            return json.loads(od_path.read_text())
        except Exception:
            pass

    # Fallback to precomputed — these never expire, ignore age
    pc_path = PRECOMPUTED_DIR / f"{ticker}_result.json"
    if pc_path.exists():
        try:
            return json.loads(pc_path.read_text())
        except Exception:
            pass

    return None


def is_cached(ticker: str) -> bool:
    """True if a valid result exists in either cache."""
    return load_ondemand_result(ticker) is not None


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