"""
TemporalEdge — Central Config
Single source of truth. Change things here, everything updates.

Phase 6 change: paths are now cloud-aware.
On Streamlit Cloud the repo root is read-only; writable dirs are
redirected to /tmp so data/model downloads don't crash.
"""

import os
from pathlib import Path

# ── Environment detection ─────────────────────────────────────────────────────
# Streamlit Cloud sets IS_STREAMLIT_CLOUD or we detect via missing home dir
_IS_CLOUD = (
    os.environ.get("IS_STREAMLIT_CLOUD") == "1"
    or os.environ.get("STREAMLIT_SHARING_MODE") == "1"
    or not Path("/Users").exists()          # not a Mac/local machine
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

if _IS_CLOUD:
    # Streamlit Cloud: repo is read-only after deploy.
    # Write all generated data to /tmp which is writable.
    _WRITABLE = Path("/tmp/temporaledge")
else:
    _WRITABLE = ROOT

DATA_RAW    = _WRITABLE / "data" / "raw"
DATA_FEAT   = _WRITABLE / "data" / "features"
DATA_MACRO  = _WRITABLE / "data" / "macro"
DATA_CACHE  = _WRITABLE / "data" / "cache"

# Dashboard cache lives inside the repo so it can be committed by GitHub Actions
# and served directly on both local and cloud.
DASHBOARD_CACHE = ROOT / "dashboard" / "cache"

for p in [DATA_RAW, DATA_FEAT, DATA_MACRO, DATA_CACHE, DASHBOARD_CACHE,
          DASHBOARD_CACHE / "on_demand"]:
    p.mkdir(parents=True, exist_ok=True)

# ── MLflow ───────────────────────────────────────────────────────────────────
# Disabled on cloud (no persistent sqlite). Trainer falls back gracefully.
MLFLOW_ENABLED = not _IS_CLOUD
MLFLOW_URI     = f"sqlite:///{ROOT}/mlflow.db" if MLFLOW_ENABLED else None

# ── Your Portfolio ───────────────────────────────────────────────────────────
PORTFOLIO = {
    "VOO":  {"name": "Vanguard S&P 500 ETF",           "monthly_usd": 15, "type": "us_etf",       "sector": "broad_market"},
    "VTI":  {"name": "Vanguard Total Stock Market",     "monthly_usd": 10, "type": "us_etf",       "sector": "broad_market"},
    "NVDA": {"name": "NVIDIA",                          "monthly_usd":  5, "type": "stock",        "sector": "tech"},
    "AAPL": {"name": "Apple",                           "monthly_usd":  5, "type": "stock",        "sector": "tech"},
    "SCHD": {"name": "Schwab Dividend ETF",             "monthly_usd":  4, "type": "dividend_etf", "sector": "dividend"},
    "VXUS": {"name": "Vanguard Total International",    "monthly_usd":  5, "type": "intl_etf",     "sector": "international"},
    "TSLA": {"name": "Tesla",                           "monthly_usd":  2, "type": "stock",        "sector": "ev_tech"},
    "BND":  {"name": "Vanguard Total Bond Market",      "monthly_usd":  3, "type": "bond_etf",     "sector": "fixed_income"},
    "VYM":  {"name": "Vanguard High Dividend Yield",    "monthly_usd":  2, "type": "dividend_etf", "sector": "dividend"},
    "VEA":  {"name": "Vanguard FTSE Developed Markets", "monthly_usd":  2, "type": "intl_etf",     "sector": "international"},
    "VWO":  {"name": "Vanguard FTSE Emerging Markets",  "monthly_usd":  2, "type": "em_etf",       "sector": "emerging"},
}

# ── Macro / Reference Tickers ────────────────────────────────────────────────
MACRO_TICKERS = {
    "^VIX":       "vix",
    "^VXN":       "vix_nasdaq",
    "^GSPC":      "sp500",
    "^NDX":       "nasdaq100",
    "^RUT":       "russell2000",
    "^TNX":       "us10y_yield",
    "^IRX":       "us3m_yield",
    "^TYX":       "us30y_yield",
    "DX-Y.NYB":   "usd_index",
    "EURUSD=X":   "eur_usd",
    "JPY=X":      "usd_jpy",
    "CNY=X":      "usd_cny",
    "GC=F":       "gold",
    "CL=F":       "oil_wti",
    "BZ=F":       "oil_brent",
    "HG=F":       "copper",
    "NG=F":       "nat_gas",
    "SI=F":       "silver",
    "HYG":        "high_yield_etf",
    "LQD":        "invest_grade_etf",
    "QQQ":        "tech_etf",
    "XLF":        "financials_etf",
    "XLE":        "energy_etf",
    "SMH":        "semis_etf",
}

# ── Date Range ───────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"

# ── Purchase Window ──────────────────────────────────────────────────────────
PURCHASE_WINDOW_START = 18
PURCHASE_WINDOW_END   = 5
YOUR_CURRENT_DAY      = 27

# ── Feature Engineering ──────────────────────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 21, 63]
FORWARD_WINDOWS = [7, 14, 21, 30]

# ── Model ────────────────────────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "verbose":          -1,
    "n_jobs":           -1,
}

WALK_FORWARD_TRAIN_MONTHS = 36
WALK_FORWARD_TEST_MONTHS  = 1

# ── Monte Carlo ──────────────────────────────────────────────────────────────
MONTE_CARLO_RUNS_BACKTEST = 10_000
MONTE_CARLO_RUNS_LIVE     = 1_000

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
