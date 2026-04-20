"""
TemporalEdge — Central Config
Single source of truth. Change things here, everything updates.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_RAW    = ROOT / "data" / "raw"
DATA_FEAT   = ROOT / "data" / "features"
DATA_MACRO  = ROOT / "data" / "macro"
DATA_CACHE  = ROOT / "data" / "cache"

for p in [DATA_RAW, DATA_FEAT, DATA_MACRO, DATA_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

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
# These are fetched alongside portfolio tickers but stored separately
MACRO_TICKERS = {
    # Volatility & Sentiment
    "^VIX":       "vix",
    "^VXN":       "vix_nasdaq",   # Nasdaq VIX — if download fails, skipped gracefully

    # Equity Benchmarks
    "^GSPC":      "sp500",
    "^NDX":       "nasdaq100",
    "^RUT":       "russell2000",

    # Rates & Fixed Income
    "^TNX":       "us10y_yield",
    "^IRX":       "us3m_yield",
    "^TYX":       "us30y_yield",

    # Currencies
    "DX-Y.NYB":   "usd_index",
    "EURUSD=X":   "eur_usd",
    "JPY=X":      "usd_jpy",
    "CNY=X":      "usd_cny",

    # Commodities
    "GC=F":       "gold",
    "CL=F":       "oil_wti",
    "BZ=F":       "oil_brent",
    "HG=F":       "copper",
    "NG=F":       "nat_gas",
    "SI=F":       "silver",

    # Credit / Risk
    "HYG":        "high_yield_etf",
    "LQD":        "invest_grade_etf",

    # Sector Proxies
    "QQQ":        "tech_etf",
    "XLF":        "financials_etf",
    "XLE":        "energy_etf",
    "SMH":        "semis_etf",
}

# ── Date Range ───────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"   # yfinance clips automatically per ticker's IPO

# ── Purchase Window ──────────────────────────────────────────────────────────
# Days of month considered for optimal buy (your current = 27th/28th)
PURCHASE_WINDOW_START = 18   # start of window
PURCHASE_WINDOW_END   = 5    # end = first 5 days of NEXT month
YOUR_CURRENT_DAY      = 27   # your Robinhood recurring day

# ── Feature Engineering ──────────────────────────────────────────────────────
# Rolling windows used for technical indicators
ROLLING_WINDOWS = [5, 10, 21, 63]       # ~1w, 2w, 1mo, 1q
FORWARD_WINDOWS = [7, 14, 21, 30]       # prediction horizons (days)

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
    "n_jobs":           -1,   # uses all M2 cores
}

WALK_FORWARD_TRAIN_MONTHS = 36   # 3 years training window
WALK_FORWARD_TEST_MONTHS  = 1    # predict 1 month at a time

# ── Monte Carlo ──────────────────────────────────────────────────────────────
MONTE_CARLO_RUNS_BACKTEST = 10_000
MONTE_CARLO_RUNS_LIVE     = 1_000    # faster for web app requests

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
