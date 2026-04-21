"""
TemporalEdge — Feature Selector  (Phase 3, Step 1)
Selects the best features per ticker for LightGBM training.
Informed by Phase 2 EDA findings:
  - Technical (SMA position, RSI, Stochastic) = 34.9% importance
  - Momentum (5d, 10d returns) = 27.5% importance
  - Calendar (day_of_month, days_to_month_end) = 16.9% importance
  - Macro (VIX, yields, oil) = 13.3% importance
  - Volatility (ATR, BB width) = 5.7% importance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger("models.features")

# ── Features to ALWAYS exclude ────────────────────────────────────────────────
# These are target leakage, identifiers, or raw price levels
EXCLUDE_ALWAYS = {
    "close", "open", "high", "low", "volume",         # raw prices
    "ticker",                                          # identifier
    "is_optimal_buy_day", "vs_day27_pct",              # targets
    "day27_price",                                     # leakage
    "fwd_ret_7d", "fwd_ret_14d",                       # future returns
    "fwd_ret_21d", "fwd_ret_30d",                      # future returns
    "sma_5", "sma_10", "sma_21", "sma_63",            # raw price levels (use ratio instead)
    "bb_upper", "bb_lower",                            # raw price levels
    "macd", "macd_signal",                             # raw levels (use hist instead)
    "atr_14",                                          # raw ATR (use pct instead)
}

# ── Core feature groups (informed directly by EDA importance ranking) ─────────
FEATURE_GROUPS = {

    # Group 1: Technical — 34.9% importance in EDA
    # Price position relative to moving averages is the #1 signal
    "technical": [
        "price_vs_sma5",        # price below 5-day SMA = short pullback
        "price_vs_sma10",       # price below 10-day SMA = medium pullback
        "price_vs_sma21",       # price vs 1-month average
        "price_vs_sma63",       # price vs 1-quarter average (trend context)
        "bb_position",          # where in Bollinger Band (0=lower, 1=upper)
        "bb_width",             # band width = volatility regime
        "rsi_7",                # 7-day RSI — more responsive than 14
        "rsi_14",               # 14-day RSI — standard
        "stoch_k",              # stochastic %K
        "stoch_d",              # stochastic %D (smoothed)
        "macd_hist",            # MACD histogram (momentum direction)
        "atr_pct",              # ATR as % of price (volatility intensity)
    ],

    # Group 2: Momentum — 27.5% importance
    # Recent return direction predicts good entry timing
    "momentum": [
        "ret_5d",               # 5-day return (short momentum)
        "ret_10d",              # 10-day return
        "ret_21d",              # 21-day return (monthly momentum)
        "ret_63d",              # 63-day return (quarterly trend)
        "sp500_ret_1d",         # S&P 500 yesterday (market context)
        "sp500_ret_5d",         # S&P 500 5-day (market momentum)
        "sp500_ret_21d",        # S&P 500 monthly trend
    ],

    # Group 3: Calendar — 16.9% importance
    # Systematic intra-month timing effects
    "calendar": [
        "day_of_month",                # raw day number
        "days_to_month_end",           # countdown to month end
        "days_from_month_start",       # days since month start
        "week_of_month",               # which week (1-5)
        "day_of_week",                 # Mon-Fri (0-4)
        "is_options_expiry",           # exact options expiry day
        "days_to_options_expiry",      # countdown to expiry
        "near_options_expiry",         # within 3 days of expiry
        "month_end_rebal_window",      # last 3 days of month
        "is_quarter_end",              # quarter-end rebalancing
        "is_fomc_month",               # Fed meeting this month
        "is_nfp_week",                 # jobs report week (1st Friday)
        "is_january_effect",           # Jan 1-10 effect
        "is_tax_loss_season",          # Dec 15-31 tax loss harvesting
    ],

    # Group 4: Macro — 13.3% importance (RF underestimates, LightGBM will extract more)
    # VIX regime is the clearest signal from EDA section 4
    "macro": [
        "vix",                         # VIX level (fear gauge)
        "vix_change_5d",               # VIX direction (rising fear?)
        "vix_regime",                  # 0=calm, 1=normal, 2=elevated, 3=fear
        "sp500_vol_21d",               # realised market volatility
        "yield_change_5d",             # rate direction (critical for BND)
        "yield_curve_10y3m",           # yield curve shape (recession signal)
        "gold_ret_5d",                 # gold = risk-off proxy
        "oil_ret_5d",                  # oil = inflation + growth signal
        "copper_ret_21d",              # copper = global growth "Dr. Copper"
        "usd_ret_5d",                  # USD direction (critical for VXUS/VEA/VWO)
        "credit_spread_proxy",         # HYG vs LQD = credit risk appetite
        "fed_funds_rate",              # current rate regime
        "yield_curve_10y2y",           # 2s10s spread (FRED official)
    ],

    # Group 5: Volatility regime — 5.7% importance (but important for position sizing)
    "volatility": [
        "vol_5d",                      # 5-day realised vol
        "vol_21d",                     # 21-day realised vol
        "vol_ratio_5d",                # vol vs recent average (vol spike?)
        "vol_ratio_21d",               # vol vs monthly average
        "daily_range",                 # intraday high-low range
        "range_21d_avg",               # average intraday range
    ],

    # Group 6: Macro releases — event-driven signals
    "macro_releases": [
        "cpi_yoy",                     # inflation level
        "core_pce",                    # Fed's preferred inflation
        "unemployment",                # labour market
        "consumer_sentiment",          # forward-looking demand
        "fed_funds_rate",              # current rate environment
        "tips_10y",                    # real yield
    ],
}

# ── Ticker-specific feature priorities ────────────────────────────────────────
# Based on EDA correlation findings per ticker
TICKER_PRIORITY_FEATURES = {
    "NVDA": [
        "near_options_expiry",     # #1 for NVDA from EDA section 4
        "days_to_options_expiry",
        "vix_change_5d",           # #2 for NVDA
        "sp500_ret_5d",
        "price_vs_sma5",
        "stoch_k",
    ],
    "TSLA": [
        "vix_change_5d",
        "bb_position",
        "ret_5d",
        "vol_ratio_21d",
        "price_vs_sma5",
    ],
    "AAPL": [
        "sp500_ret_5d",
        "price_vs_sma5",
        "rsi_7",
        "days_to_options_expiry",
        "usd_ret_5d",
    ],
    "VOO": [
        "sp500_ret_5d",            # VOO IS the S&P, momentum matters most
        "sp500_ret_21d",
        "near_options_expiry",
        "yield_change_5d",
        "vix_change_5d",
    ],
    "VTI": [
        "sp500_ret_5d",
        "sp500_ret_21d",
        "near_options_expiry",
        "vix_change_5d",
    ],
    "BND": [
        "yield_change_5d",         # #1 for BND from EDA
        "yield_curve_10y3m",       # #2 for BND
        "fed_funds_rate",
        "tips_10y",
        "vix",
    ],
    "SCHD": [
        "yield_change_5d",         # dividend ETF sensitive to rates
        "fed_funds_rate",
        "sp500_ret_5d",
        "days_to_options_expiry",
    ],
    "VYM": [
        "yield_change_5d",
        "fed_funds_rate",
        "sp500_ret_5d",
    ],
    "VXUS": [
        "usd_ret_5d",              # USD is critical for international ETFs
        "copper_ret_21d",          # global growth signal
        "sp500_ret_5d",
        "vix_change_5d",
    ],
    "VEA": [
        "usd_ret_5d",
        "copper_ret_21d",
        "sp500_ret_21d",
    ],
    "VWO": [
        "usd_ret_5d",              # EM most sensitive to USD
        "copper_ret_21d",
        "oil_ret_5d",
        "vix_change_5d",
    ],
}

# ── Regime feature (add to all models) ───────────────────────────────────────
# EDA showed regime matters — crash=+1.46%, sideways=-0.14%
REGIME_FEATURE = "vix_regime"


def get_feature_list(ticker: str, df_columns: list[str], max_features: int = 45) -> list[str]:
    """
    Returns ordered feature list for a given ticker.
    Priority: ticker-specific > technical > momentum > calendar > macro > volatility
    Filters to only columns that actually exist in the dataframe.
    """
    existing = set(df_columns) - EXCLUDE_ALWAYS

    ordered = []

    # 1. Always include regime feature
    if REGIME_FEATURE in existing:
        ordered.append(REGIME_FEATURE)

    # 2. Ticker-specific priority features first
    for feat in TICKER_PRIORITY_FEATURES.get(ticker, []):
        if feat in existing and feat not in ordered:
            ordered.append(feat)

    # 3. Then all feature groups in importance order
    group_order = ["technical", "momentum", "calendar", "macro", "volatility", "macro_releases"]
    for group in group_order:
        for feat in FEATURE_GROUPS.get(group, []):
            if feat in existing and feat not in ordered:
                ordered.append(feat)

    # 4. Cap at max_features
    selected = ordered[:max_features]

    log.info(f"  {ticker}: {len(selected)} features selected from {len(existing)} available")
    return selected


def get_regime_label(vix: float) -> str:
    """Map VIX value to regime label for logging/reporting."""
    if vix < 15:   return "calm"
    if vix < 20:   return "normal"
    if vix < 30:   return "elevated"
    return "fear"
