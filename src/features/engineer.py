"""
TemporalEdge — Feature Engineering  (Phase 1, Step 2)
Takes raw OHLCV + macro data → builds the full feature store.
Output: one Parquet per ticker in data/features/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

from src.config import (
    PORTFOLIO, MACRO_TICKERS, YOUR_CURRENT_DAY,
    PURCHASE_WINDOW_START, ROLLING_WINDOWS, FORWARD_WINDOWS,
    DATA_RAW, DATA_MACRO, DATA_FEAT
)
from src.utils.logger import get_logger

log = get_logger("data.features")


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.warning(f"Could not load {path.name}: {e}")
    return None


def load_macro_panel() -> pd.DataFrame:
    """Merge all macro price series + FRED + calendar into one wide DataFrame."""
    frames = {}

    # Yahoo macro tickers
    for yf_ticker, col_name in MACRO_TICKERS.items():
        safe = yf_ticker.replace("^","").replace("=","_").replace("-","_")
        path = DATA_MACRO / f"{safe}.parquet"
        df = _load_parquet(path)
        if df is not None and "close" in df.columns:
            frames[col_name] = df["close"].rename(col_name)

    # Merge into one DataFrame
    if not frames:
        log.warning("No macro data found — run collector first")
        return pd.DataFrame()

    macro = pd.DataFrame(frames)
    macro.index.name = "date"

    # Derived macro features
    if "us10y_yield" in macro and "us3m_yield" in macro:
        macro["yield_curve_10y3m"] = macro["us10y_yield"] - macro["us3m_yield"]
    if "us10y_yield" in macro:
        macro["yield_curve_level"] = macro["us10y_yield"]
        macro["yield_change_5d"]   = macro["us10y_yield"].pct_change(5)
    if "vix" in macro:
        macro["vix_change_5d"]     = macro["vix"].pct_change(5)
        macro["vix_change_21d"]    = macro["vix"].pct_change(21)
        macro["vix_regime"]        = pd.cut(
            macro["vix"],
            bins=[0, 15, 20, 30, 100],
            labels=[0, 1, 2, 3]         # 0=calm 1=normal 2=elevated 3=fear
        ).astype(float)
    if "sp500" in macro:
        macro["sp500_ret_1d"]      = macro["sp500"].pct_change(1)
        macro["sp500_ret_5d"]      = macro["sp500"].pct_change(5)
        macro["sp500_ret_21d"]     = macro["sp500"].pct_change(21)
        macro["sp500_vol_21d"]     = macro["sp500_ret_1d"].rolling(21).std()
    if "gold" in macro:
        macro["gold_ret_5d"]       = macro["gold"].pct_change(5)
        macro["gold_ret_21d"]      = macro["gold"].pct_change(21)
    if "oil_wti" in macro:
        macro["oil_ret_5d"]        = macro["oil_wti"].pct_change(5)
        macro["oil_ret_21d"]       = macro["oil_wti"].pct_change(21)
    if "copper" in macro:
        macro["copper_ret_21d"]    = macro["copper"].pct_change(21)  # Dr. Copper
    if "usd_index" in macro:
        macro["usd_ret_5d"]        = macro["usd_index"].pct_change(5)
        macro["usd_ret_21d"]       = macro["usd_index"].pct_change(21)
    if "high_yield_etf" in macro and "invest_grade_etf" in macro:
        macro["credit_spread_proxy"] = (
            macro["invest_grade_etf"].pct_change(1) - macro["high_yield_etf"].pct_change(1)
        )

    # Add FRED data if available
    fred_path = DATA_MACRO / "fred_macro.parquet"
    fred = _load_parquet(fred_path)
    if fred is not None:
        # Drop any derived columns that duplicate FRED series names.
        # yield_curve_10y3m is derived above from Yahoo yields AND downloaded
        # directly from FRED as T10Y3M. FRED is the authoritative source — drop
        # the derived version before joining so there is no column name collision.
        fred_col_names = fred.columns.tolist()
        cols_to_drop = [c for c in fred_col_names if c in macro.columns]
        if cols_to_drop:
            log.info(f"  Dropping derived cols superseded by FRED: {cols_to_drop}")
            macro = macro.drop(columns=cols_to_drop)
        macro = macro.join(fred, how="left")
        macro = macro.ffill()

    # Add calendar
    cal_path = DATA_MACRO / "calendar.parquet"
    cal = _load_parquet(cal_path)
    if cal is not None:
        macro = macro.join(cal, how="left")

    log.info(f"  Macro panel: {macro.shape[0]:,} rows × {macro.shape[1]} cols")
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_features(
    ticker: str,
    macro: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame | None:
    """
    Build full feature set for a single ticker.
    Returns DataFrame ready for modelling.
    """
    safe = ticker.replace("^","").replace("=","_").replace("-","_")
    path = DATA_RAW / f"{safe}.parquet"
    raw = _load_parquet(path)

    if raw is None:
        log.warning(f"  ⚠  {ticker}: raw data not found")
        return None

    df = raw.copy()

    # ── Price returns ────────────────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"ret_{w}d"]      = df["close"].pct_change(w)
        df[f"vol_{w}d"]      = df["close"].pct_change(1).rolling(w).std()
        df[f"sma_{w}"]       = df["close"].rolling(w).mean()
        df[f"vol_ratio_{w}d"]= (
            df["volume"] / (df["volume"].rolling(w).mean() + 1e-9)
            if "volume" in df.columns else np.nan
        )

    # Price vs moving averages
    for w in ROLLING_WINDOWS:
        df[f"price_vs_sma{w}"] = (df["close"] - df[f"sma_{w}"]) / (df[f"sma_{w}"] + 1e-9)

    # High-low range (intra-period volatility)
    if "high" in df.columns and "low" in df.columns:
        df["daily_range"]    = (df["high"] - df["low"]) / df["close"]
        df["range_21d_avg"]  = df["daily_range"].rolling(21).mean()

    # ── Technical indicators (via ta library) ────────────────────────────────
    if HAS_TA:
        # RSI
        df["rsi_14"]         = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["rsi_7"]          = ta.momentum.RSIIndicator(df["close"], 7).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"]           = macd.macd()
        df["macd_signal"]    = macd.macd_signal()
        df["macd_hist"]      = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], 20)
        df["bb_upper"]       = bb.bollinger_hband()
        df["bb_lower"]       = bb.bollinger_lband()
        df["bb_width"]       = (df["bb_upper"] - df["bb_lower"]) / (df["close"] + 1e-9)
        df["bb_position"]    = (
            (df["close"] - df["bb_lower"]) /
            (df["bb_upper"] - df["bb_lower"] + 1e-9)
        ).clip(0, 1)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"]        = stoch.stoch()
        df["stoch_d"]        = stoch.stoch_signal()

        # ATR (average true range — volatility)
        if "high" in df.columns and "low" in df.columns:
            df["atr_14"]     = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], 14
            ).average_true_range()
            df["atr_pct"]    = df["atr_14"] / df["close"]
    else:
        log.warning("  ta library not found — skipping technical indicators")
        log.warning("  Install with: pip install ta")

    # ── Calendar features (from macro panel) ─────────────────────────────────
    if not macro.empty:
        calendar_cols = [
            c for c in macro.columns if any(c.startswith(p) for p in [
                "day_of", "week_of", "month", "quarter", "year",
                "is_", "days_to", "days_from", "near_", "presidential"
            ])
        ]
        df = df.join(macro[calendar_cols], how="left")

    # ── Macro features ────────────────────────────────────────────────────────
    if not macro.empty:
        macro_feat_cols = [
            c for c in macro.columns if c not in calendar_cols
            if c in macro.columns
        ]
        df = df.join(macro[macro_feat_cols], how="left")

    # ── Target variables ──────────────────────────────────────────────────────
    for w in FORWARD_WINDOWS:
        # Forward return: how much does price change w days AFTER buying today
        df[f"fwd_ret_{w}d"]  = df["close"].pct_change(w).shift(-w)

    # Is this day the lowest-price day in the purchase window this month?
    # Purchase window = day >= PURCHASE_WINDOW_START
    window_mask = df.index.day >= PURCHASE_WINDOW_START
    df["ym"] = df.index.to_period("M")

    df["is_optimal_buy_day"] = 0
    for period, group in df[window_mask].groupby("ym"):
        if len(group) == 0:
            continue
        min_idx = group["close"].idxmin()
        df.loc[min_idx, "is_optimal_buy_day"] = 1

    df = df.drop(columns=["ym"])

    # Saving vs buying on your_current_day (27th)
    def get_day27_price(grp):
        day27 = grp[grp.index.day == YOUR_CURRENT_DAY]
        if len(day27) == 0:
            # Closest day if 27th is a weekend/holiday
            candidates = grp[grp.index.day >= 25]
            day27 = candidates.iloc[[0]] if len(candidates) > 0 else grp.iloc[[-1]]
        return day27["close"].values[0] if len(day27) > 0 else np.nan

    monthly = df.groupby(df.index.to_period("M"))
    day27_prices = monthly.apply(get_day27_price)
    day27_map = {day: price for period, price in day27_prices.items()
                 for day in df.index if day.to_period("M") == period}
    df["day27_price"] = df.index.map(day27_map.get)
    df["vs_day27_pct"] = (df["day27_price"] - df["close"]) / (df["day27_price"] + 1e-9)

    # Clean up
    df = df.drop(columns=["ticker"], errors="ignore")
    df.index.name = "date"

    if save:
        out_path = DATA_FEAT / f"{ticker}.parquet"
        df.to_parquet(out_path)
        log.info(
            f"  ✓ {ticker:6s} features: "
            f"{df.shape[0]:,} rows × {df.shape[1]} cols → {out_path.name}"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_all_features() -> dict[str, pd.DataFrame]:
    """Build feature store for all portfolio tickers."""
    log.info("\n" + "=" * 60)
    log.info("  TEMPORALEDGE — PHASE 1: FEATURE ENGINEERING")
    log.info("=" * 60)

    log.info("\nLoading macro panel...")
    macro = load_macro_panel()

    results = {}
    log.info(f"\nBuilding features for {len(PORTFOLIO)} portfolio tickers...")
    for ticker in PORTFOLIO:
        df = build_ticker_features(ticker, macro, save=True)
        if df is not None:
            results[ticker] = df

    log.info(f"\n{'=' * 60}")
    log.info(f"  Feature store complete: {len(results)}/{len(PORTFOLIO)} tickers")
    log.info(f"  Output: {DATA_FEAT}")

    # Quick data quality report
    for t, df in results.items():
        optimal_pct = df["is_optimal_buy_day"].mean() * 100
        avg_saving  = df["vs_day27_pct"].mean() * 100
        log.info(
            f"  {t:6s} | optimal days: {optimal_pct:.1f}% of window | "
            f"avg saving vs day27: {avg_saving:+.2f}%"
        )

    log.info("=" * 60)
    return results


if __name__ == "__main__":
    build_all_features()
