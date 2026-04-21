"""
TemporalEdge — Data Collector  (Phase 1, Step 1)
Downloads all portfolio + macro tickers from Yahoo Finance in one call.
Saves raw Parquet files partitioned by ticker.
Run this once, then it only refreshes new data on subsequent runs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    PORTFOLIO, MACRO_TICKERS, START_DATE,
    DATA_RAW, DATA_MACRO
)
from src.utils.logger import get_logger

log = get_logger("data.collector")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parquet_path(directory: Path, ticker: str) -> Path:
    """Safe filename: replace ^ and = for filesystem compatibility."""
    safe = ticker.replace("^", "").replace("=", "_").replace("-", "_")
    return directory / f"{safe}.parquet"


def _needs_refresh(path: Path, max_age_hours: int = 12) -> bool:
    """Returns True if file doesn't exist or is older than max_age_hours."""
    if not path.exists():
        return True
    age = datetime.now().timestamp() - path.stat().st_mtime
    return age > max_age_hours * 3600


def _load_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            log.warning(f"Could not read cache {path.name}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main download
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(
    tickers: list[str],
    start: str = START_DATE,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Bulk-download OHLCV for all tickers.
    Returns dict: {ticker: DataFrame with columns [open, high, low, close, volume]}
    Caches each ticker as its own Parquet file.
    """
    to_download, cached_results = [], {}

    for t in tickers:
        dest = DATA_RAW if t in PORTFOLIO else DATA_MACRO
        path = _parquet_path(dest, t)

        if not force_refresh and not _needs_refresh(path):
            df = _load_cached(path)
            if df is not None:
                cached_results[t] = df
                continue
        to_download.append(t)

    if cached_results:
        log.info(f"  Loaded {len(cached_results)} tickers from cache")

    if not to_download:
        log.info("  All tickers up-to-date. Skipping download.")
        return cached_results

    log.info(f"  Downloading {len(to_download)} tickers from Yahoo Finance...")
    log.info(f"  Tickers: {to_download}")

    try:
        raw = yf.download(
            tickers=to_download,
            start=start,
            auto_adjust=True,
            progress=True,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        log.error(f"Download failed: {e}")
        raise

    results = dict(cached_results)

    # yfinance returns different structure for 1 vs multiple tickers
    if len(to_download) == 1:
        t = to_download[0]
        df = _clean_ohlcv(raw, t)
        if df is not None:
            dest = DATA_RAW if t in PORTFOLIO else DATA_MACRO
            df.to_parquet(_parquet_path(dest, t))
            results[t] = df
    else:
        for t in to_download:
            try:
                ticker_raw = raw[t] if t in raw.columns.get_level_values(0) else None
                df = _clean_ohlcv(ticker_raw, t)
                if df is not None:
                    dest = DATA_RAW if t in PORTFOLIO else DATA_MACRO
                    df.to_parquet(_parquet_path(dest, t))
                    results[t] = df
            except KeyError:
                log.warning(f"  ⚠  {t}: not found in download response — skipping")
            except Exception as e:
                log.warning(f"  ⚠  {t}: failed ({type(e).__name__}: {e}) — skipping")

    return results


def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Standardise column names, drop NaN rows, add ticker column.

    NOTE: yfinance fills the full date index from successful tickers in a
    multi-ticker batch. A FAILED ticker therefore comes back with real rows
    but ALL-NaN values — df.empty is False, but every Close is NaN.
    The guard must happen AFTER dropna, not before it.
    """
    if df is None:
        log.warning(f"  ⚠  {ticker}: received None — skipping")
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if "close" not in keep:
        log.warning(f"  ⚠  {ticker}: no close column found — skipping")
        return None

    # Ensure float before dropna so non-numeric strings become NaN too
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[keep].dropna(subset=["close"])

    # THIS is the correct place to check — AFTER dropna.
    # A failed ticker in a multi-ticker batch arrives with real rows
    # but all-NaN values. dropna wipes them, leaving 0 rows here.
    if len(df) == 0:
        log.warning(f"  ⚠  {ticker}: all rows NaN after cleaning (download likely failed) — skipping")
        return None

    df["ticker"] = ticker

    log.info(
        f"  ✓ {ticker:12s} "
        f"{len(df):6,} rows | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FRED macro data (rates, inflation, PMI etc.)
# ─────────────────────────────────────────────────────────────────────────────

FRED_SERIES = {
    # Inflation
    "CPIAUCSL":   "cpi_yoy",            # CPI all items
    "PCEPILFE":   "core_pce",           # Core PCE (Fed's preferred)
    "PPIACO":     "ppi",                # Producer Price Index

    # Growth & Labour
    "GDP":        "gdp_growth",         # Real GDP
    "UNRATE":     "unemployment",       # Unemployment rate
    "ICSA":       "initial_jobless",    # Weekly jobless claims

    # Manufacturing & Consumer
    "UMCSENT":    "consumer_sentiment", # U Michigan Consumer Sentiment
    "RSAFS":      "retail_sales",       # Retail sales
    "INDPRO":     "industrial_prod",    # Industrial production

    # Fed Funds
    "FEDFUNDS":   "fed_funds_rate",     # Effective fed funds rate
    "DFII10":     "tips_10y",           # 10y TIPS (real yield)
    "T10Y2Y":     "yield_curve_10y2y",  # 10Y minus 2Y spread
    "T10Y3M":     "yield_curve_10y3m",  # 10Y minus 3M spread
}


def download_fred_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download macro series from FRED via pandas-datareader.
    Returns wide DataFrame with one column per series, daily frequency (ffill).
    """
    cache_path = DATA_MACRO / "fred_macro.parquet"

    if not force_refresh and not _needs_refresh(cache_path, max_age_hours=24):
        log.info("  FRED data up-to-date, loading from cache")
        return pd.read_parquet(cache_path)

    log.info(f"  Downloading {len(FRED_SERIES)} FRED series...")

    try:
        import pandas_datareader.data as web
    except ImportError:
        log.warning("  pandas-datareader not installed. Run: pip install pandas-datareader")
        log.warning("  Skipping FRED data — continuing without it.")
        return pd.DataFrame()

    frames = {}
    for series_id, col_name in FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", start=START_DATE)
            s.columns = [col_name]
            frames[col_name] = s
            log.info(f"  ✓ FRED {series_id:12s} → {col_name}")
        except Exception as e:
            log.warning(f"  ⚠  FRED {series_id}: {e}")

    if not frames:
        log.warning("  No FRED data downloaded")
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1)
    macro.index.name = "date"

    # FRED data is monthly/weekly — forward-fill to daily
    daily_idx = pd.date_range(start=macro.index.min(), end=datetime.today(), freq="B")
    macro = macro.reindex(daily_idx).ffill()
    macro.index.name = "date"

    macro.to_parquet(cache_path)
    log.info(f"  ✓ FRED macro saved: {macro.shape[0]:,} rows × {macro.shape[1]} cols")
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# Calendar events (FOMC, options expiry, earnings seasons)
# ─────────────────────────────────────────────────────────────────────────────

def build_calendar_features(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build calendar-based features for every trading day.
    No external API needed — purely computed.
    """
    df = pd.DataFrame(index=date_index)

    df["day_of_month"]      = date_index.day
    df["day_of_week"]       = date_index.dayofweek          # 0=Mon 4=Fri
    df["week_of_month"]     = (date_index.day - 1) // 7 + 1
    df["month"]             = date_index.month
    df["quarter"]           = date_index.quarter
    df["year"]              = date_index.year
    df["is_month_end"]      = date_index.is_month_end.astype(int)
    df["is_month_start"]    = date_index.is_month_start.astype(int)
    df["days_to_month_end"] = date_index.days_in_month - date_index.day
    df["days_from_month_start"] = date_index.day - 1

    # Options expiry: 3rd Friday of each month
    # Build expiry list with +2 year buffer so end-of-range dates always have a future expiry
    import calendar as cal_mod
    expiry_list = []
    for yr in range(date_index.year.min(), date_index.year.max() + 2):
        for mo in range(1, 13):
            first_day, n_days = cal_mod.monthrange(yr, mo)
            first_friday = (4 - first_day) % 7 + 1
            third_friday = first_friday + 14
            if third_friday <= n_days:
                expiry_list.append(pd.Timestamp(yr, mo, third_friday))
    expiry_list.sort()

    df["is_options_expiry"] = date_index.isin(set(expiry_list)).astype(int)

    # Vectorised days-to-next-expiry using numpy searchsorted
    # Root cause of original bug: .map() on DatetimeIndex returns an Index object
    # which has no .clip() method — only Series and arrays do.
    # Fix: use numpy searchsorted + array arithmetic instead of a per-row lambda.
    expiry_ns = np.array([e.value for e in expiry_list])   # timestamps in nanoseconds
    date_ns   = np.array([d.value for d in date_index])
    idx       = np.searchsorted(expiry_ns, date_ns, side="left")
    idx       = np.clip(idx, 0, len(expiry_ns) - 1)
    days_to   = ((expiry_ns[idx] - date_ns) / 86_400_000_000_000).astype(int)
    days_to   = np.clip(days_to, 0, 30)

    df["days_to_options_expiry"] = days_to
    df["near_options_expiry"]    = (df["days_to_options_expiry"] <= 3).astype(int)

    # Month-end rebalancing pressure window (last 3 trading days)
    df["month_end_rebal_window"]    = (df["days_to_month_end"] <= 3).astype(int)

    # Quarter end (stronger rebalancing)
    df["is_quarter_end"]            = (
        df["is_month_end"] & df["month"].isin([3, 6, 9, 12])
    ).astype(int)

    # January effect window
    df["is_january_effect"]         = (
        (df["month"] == 1) & (df["day_of_month"] <= 10)
    ).astype(int)

    # Tax loss harvesting season
    df["is_tax_loss_season"]        = (
        (df["month"] == 12) & (df["day_of_month"] >= 15)
    ).astype(int)

    # Known FOMC meeting months (they occur ~8 times/year)
    # Approximate: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
    FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 11, 12}
    df["is_fomc_month"]             = df["month"].isin(FOMC_MONTHS).astype(int)

    # NFP Friday: first Friday of month (jobs report)
    df["is_nfp_week"]               = (
        (df["day_of_week"] == 4) & (df["day_of_month"] <= 7)
    ).astype(int)

    # Presidential cycle (year 3 historically strongest)
    df["presidential_cycle_year"]   = ((df["year"] - 2020) % 4) + 1

    log.info(f"  ✓ Calendar features: {df.shape[1]} columns for {len(df):,} days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_collection(force_refresh: bool = False) -> dict:
    """
    Run the full data collection pipeline.
    Returns summary dict with counts and date ranges.
    """
    log.info("=" * 60)
    log.info("  TEMPORALEDGE — PHASE 1: DATA COLLECTION")
    log.info("=" * 60)

    # 1. All tickers at once
    all_tickers = list(PORTFOLIO.keys()) + list(MACRO_TICKERS.keys())
    log.info(f"\n[1/3] Downloading price data ({len(all_tickers)} tickers)...")
    price_data = download_prices(all_tickers, force_refresh=force_refresh)

    # 2. FRED macro
    log.info("\n[2/3] Downloading FRED macro series...")
    fred_data = download_fred_data(force_refresh=force_refresh)

    # 3. Calendar
    log.info("\n[3/3] Building calendar features...")
    # Use the union of all available dates as index
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[df.index for df in price_data.values() if df is not None]))
    )
    calendar_data = build_calendar_features(all_dates)
    calendar_data.to_parquet(DATA_MACRO / "calendar.parquet")

    # ── Summary ──────────────────────────────────────────────────────────────
    portfolio_ok  = [t for t in PORTFOLIO if t in price_data]
    macro_ok      = [t for t in MACRO_TICKERS if t in price_data]
    fred_cols     = fred_data.shape[1] if not fred_data.empty else 0

    log.info("\n" + "=" * 60)
    log.info("  COLLECTION COMPLETE")
    log.info("=" * 60)
    log.info(f"  Portfolio tickers : {len(portfolio_ok)}/{len(PORTFOLIO)}")
    log.info(f"  Macro tickers     : {len(macro_ok)}/{len(MACRO_TICKERS)}")
    log.info(f"  FRED series       : {fred_cols}")
    log.info(f"  Calendar features : {calendar_data.shape[1]}")

    if portfolio_ok:
        dates = [price_data[t].index for t in portfolio_ok]
        earliest = min(d.min() for d in dates).date()
        latest   = max(d.max() for d in dates).date()
        log.info(f"  Date range        : {earliest} → {latest}")

    log.info(f"\n  Data saved to: {DATA_RAW.parent}")
    log.info("=" * 60)

    return {
        "portfolio_tickers": portfolio_ok,
        "macro_tickers":     macro_ok,
        "fred_series":       fred_cols,
        "calendar_cols":     calendar_data.shape[1],
        "price_data":        price_data,
        "fred_data":         fred_data,
        "calendar_data":     calendar_data,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TemporalEdge data collector")
    parser.add_argument("--refresh", action="store_true", help="Force re-download all data")
    args = parser.parse_args()

    result = run_collection(force_refresh=args.refresh)
