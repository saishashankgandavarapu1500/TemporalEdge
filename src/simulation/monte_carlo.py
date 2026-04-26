"""
TemporalEdge — Monte Carlo Engine  (Phase 4, Step 3)

Runs the controlled parallel simulation:
  Same macro environment → Agent A (day 27) vs Agent B (AI-optimised)
  Only the entry date changes between agents.

1,000 runs per ticker, each run = one sampled historical month.
Designed to complete in <5 seconds for Streamlit web app use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import PORTFOLIO, ROOT
from src.simulation.scenario_sampler import sampler
from src.simulation.agents import agent_a, agent_b, compare_agents, TICKER_TIERS
from src.utils.logger import get_logger

log = get_logger("simulation.monte_carlo")

MODELS_DIR = ROOT / "models"
N_RUNS_BACKTEST = 1000   # full simulation
N_RUNS_LIVE     = 1000   # same for web app — fast enough on M2


# VTI MODEL PROXY — Phase 4 finding
# VTI own model: 50.9% win rate (random). Root cause: pre-2010 dot-com/GFC
# training data has different timing patterns than post-2010 regime.
# Fix: use VOO model for VTI — same securities, >0.99 correlation post-2010,
# all macro features are market-wide and transfer directly.
# VTI scenarios still use VTI actual prices (fair comparison preserved).
MODEL_PROXY = {
    "VTI": "VOO",
}


def load_model(ticker: str) -> dict | None:
    """
    Load saved model bundle from Phase 3.
    VTI uses VOO model as proxy (see MODEL_PROXY).
    """
    model_ticker = MODEL_PROXY.get(ticker, ticker)
    if model_ticker != ticker:
        log.info(f"  {ticker}: loading {model_ticker} model as proxy "
                 f"(own model ~random, Phase 4 finding)")

    path = MODELS_DIR / f"{model_ticker}_model.pkl"
    if not path.exists():
        log.warning(f"  Model not found: {path}. Run Phase 3 first.")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    bundle["proxy_used"]      = (model_ticker != ticker)
    bundle["proxy_ticker"]    = model_ticker if model_ticker != ticker else None
    bundle["original_ticker"] = ticker
    return bundle


def run_simulation(
    ticker: str,
    monthly_usd: float,
    n_runs: int = N_RUNS_LIVE,
    regime_filter: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation for one ticker.

    Args:
        ticker:       portfolio ticker (e.g. 'VOO')
        monthly_usd:  monthly investment amount in dollars
        n_runs:       number of simulation runs
        regime_filter: restrict to one VIX regime ('calm','normal','elevated','fear')
        seed:         random seed for reproducibility

    Returns dict with:
        outcomes:     list of per-run comparison results
        summary:      aggregated statistics
        regime_stats: breakdown by VIX regime
    """
    log.info(f"  Running Monte Carlo: {ticker} | {n_runs} runs | ${monthly_usd}/mo")

    # Load model
    model = load_model(ticker)
    if model is None:
        return {"error": f"Model not found for {ticker}"}

    # Sample scenarios
    scenarios = sampler.sample(
        ticker,
        n=n_runs,
        regime_filter=regime_filter,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    # Run both agents on each scenario
    outcomes = []
    for scenario in scenarios:
        try:
            result_a  = agent_a(scenario)
            result_b  = agent_b(scenario, model, rng)
            comparison = compare_agents(result_a, result_b, monthly_usd)
            outcomes.append(comparison)
        except Exception as e:
            log.debug(f"  Run failed ({e}), skipping")
            continue

    if not outcomes:
        return {"error": f"No valid outcomes for {ticker}"}

    df = pd.DataFrame(outcomes)

    # ── Aggregate statistics ───────────────────────────────────────────────
    win_rate    = df["agent_b_wins"].mean() * 100
    avg_saving  = df["saving_pct"].mean()
    med_saving  = df["saving_pct"].median()

    # Percentile distribution of saving%
    p10 = float(np.percentile(df["saving_pct"], 10))
    p25 = float(np.percentile(df["saving_pct"], 25))
    p50 = float(np.percentile(df["saving_pct"], 50))
    p75 = float(np.percentile(df["saving_pct"], 75))
    p90 = float(np.percentile(df["saving_pct"], 90))

    # Dollar impact
    avg_dollar  = df["dollar_advantage"].mean()
    total_dollar_per_year = avg_dollar * 12

    # Compounding projections (base case: 45% capture, 15% return)
    base_rate = 0.15
    opt_rate  = base_rate + (avg_saving / 100 * 0.45 * 12 * 0.3)

    projections = {}
    for years in [1, 3, 5, 10, 20]:
        n = years * 12
        r_b = base_rate / 12
        r_o = opt_rate / 12
        fv_b = monthly_usd * ((1 + r_b) ** n - 1) / r_b
        fv_o = monthly_usd * ((1 + r_o) ** n - 1) / r_o
        projections[years] = {
            "fixed":     round(fv_b, 0),
            "optimised": round(fv_o, 0),
            "extra":     round(fv_o - fv_b, 0),
        }

    # ── Regime breakdown ───────────────────────────────────────────────────
    regime_stats = {}
    for regime in ["calm", "normal", "elevated", "fear"]:
        sub = df[df["vix_regime"] == regime]
        if len(sub) < 5:
            continue
        regime_stats[regime] = {
            "n_runs":    len(sub),
            "win_rate":  round(sub["agent_b_wins"].mean() * 100, 1),
            "avg_saving":round(sub["saving_pct"].mean(), 3),
            "pct_of_runs": round(len(sub) / len(df) * 100, 1),
        }

    # ── Confidence-stratified results ──────────────────────────────────────
    # Does higher model confidence correlate with better outcomes?
    conf_stats = {}
    for label, lo, hi in [("low", 0, 0.33), ("medium", 0.33, 0.67), ("high", 0.67, 1.0)]:
        sub = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(sub) < 5:
            continue
        conf_stats[label] = {
            "n_runs":    len(sub),
            "win_rate":  round(sub["agent_b_wins"].mean() * 100, 1),
            "avg_saving":round(sub["saving_pct"].mean(), 3),
        }

    summary = {
        "ticker":               ticker,
        "monthly_usd":          monthly_usd,
        "n_runs":               len(df),
        "tier":                 TICKER_TIERS.get(ticker, "B"),
        "win_rate_pct":         round(win_rate, 1),
        "avg_saving_pct":       round(avg_saving, 3),
        "median_saving_pct":    round(med_saving, 3),
        "percentiles": {
            "p10": round(p10, 3), "p25": round(p25, 3),
            "p50": round(p50, 3), "p75": round(p75, 3),
            "p90": round(p90, 3),
        },
        "avg_dollar_per_month": round(avg_dollar, 4),
        "avg_dollar_per_year":  round(total_dollar_per_year, 2),
        "opt_rate_pct":         round(opt_rate * 100, 2),
        "projections":          projections,
        "regime_stats":         regime_stats,
        "confidence_stats":     conf_stats,
    }

    log.info(
        f"  ✓ {ticker}: win={win_rate:.1f}% | "
        f"avg_save={avg_saving:+.3f}% | "
        f"p10={p10:+.2f}% p50={p50:+.2f}% p90={p90:+.2f}%"
    )

    return {
        "summary":  summary,
        "outcomes": df,
    }


def run_portfolio_simulation(
    n_runs: int = N_RUNS_BACKTEST,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation for all 11 portfolio tickers.
    Uses each ticker's actual monthly_usd from config.
    """
    log.info("=" * 60)
    log.info("  TEMPORALEDGE — PHASE 4: PORTFOLIO MONTE CARLO")
    log.info(f"  {n_runs} runs per ticker × 11 tickers")
    log.info("=" * 60)

    all_results = {}

    for ticker, info in PORTFOLIO.items():
        monthly_usd = info["monthly_usd"]
        result = run_simulation(ticker, monthly_usd, n_runs=n_runs, seed=seed)
        if "error" not in result:
            all_results[ticker] = result

    return all_results


def run_single_ticker_simulation(
    ticker: str,
    monthly_usd: float,
    n_runs: int = N_RUNS_LIVE,
    seed: int = 42,
) -> dict:
    """
    Entry point for the Streamlit web app.
    Accepts any ticker, not just portfolio tickers.

    For portfolio tickers: uses pre-built feature store directly.
    For unknown tickers:
      1. Downloads real historical data via yfinance
      2. Builds features using the same pipeline
      3. Finds nearest model by beta + sector (NOT by data substitution)
      4. Runs simulation with REAL ticker data + proxy model weights
      5. Tags result with proxy info so UI can be honest about it
    """
    monthly_usd   = float(monthly_usd)
    proxy_model   = None
    proxy_label   = None

    if ticker not in PORTFOLIO:
        log.info(f"  {ticker}: unknown ticker — building features on-the-fly")

        # Step 1: Find model proxy (weights only, not data)
        proxy_model = _find_model_proxy(ticker)
        proxy_label = proxy_model
        log.info(f"  {ticker}: using {proxy_model} model weights as proxy")

        # Step 2: Download + build features for the REAL ticker
        success = _build_features_on_the_fly(ticker)
        if not success:
            log.warning(f"  {ticker}: feature build failed — falling back to {proxy_model} data")
            # Last resort: run proxy data (clearly labeled)
            result = run_simulation(proxy_model, monthly_usd, n_runs=n_runs, seed=seed)
            if "summary" in result:
                result["summary"]["proxy_ticker"]  = proxy_model
                result["summary"]["proxy_reason"]  = "feature_build_failed"
                result["summary"]["is_proxy_data"] = True
                result["warning"] = (
                    f"⚠️  No data available for {ticker}. "
                    f"Showing {proxy_model} results as a rough proxy. "
                    f"These numbers do NOT represent {ticker}."
                )
            return result

        # Step 3: Add ticker to PORTFOLIO-like config temporarily
        # so ScenarioSampler can load its feature file
        _TEMP_PORTFOLIO[ticker] = {
            "name": ticker, "monthly_usd": monthly_usd,
            "type": "unknown", "sector": "unknown",
        }

        # Step 4: Patch MODEL_PROXY so the simulation uses proxy model weights
        _ORIG_PROXY = dict(MODEL_PROXY)
        MODEL_PROXY[ticker] = proxy_model

        try:
            result = run_simulation(ticker, monthly_usd, n_runs=n_runs, seed=seed)
        finally:
            # Clean up temp state
            MODEL_PROXY.clear()
            MODEL_PROXY.update(_ORIG_PROXY)
            if ticker in _TEMP_PORTFOLIO:
                del _TEMP_PORTFOLIO[ticker]

        # Tag result with proxy information
        if "summary" in result:
            result["summary"]["proxy_ticker"]   = proxy_model
            result["summary"]["proxy_reason"]   = "model_weights_only"
            result["summary"]["is_proxy_data"]  = False
            result["summary"]["real_data"]      = True
        return result

    return run_simulation(ticker, monthly_usd, n_runs=n_runs, seed=seed)


# Temp storage for on-the-fly tickers
_TEMP_PORTFOLIO: dict = {}


def _build_features_on_the_fly(ticker: str) -> bool:
    """
    Download historical data for an unknown ticker and build its feature file.
    Returns True if successful, False if insufficient data.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from src.config import DATA_RAW, DATA_FEAT, START_DATE
        from src.features.engineer import build_ticker_features, load_macro_panel

        log.info(f"  {ticker}: downloading historical data...")
        raw = yf.download(ticker, start="2010-01-01",
                          auto_adjust=True, progress=False)

        if raw.empty or len(raw) < 200:
            log.warning(f"  {ticker}: insufficient data ({len(raw)} rows)")
            return False

        # Save raw parquet
        df_raw = raw.copy()
        df_raw.columns = [c.lower() for c in df_raw.columns]
        df_raw.index.name = "date"
        df_raw["ticker"] = ticker
        raw_path = DATA_RAW / f"{ticker}.parquet"
        df_raw.to_parquet(raw_path)

        # Build features using existing macro panel
        log.info(f"  {ticker}: building features ({len(df_raw):,} rows)...")
        macro = load_macro_panel()
        feat_df = build_ticker_features(ticker, macro, save=True)

        if feat_df is None or len(feat_df) < 100:
            log.warning(f"  {ticker}: feature build produced insufficient rows")
            return False

        log.info(f"  {ticker}: features built ✓ ({len(feat_df):,} rows)")
        return True

    except Exception as e:
        log.warning(f"  {ticker}: on-the-fly build failed: {e}")
        return False


def _find_model_proxy(ticker: str) -> str:
    """
    Find the best MODEL PROXY for an unknown ticker.
    Uses yfinance to get real beta and sector, then maps to
    the closest portfolio ticker's MODEL WEIGHTS.

    This is MODEL-only proxy (weights). The simulation always
    uses REAL historical data for the actual ticker.

    Volatility tiers:
      Low vol (beta < 0.7):   BND / SCHD / VYM
      Medium vol (0.7-1.2):   VOO / VTI / AAPL
      High vol (1.2-2.0):     NVDA / VWO / VXUS
      Very high (> 2.0):      TSLA
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).fast_info
        quote_type = getattr(info, "quote_type", "").upper()
        # Get beta as volatility proxy
        full_info  = yf.Ticker(ticker).info
        beta       = full_info.get("beta", 1.0) or 1.0
        sector     = full_info.get("sector", "").lower()
        asset_type = full_info.get("quoteType", "").upper()
    except Exception:
        beta, sector, asset_type = 1.0, "", "EQUITY"

    log.info(f"  {ticker}: beta={beta:.2f} sector={sector} type={asset_type}")

    # Bond ETFs
    if any(x in ticker.upper() for x in ["BOND","TLT","IEF","AGG","BND","LQD","HYG"]):
        return "BND"

    # International / EM ETFs
    if any(x in ticker.upper() for x in ["EEM","EWZ","VWO","FXI","IEMG"]):
        return "VWO"
    if any(x in ticker.upper() for x in ["EFA","VEA","IEFA","EWJ","VGK"]):
        return "VEA"
    if any(x in ticker.upper() for x in ["VXUS","IXUS","CWI"]):
        return "VXUS"

    # Dividend ETFs
    if any(x in ticker.upper() for x in ["DIV","DGRO","VYM","SCHD","HDV"]):
        return "SCHD"

    # Map by volatility + sector
    if beta > 2.5:
        # Extremely volatile — meme stocks, high-beta plays
        log.info(f"  {ticker}: very high beta ({beta:.1f}) → TSLA proxy (Tier C)")
        return "TSLA"
    elif beta > 1.5:
        # High vol tech/growth
        if "technology" in sector or "communication" in sector:
            log.info(f"  {ticker}: high beta tech ({beta:.1f}) → NVDA proxy (Tier B)")
            return "NVDA"
        else:
            log.info(f"  {ticker}: high beta non-tech ({beta:.1f}) → TSLA proxy (Tier C)")
            return "TSLA"
    elif beta > 1.0:
        # Moderate-high vol
        if "technology" in sector or "communication" in sector:
            log.info(f"  {ticker}: moderate-high beta tech ({beta:.1f}) → AAPL proxy (Tier A)")
            return "AAPL"
        elif "financial" in sector:
            log.info(f"  {ticker}: financial ({beta:.1f}) → VOO proxy (Tier A)")
            return "VOO"
        else:
            log.info(f"  {ticker}: moderate-high beta ({beta:.1f}) → NVDA proxy (Tier B)")
            return "NVDA"
    elif beta > 0.6:
        # Moderate vol — broad market-like
        log.info(f"  {ticker}: moderate beta ({beta:.1f}) → VOO proxy (Tier A)")
        return "VOO"
    else:
        # Low vol — bond/dividend-like
        log.info(f"  {ticker}: low beta ({beta:.1f}) → BND proxy (Tier A)")
        return "BND"
