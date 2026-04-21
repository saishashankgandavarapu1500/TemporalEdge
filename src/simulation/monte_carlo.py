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


def load_model(ticker: str) -> dict | None:
    """Load saved model bundle from Phase 3."""
    path = MODELS_DIR / f"{ticker}_model.pkl"
    if not path.exists():
        log.warning(f"  Model not found: {path}. Run Phase 3 first.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


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
    Accepts any ticker (not just portfolio tickers).
    Falls back to nearest sector model if ticker not in portfolio.
    """
    # If ticker not in portfolio, use closest sector proxy
    if ticker not in PORTFOLIO:
        log.info(f"  {ticker} not in portfolio — finding closest sector proxy")
        ticker = _find_sector_proxy(ticker)

    monthly_usd = float(monthly_usd)
    return run_simulation(ticker, monthly_usd, n_runs=n_runs, seed=seed)


def _find_sector_proxy(ticker: str) -> str:
    """
    For arbitrary tickers entered in the web app,
    find the closest portfolio ticker by sector/type.
    Simple heuristic — proper version will use yfinance sector info.
    """
    # Default proxies by common sector patterns
    proxies = {
        "tech":    "NVDA",   # tech stocks
        "etf":     "VOO",    # broad ETFs
        "bond":    "BND",    # bond ETFs
        "intl":    "VXUS",   # international
        "div":     "SCHD",   # dividend
    }
    ticker_upper = ticker.upper()

    # Simple rules
    if any(x in ticker_upper for x in ["BOND","TLT","IEF","AGG"]):
        return "BND"
    if any(x in ticker_upper for x in ["EEM","VWO","EWZ"]):
        return "VWO"
    if any(x in ticker_upper for x in ["EFA","VEA","IEFA"]):
        return "VEA"

    # Default to VOO for unknown tickers
    log.info(f"  No proxy found for {ticker}, using VOO")
    return "VOO"
