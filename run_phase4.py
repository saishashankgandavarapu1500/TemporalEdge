"""
TemporalEdge — Phase 4 Runner
Runs the full portfolio Monte Carlo simulation and prints results.

Usage:
  python run_phase4.py                    # full portfolio, 1000 runs
  python run_phase4.py --ticker VOO       # single ticker
  python run_phase4.py --ticker VOO --monthly 50   # custom amount
  python run_phase4.py --regime fear      # only fear-regime scenarios
  python run_phase4.py --runs 500         # faster run for testing
"""

import sys
import time
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.config import PORTFOLIO, ROOT
from src.simulation.monte_carlo import (
    run_simulation,
    run_portfolio_simulation,
    load_model,
)
from src.simulation.results import format_cli_report
from src.simulation.scenario_sampler import sampler
from src.utils.logger import get_logger

log = get_logger("pipeline.phase4")

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def print_scenario_stats(ticker: str):
    """Print scenario pool info before running."""
    try:
        stats = sampler.get_scenario_stats(ticker)
        regime = stats["regime_pct"]
        log.info(
            f"  Scenario pool: {stats['n_scenarios']} months | "
            f"{stats['date_range']} | "
            f"VIX median={stats['vix_median']}"
        )
        log.info(
            f"  Regime distribution: "
            f"calm={regime.get('calm',0):.0f}% "
            f"normal={regime.get('normal',0):.0f}% "
            f"elevated={regime.get('elevated',0):.0f}% "
            f"fear={regime.get('fear',0):.0f}%"
        )
    except Exception as e:
        log.debug(f"  Scenario stats unavailable: {e}")


def main():
    parser = argparse.ArgumentParser(description="TemporalEdge Phase 4 — Monte Carlo")
    parser.add_argument("--ticker",  type=str,   default=None,
                        help="Single ticker (e.g. --ticker VOO). Default: all 11.")
    parser.add_argument("--monthly", type=float, default=None,
                        help="Monthly investment $ (overrides portfolio default)")
    parser.add_argument("--runs",    type=int,   default=1000,
                        help="Number of Monte Carlo runs (default: 1000)")
    parser.add_argument("--regime",  type=str,   default=None,
                        choices=["calm", "normal", "elevated", "fear"],
                        help="Restrict to one VIX regime")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    t0 = time.time()

    log.info("\n" + "🎲 " * 20)
    log.info("  TEMPORALEDGE — PHASE 4: MONTE CARLO SIMULATION")
    log.info("  Same macro environment. Same market. Only the date changes.")
    log.info("🎲 " * 20 + "\n")

    if args.ticker:
        # ── Single ticker mode ────────────────────────────────────────────
        ticker = args.ticker.upper()
        if ticker not in PORTFOLIO:
            log.error(f"  {ticker} not in portfolio. Available: {list(PORTFOLIO.keys())}")
            sys.exit(1)

        monthly = args.monthly or PORTFOLIO[ticker]["monthly_usd"]
        log.info(f"  Mode: single ticker — {ticker} | ${monthly}/mo | {args.runs} runs")
        if args.regime:
            log.info(f"  Regime filter: {args.regime} only")
        log.info("")

        print_scenario_stats(ticker)

        result = run_simulation(
            ticker, monthly,
            n_runs=args.runs,
            regime_filter=args.regime,
            seed=args.seed,
        )

        if "error" in result:
            log.error(f"  Simulation failed: {result['error']}")
            sys.exit(1)

        s = result["summary"]
        p = s["percentiles"]

        log.info(f"\n{'=' * 60}")
        log.info(f"  RESULTS: {ticker} | {args.runs} runs")
        log.info(f"{'=' * 60}")
        log.info(f"  Win rate:        {s['win_rate_pct']:.1f}%")
        log.info(f"  Avg saving:      {s['avg_saving_pct']:+.3f}%")
        log.info(f"  Distribution:    p10={p['p10']:+.2f}% | p50={p['p50']:+.2f}% | p90={p['p90']:+.2f}%")
        log.info(f"  Dollar/month:    ${s['avg_dollar_per_month']:+.4f}")
        log.info(f"  Dollar/year:     ${s['avg_dollar_per_year']:+.2f}")
        log.info(f"  Model tier:      {s['tier']} ({{'A':'high precision','B':'medium','C':'low'}}[s['tier']])")

        log.info(f"\n  Regime breakdown:")
        for regime, stats in s["regime_stats"].items():
            log.info(f"    {regime:<10}: win={stats['win_rate']:.1f}% save={stats['avg_saving']:+.3f}% (n={stats['n_runs']})")

        log.info(f"\n  20-year projection (base case, 45% capture):")
        for yr, proj in s["projections"].items():
            log.info(f"    {yr:>2} yr: fixed=${proj['fixed']:>8,.0f} | "
                     f"opt=${proj['optimised']:>8,.0f} | "
                     f"extra=${proj['extra']:>6,.0f}")

        # Save
        out = RESULTS_DIR / f"{ticker}_simulation.json"
        with open(out, "w") as f:
            json.dump({
                "summary":  s,
                "outcomes": result["outcomes"].to_dict(orient="records")
            }, f, indent=2, default=str)
        log.info(f"\n  Results saved: {out}")

    else:
        # ── Full portfolio mode ───────────────────────────────────────────
        log.info(f"  Mode: full portfolio | {args.runs} runs per ticker\n")

        # Pre-build all scenario pools and print stats
        log.info("  Building scenario pools...")
        for ticker in PORTFOLIO:
            print_scenario_stats(ticker)
        log.info("")

        all_results = {}
        for ticker, info in PORTFOLIO.items():
            monthly = info["monthly_usd"]
            log.info(f"\n{'─' * 50}")
            log.info(f"  {ticker} — {info['name']} (${monthly}/mo)")
            result = run_simulation(
                ticker, monthly,
                n_runs=args.runs,
                regime_filter=args.regime,
                seed=args.seed,
            )
            if "error" not in result:
                all_results[ticker] = result

        # Full report
        report = format_cli_report(all_results)
        log.info(report)

        # Save all results
        out = RESULTS_DIR / "portfolio_simulation.json"
        save_data = {}
        for t, r in all_results.items():
            save_data[t] = {
                "summary":  r["summary"],
                "outcomes": r["outcomes"].to_dict(orient="records"),
            }
        with open(out, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        log.info(f"\n  Full results saved: {out}")

    elapsed = time.time() - t0
    log.info(f"\n✅ Phase 4 complete in {elapsed:.1f}s")
    log.info(f"   Ready for Phase 5: Streamlit dashboard\n")


if __name__ == "__main__":
    main()
