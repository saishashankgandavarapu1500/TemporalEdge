"""
TemporalEdge — Results Aggregator  (Phase 4, Step 4)
Formats Monte Carlo output for the backtest report and Streamlit dashboard.

Two output modes:
  1. CLI report  — printed to terminal after run_phase4.py
  2. API payload — structured dict returned to Streamlit web app
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.config import PORTFOLIO
from src.utils.logger import get_logger

log = get_logger("simulation.results")


def format_cli_report(all_results: dict) -> str:
    """
    Format portfolio-wide Monte Carlo results as a terminal report.
    """
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append("  PHASE 4 — MONTE CARLO SIMULATION REPORT")
    lines.append("=" * 72)

    # ── Per-ticker summary table ──────────────────────────────────────────
    lines.append(f"\n  {'Ticker':<8} {'Tier':<5} {'Win%':>6} {'AvgSave':>9} "
                 f"{'P10':>7} {'P50':>7} {'P90':>7} {'$/yr':>8}")
    lines.append(f"  {'─'*67}")

    total_monthly = 0
    total_annual  = 0

    for ticker in sorted(all_results, key=lambda t: -all_results[t]["summary"]["win_rate_pct"]):
        s = all_results[ticker]["summary"]
        p = s["percentiles"]
        lines.append(
            f"  {ticker:<8} {s['tier']:<5} "
            f"{s['win_rate_pct']:>5.1f}% "
            f"{s['avg_saving_pct']:>+8.3f}% "
            f"{p['p10']:>+6.2f}% "
            f"{p['p50']:>+6.2f}% "
            f"{p['p90']:>+6.2f}% "
            f"${s['avg_dollar_per_year']:>+6.2f}"
        )
        total_monthly += s["monthly_usd"]
        total_annual  += s["avg_dollar_per_year"]

    lines.append(f"  {'─'*67}")
    avg_win = np.mean([r["summary"]["win_rate_pct"] for r in all_results.values()])
    lines.append(f"  {'PORTFOLIO':<8} {'':5} {avg_win:>5.1f}% "
                 f"{'':>9} {'':>7} {'':>7} {'':>7} ${total_annual:>+6.2f}/yr")

    # ── Regime breakdown ──────────────────────────────────────────────────
    lines.append(f"\n  {'─'*72}")
    lines.append("  REGIME BREAKDOWN — When does the model add most value?")
    lines.append(f"  {'─'*72}")
    lines.append(f"  {'Regime':<12} {'Freq%':>6} {'Win%':>7} {'AvgSave':>9} {'Interpretation'}")
    lines.append(f"  {'─'*72}")

    # Aggregate regime stats across all tickers
    regime_agg = {}
    for ticker, result in all_results.items():
        for regime, stats in result["summary"]["regime_stats"].items():
            if regime not in regime_agg:
                regime_agg[regime] = {"wins": [], "saves": [], "freqs": []}
            regime_agg[regime]["wins"].append(stats["win_rate"])
            regime_agg[regime]["saves"].append(stats["avg_saving"])
            regime_agg[regime]["freqs"].append(stats["pct_of_runs"])

    regime_order   = ["fear", "elevated", "normal", "calm"]
    interpretations = {
        "fear":     "VIX>30 — highest value, big swings to exploit",
        "elevated": "VIX 20-30 — model earns its keep",
        "normal":   "VIX 15-20 — moderate benefit",
        "calm":     "VIX<15 — day 27 is nearly optimal, skip timing",
    }
    for regime in regime_order:
        if regime not in regime_agg:
            continue
        agg = regime_agg[regime]
        freq = np.mean(agg["freqs"])
        win  = np.mean(agg["wins"])
        save = np.mean(agg["saves"])
        lines.append(
            f"  {regime:<12} {freq:>5.1f}% {win:>6.1f}% "
            f"{save:>+8.3f}% {interpretations[regime]}"
        )

    # ── 3-scenario compounding table ──────────────────────────────────────
    lines.append(f"\n  {'─'*72}")
    lines.append("  COMPOUNDING IMPACT — 3 SCENARIOS ($"
                 f"{int(total_monthly)}/month total portfolio)")
    lines.append(f"  {'─'*72}")

    # Use the portfolio-blended avg_saving
    avg_saving_portfolio = np.mean([
        r["summary"]["avg_saving_pct"] for r in all_results.values()
    ])
    base_rate = 0.15

    scenarios = [
        ("Conservative", 0.10, 0.25, "bear decade + VIX<20 most months"),
        ("Base case",    0.15, 0.45, "normal market + mixed VIX"),
        ("Optimistic",   0.15, 1.00, "15% returns + model fires every month"),
    ]

    lines.append(f"\n  {'':>7} {'Conservative':>14} {'Base Case':>14} {'Optimistic':>14}")
    lines.append(f"  {'─'*55}")

    for years in [1, 3, 5, 10, 20]:
        row = f"  {years:>2} yr:"
        for _, base, capture, _ in scenarios:
            opt = base + (avg_saving_portfolio / 100 * capture * 12 * 0.3)
            n = years * 12
            r_b = base / 12
            r_o = opt / 12
            fv_b = total_monthly * ((1 + r_b) ** n - 1) / r_b
            fv_o = total_monthly * ((1 + r_o) ** n - 1) / r_o
            row += f"    +${fv_o - fv_b:>8,.0f}"
        lines.append(row)

    lines.append(f"\n  eff. rates:")
    rate_row = "  "
    for _, base, capture, _ in scenarios:
        opt = base + (avg_saving_portfolio / 100 * capture * 12 * 0.3)
        rate_row += f"       {opt*100:>5.1f}% p.a.   "
    lines.append(rate_row)

    # ── Confidence validation ─────────────────────────────────────────────
    lines.append(f"\n  {'─'*72}")
    lines.append("  CONFIDENCE CALIBRATION — Does model confidence predict outcomes?")
    lines.append(f"  {'─'*72}")
    lines.append(f"  {'Confidence':<12} {'Win%':>7} {'AvgSave':>9} {'Interpretation'}")
    lines.append(f"  {'─'*72}")

    conf_agg = {}
    for ticker, result in all_results.items():
        for level, stats in result["summary"]["confidence_stats"].items():
            if level not in conf_agg:
                conf_agg[level] = {"wins": [], "saves": []}
            conf_agg[level]["wins"].append(stats["win_rate"])
            conf_agg[level]["saves"].append(stats["avg_saving"])

    conf_interp = {
        "low":    "Don't override day 27 — signal weak",
        "medium": "Consider shifting 2-3 days earlier",
        "high":   "Act on recommendation — strong signal",
    }
    for level in ["low", "medium", "high"]:
        if level not in conf_agg:
            continue
        win  = np.mean(conf_agg[level]["wins"])
        save = np.mean(conf_agg[level]["saves"])
        lines.append(
            f"  {level:<12} {win:>6.1f}% {save:>+8.3f}% {conf_interp[level]}"
        )

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def format_api_payload(result: dict, monthly_usd: float) -> dict:
    """
    Format single-ticker result as structured payload for Streamlit web app.
    This is what the calculator UI receives and renders.
    """
    s = result["summary"]
    p = s["percentiles"]
    df = result["outcomes"]

    # Monthly saving distribution for the fan chart
    savings = df["saving_pct"].tolist()

    return {
        # Core numbers
        "ticker":           s["ticker"],
        "monthly_usd":      monthly_usd,
        "n_runs":           s["n_runs"],
        "tier":             s["tier"],

        # Performance
        "win_rate_pct":     s["win_rate_pct"],
        "avg_saving_pct":   s["avg_saving_pct"],
        "median_saving_pct":s["median_saving_pct"],

        # Distribution (for histogram / fan chart)
        "percentiles":      p,
        "savings_distribution": savings[:200],  # sample for chart

        # Dollar impact
        "avg_dollar_per_month": s["avg_dollar_per_month"],
        "avg_dollar_per_year":  s["avg_dollar_per_year"],

        # Compounding projections (base case only for calculator)
        "projections":      s["projections"],

        # Regime breakdown (for breakdown chart)
        "regime_stats":     s["regime_stats"],

        # Confidence calibration
        "confidence_stats": s["confidence_stats"],

        # Headline for display
        "headline": (
            f"In {s['win_rate_pct']:.0f}% of simulated months, "
            f"timing beat day 27 by an average of {s['avg_saving_pct']:+.2f}%"
        ),

        # Honest caveat based on tier
        "caveat": {
            "A": "High confidence — model reliably pinpoints the optimal day",
            "B": "Moderate confidence — recommendation accurate within ±2 days",
            "C": "Low confidence — model identifies the right week, not the exact day",
        }.get(s["tier"], ""),
    }
