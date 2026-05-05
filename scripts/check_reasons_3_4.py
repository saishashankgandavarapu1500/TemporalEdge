"""
TemporalEdge — Reason 3 & 4 Checks
Reason 3: Does the model only work in certain regimes (high VIX)?
Reason 4: Is the Sharpe inflated by low-volatility tickers (BND)?

Run with: python scripts/check_reasons_3_4.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PORTFOLIO, ROOT
from src.utils.logger import get_logger

log = get_logger("validation.reasons_3_4")

MODELS_DIR = ROOT / "models"
DATA_RAW   = ROOT / "data" / "raw"
DATA_MACRO = ROOT / "data" / "macro"


# ─────────────────────────────────────────────────────────────────────────────
# Reason 3 — Regime dependence
# ─────────────────────────────────────────────────────────────────────────────

def check_regime_dependence():
    """
    Break backtest results by VIX regime.
    Does the model only work in fear/elevated regimes?
    If win rate drops to ~50% in calm regimes → regime-dependent.
    If win rate stays >60% across all regimes → robust.
    """
    log.info("\n" + "=" * 60)
    log.info("  REASON 3 — REGIME DEPENDENCE CHECK")
    log.info("  Does the model only work in high-VIX environments?")
    log.info("=" * 60)

    # Load VIX data
    vix_path = DATA_MACRO / "VIX.parquet"
    if not vix_path.exists():
        # Try raw folder
        vix_path = DATA_RAW / "VIX.parquet"
    if not vix_path.exists():
        # Try with ^ prefix removed
        for p in DATA_RAW.glob("*VIX*"):
            vix_path = p
            break

    try:
        vix = pd.read_parquet(vix_path)
        vix.index = pd.to_datetime(vix.index)
        vix_col = "close" if "close" in vix.columns else vix.columns[0]
        vix_monthly = vix[vix_col].resample("M").mean()
        has_vix = True
        log.info(f"  VIX data loaded: {vix_path.name}")
    except Exception as e:
        log.warning(f"  VIX data not found ({e}) — using period-based proxy")
        has_vix = False

    def get_regime(vix_val):
        if vix_val < 15:   return "calm"
        if vix_val < 20:   return "normal"
        if vix_val < 30:   return "elevated"
        return "fear"

    all_regime_results = {}

    for ticker in PORTFOLIO:
        csv_path = MODELS_DIR / f"{ticker}_backtest.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["period_dt"] = pd.to_datetime(df["period"].astype(str))

        if has_vix:
            # Map VIX monthly avg to each backtest period
            df["vix_avg"] = df["period_dt"].apply(
                lambda d: vix_monthly.get(
                    vix_monthly.index[vix_monthly.index <= d][-1]
                    if any(vix_monthly.index <= d) else vix_monthly.index[0]
                , np.nan)
            )
            df["regime"] = df["vix_avg"].apply(
                lambda v: get_regime(v) if pd.notna(v) else "unknown"
            )
        else:
            # Fallback: use year as proxy for regime era
            df["regime"] = df["period_dt"].dt.year.apply(
                lambda y: "fear" if y in [2008,2009,2020,2022]
                else "elevated" if y in [2010,2011,2015,2018,2023]
                else "calm"
            )

        regime_stats = {}
        for regime in ["calm", "normal", "elevated", "fear"]:
            subset = df[df["regime"] == regime]
            if len(subset) < 5:
                continue
            wr = subset["model_beat_27"].mean() * 100
            avg_save = subset["model_saving_vs_27"].mean()
            n = len(subset)
            # Binomial test for this regime
            n_wins = subset["model_beat_27"].sum()
            p_val = stats.binomtest(n_wins, n, p=0.5, alternative="greater").pvalue
            regime_stats[regime] = {
                "n": n,
                "win_rate": round(wr, 1),
                "avg_saving": round(avg_save, 4),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
            }

        all_regime_results[ticker] = regime_stats

        log.info(f"\n  {ticker}:")
        log.info(f"  {'Regime':<12} {'N':>5} {'Win%':>7} {'AvgSave%':>10} {'p-value':>10} {'Sig?':>6}")
        log.info(f"  {'─'*50}")
        for regime in ["calm", "normal", "elevated", "fear"]:
            if regime not in regime_stats:
                log.info(f"  {regime:<12} {'<5 months — skip':>35}")
                continue
            r = regime_stats[regime]
            sig = "✓" if r["significant"] else "✗"
            log.info(
                f"  {regime:<12} {r['n']:>5} {r['win_rate']:>6.1f}% "
                f"{r['avg_saving']:>+9.4f}% {r['p_value']:>10.4f} {sig:>6}"
            )

    # Summary — does calm regime kill the edge?
    log.info(f"\n  {'─'*60}")
    log.info("  VERDICT:")
    calm_rates = [
        r["calm"]["win_rate"]
        for r in all_regime_results.values()
        if "calm" in r
    ]
    fear_rates = [
        r["fear"]["win_rate"]
        for r in all_regime_results.values()
        if "fear" in r
    ]

    if calm_rates:
        avg_calm = np.mean(calm_rates)
        avg_fear = np.mean(fear_rates) if fear_rates else None
        log.info(f"  Avg win rate in CALM regime:    {avg_calm:.1f}%")
        if avg_fear:
            log.info(f"  Avg win rate in FEAR regime:    {avg_fear:.1f}%")
            gap = avg_fear - avg_calm
            log.info(f"  Gap (fear - calm):              {gap:+.1f}%")

        if avg_calm >= 65:
            log.info("  → Model works across ALL regimes — not regime-dependent ✓")
        elif avg_calm >= 55:
            log.info("  → Model has reduced edge in calm regimes — partially regime-dependent")
            log.info("    (Still beats random, but weaker signal when VIX is low)")
        else:
            log.info("  → Model is REGIME-DEPENDENT — edge largely disappears in calm markets ✗")

    return all_regime_results


# ─────────────────────────────────────────────────────────────────────────────
# Reason 4 — Sharpe inflation from low volatility
# ─────────────────────────────────────────────────────────────────────────────

def check_sharpe_inflation():
    """
    Check if high Sharpe numbers are economically meaningful
    or just a mathematical artifact of low-volatility tickers.

    For each ticker, compute:
    1. Monthly saving in dollar terms (not %)
    2. Annualised dollar edge on $100/mo investment
    3. Whether the edge survives a 0.1% transaction cost assumption
    4. Economic significance vs mathematical significance
    """
    log.info("\n" + "=" * 60)
    log.info("  REASON 4 — SHARPE INFLATION CHECK")
    log.info("  Is the Sharpe economically meaningful or just math?")
    log.info("=" * 60)

    MONTHLY_INVEST = 100  # $100/month for comparison

    log.info(f"\n  Assuming ${MONTHLY_INVEST}/mo investment:")
    log.info(f"  {'Ticker':<8} {'AvgSave%':>9} {'$/mo':>7} {'$/yr':>8} "
             f"{'After0.1%cost':>14} {'Sharpe':>8} {'Economic?':>10}")
    log.info(f"  {'─'*70}")

    results = {}

    for ticker in PORTFOLIO:
        csv_path = MODELS_DIR / f"{ticker}_backtest.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        savings_pct = df["model_saving_vs_27"].values

        avg_pct    = savings_pct.mean()
        std_pct    = savings_pct.std(ddof=1)
        sharpe     = (avg_pct / std_pct) * np.sqrt(12) if std_pct > 0 else 0

        # Dollar terms
        dollar_per_month = avg_pct / 100 * MONTHLY_INVEST
        dollar_per_year  = dollar_per_month * 12

        # After transaction cost (0.1% round trip — generous for ETFs)
        transaction_cost_pct = 0.10  # % of investment
        net_saving_pct = avg_pct - transaction_cost_pct
        net_dollar_per_month = net_saving_pct / 100 * MONTHLY_INVEST
        economic = net_saving_pct > 0

        # Compounding: what does this edge compound to over 20 years?
        # If you save avg_pct more per month and it compounds at 8% annually
        annual_edge_compounded_20yr = dollar_per_year * (
            ((1.08**20) - 1) / 0.08
        )

        results[ticker] = {
            "avg_saving_pct":         round(avg_pct, 4),
            "sharpe":                 round(sharpe, 3),
            "dollar_per_month":       round(dollar_per_month, 3),
            "dollar_per_year":        round(dollar_per_year, 2),
            "net_after_cost_pct":     round(net_saving_pct, 4),
            "survives_costs":         economic,
            "20yr_compounded_value":  round(annual_edge_compounded_20yr, 0),
        }

        econ_label = "✓ YES" if economic else "✗ NO"
        log.info(
            f"  {ticker:<8} {avg_pct:>+8.3f}% "
            f"${dollar_per_month:>6.3f} "
            f"${dollar_per_year:>7.2f} "
            f"{'✓' if economic else '✗'} {net_saving_pct:>+.3f}%  "
            f"{sharpe:>8.3f} {econ_label:>10}"
        )

    # BND special case explanation
    log.info(f"\n  {'─'*60}")
    log.info("  BND DEEP DIVE (the Sharpe concern):")
    if "BND" in results:
        bnd = results["BND"]
        log.info(f"  BND avg saving:  {bnd['avg_saving_pct']:+.4f}% per month")
        log.info(f"  BND Sharpe:      {bnd['sharpe']:.3f}")
        log.info(f"  BND $ per month: ${bnd['dollar_per_month']:.3f} on $100 invested")
        log.info(f"  BND $ per year:  ${bnd['dollar_per_year']:.2f} on $100/mo")
        log.info(f"  After 0.1% cost: {'survives' if bnd['survives_costs'] else 'does NOT survive'}")
        log.info(f"\n  Why BND Sharpe is high:")
        log.info(f"  Bond ETF has tiny price moves → tiny std of savings → high Sharpe")
        log.info(f"  This is a mathematical artifact — the ECONOMIC value is small")
        log.info(f"  ${bnd['dollar_per_year']:.2f}/yr on $100/mo is real but modest")
        log.info(f"  20yr compounded value of the edge: ${bnd['20yr_compounded_value']:,.0f}")
        log.info(f"\n  HONEST ANSWER FOR INTERVIEW:")
        log.info(f"  'BND's high Sharpe reflects low volatility, not high economic value.")
        log.info(f"   The actual edge is {bnd['avg_saving_pct']:+.2f}% per month — about")
        log.info(f"   ${bnd['dollar_per_month']:.2f}/mo on a $100 investment. The Sharpe")
        log.info(f"   metric rewards consistency, which BND has. But I'm transparent")
        log.info(f"   that for low-volatility assets the economic magnitude is small.'")

    # Overall verdict
    log.info(f"\n  {'─'*60}")
    log.info("  OVERALL VERDICT ON SHARPE INFLATION:")
    surviving = [t for t, r in results.items() if r["survives_costs"]]
    not_surviving = [t for t, r in results.items() if not r["survives_costs"]]

    log.info(f"  Survive 0.1% transaction costs: {surviving}")
    if not_surviving:
        log.info(f"  Do NOT survive costs:           {not_surviving}")

    high_econ = [t for t, r in results.items()
                 if r["dollar_per_year"] > 10]  # >$10/yr on $100/mo
    log.info(f"  High economic value (>$10/yr on $100/mo): {high_econ}")

    log.info(f"\n  COMPOUNDING IMPACT (20 years, 8% annual return, $100/mo):")
    log.info(f"  {'Ticker':<8} {'Edge/yr':>10} {'20yr value':>12}")
    log.info(f"  {'─'*32}")
    for ticker, r in sorted(results.items(),
                             key=lambda x: x[1]["20yr_compounded_value"],
                             reverse=True):
        log.info(f"  {ticker:<8} ${r['dollar_per_year']:>9.2f} ${r['20yr_compounded_value']:>11,.0f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    regime_results = check_regime_dependence()
    sharpe_results = check_sharpe_inflation()

    # Save combined report
    report = {
        "regime_dependence": {
            ticker: {
                regime: stats
                for regime, stats in regimes.items()
            }
            for ticker, regimes in regime_results.items()
        },
        "sharpe_analysis": sharpe_results,
    }

    out_path = MODELS_DIR / "reasons_3_4_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n  Full report saved to: {out_path}")
