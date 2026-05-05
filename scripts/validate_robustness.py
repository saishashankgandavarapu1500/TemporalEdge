"""
TemporalEdge — Robustness Validation
Fix 2: Out-of-sample walk-forward stability check
Fix 3: Statistical significance tests (binomial, t-test, Sharpe)

Run with: python scripts/validate_robustness.py
Output:   models/robustness_report.json  +  printed summary
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PORTFOLIO, YOUR_CURRENT_DAY, ROOT
from src.utils.logger import get_logger

log = get_logger("validation.robustness")

MODELS_DIR = ROOT / "models"
REPORT_PATH = MODELS_DIR / "robustness_report.json"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — Statistical significance tests
# ─────────────────────────────────────────────────────────────────────────────

def compute_significance(results_df: pd.DataFrame, ticker: str) -> dict:
    """
    Given per-month backtest results, compute:
      1. Binomial test  — is win rate > 50% by chance?
      2. T-test         — is avg saving significantly > 0?
      3. Sharpe ratio   — of the timing edge (annualised)
    """
    savings = results_df["model_saving_vs_27"].values
    wins    = results_df["model_beat_27"].values.astype(int)

    n       = len(savings)
    n_wins  = wins.sum()
    win_rate = n_wins / n

    # ── 1. Binomial test — win rate vs random (50%) ───────────────────────────
    binom_result = stats.binomtest(n_wins, n, p=0.5, alternative="greater")
    p_value_binom = binom_result.pvalue

    # ── 2. One-sample t-test — avg saving vs 0 ───────────────────────────────
    t_stat, p_value_t = stats.ttest_1samp(savings, popmean=0, alternative="greater")

    # ── 3. Sharpe ratio of the monthly edge ──────────────────────────────────
    mean_saving = savings.mean()
    std_saving  = savings.std(ddof=1)
    sharpe      = (mean_saving / std_saving) * np.sqrt(12) if std_saving > 0 else 0.0

    # ── 4. Effect size (Cohen's d) ────────────────────────────────────────────
    cohens_d = mean_saving / std_saving if std_saving > 0 else 0.0

    # ── Significance label ────────────────────────────────────────────────────
    if p_value_binom < 0.01 and p_value_t < 0.01:
        sig_label = "HIGHLY SIGNIFICANT (p<0.01)"
    elif p_value_binom < 0.05 and p_value_t < 0.05:
        sig_label = "SIGNIFICANT (p<0.05)"
    elif p_value_binom < 0.10 or p_value_t < 0.10:
        sig_label = "MARGINAL (p<0.10)"
    else:
        sig_label = "NOT SIGNIFICANT"

    return {
        "ticker":            ticker,
        "n_months":          n,
        "n_wins":            int(n_wins),
        "win_rate_pct":      round(win_rate * 100, 1),
        "avg_saving_pct":    round(mean_saving, 4),
        "std_saving_pct":    round(std_saving, 4),
        "p_value_binomial":  round(p_value_binom, 4),
        "p_value_ttest":     round(p_value_t, 4),
        "t_statistic":       round(t_stat, 3),
        "sharpe_of_edge":    round(sharpe, 3),
        "cohens_d":          round(cohens_d, 3),
        "significance":      sig_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — Walk-forward stability check (year-by-year)
# ─────────────────────────────────────────────────────────────────────────────

def check_walk_forward_stability(results_df: pd.DataFrame, ticker: str) -> dict:
    """
    Break out-of-sample results by year.
    A robust model should maintain positive win rate across most years,
    not just in aggregate.
    """
    results_df = results_df.copy()
    results_df["year"] = pd.to_datetime(
        results_df["period"].astype(str)
    ).dt.year

    yearly = []
    for year, group in results_df.groupby("year"):
        n = len(group)
        if n < 3:   # skip years with too few months
            continue
        win_rate = group["model_beat_27"].mean() * 100
        avg_save = group["model_saving_vs_27"].mean()
        yearly.append({
            "year":         int(year),
            "n_months":     n,
            "win_rate_pct": round(win_rate, 1),
            "avg_saving":   round(avg_save, 4),
            "positive":     win_rate > 50,
        })

    yearly_df = pd.DataFrame(yearly)
    if yearly_df.empty:
        return {}

    positive_years = yearly_df["positive"].sum()
    total_years    = len(yearly_df)
    stability_pct  = positive_years / total_years * 100

    # Consistency: std of yearly win rates (lower = more stable)
    win_rate_std = yearly_df["win_rate_pct"].std()

    # Worst year — critical for interview honesty
    worst = yearly_df.loc[yearly_df["win_rate_pct"].idxmin()]
    best  = yearly_df.loc[yearly_df["win_rate_pct"].idxmax()]

    if stability_pct >= 75:
        stability_label = "STABLE"
    elif stability_pct >= 60:
        stability_label = "MOSTLY STABLE"
    else:
        stability_label = "UNSTABLE — regime-dependent"

    return {
        "ticker":             ticker,
        "total_years":        total_years,
        "positive_years":     int(positive_years),
        "stability_pct":      round(stability_pct, 1),
        "stability_label":    stability_label,
        "win_rate_std":       round(win_rate_std, 1),
        "best_year":          int(best["year"]),
        "best_year_win_rate": round(float(best["win_rate_pct"]), 1),
        "worst_year":         int(worst["year"]),
        "worst_year_win_rate":round(float(worst["win_rate_pct"]), 1),
        "yearly_breakdown":   yearly_df.to_dict("records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_validation() -> dict:
    log.info("\n" + "=" * 60)
    log.info("  TEMPORALEDGE — ROBUSTNESS VALIDATION")
    log.info("  Fix 2: Walk-forward stability | Fix 3: Statistical tests")
    log.info("=" * 60)

    full_report = {}

    for ticker in PORTFOLIO:
        csv_path = MODELS_DIR / f"{ticker}_backtest.csv"
        if not csv_path.exists():
            log.warning(f"  {ticker}: no backtest CSV — run trainer.py first")
            continue

        results_df = pd.read_csv(csv_path)
        if results_df.empty:
            continue

        log.info(f"\n{'─' * 50}")
        log.info(f"  {ticker}")
        log.info(f"{'─' * 50}")

        # Fix 3 — statistical significance
        sig = compute_significance(results_df, ticker)
        log.info(f"  Win rate:      {sig['win_rate_pct']}% ({sig['n_wins']}/{sig['n_months']} months)")
        log.info(f"  Avg saving:    {sig['avg_saving_pct']:+.4f}%")
        log.info(f"  Std saving:    {sig['std_saving_pct']:.4f}%")
        log.info(f"  Binomial p:    {sig['p_value_binomial']:.4f}")
        log.info(f"  T-test p:      {sig['p_value_ttest']:.4f}  (t={sig['t_statistic']:.2f})")
        log.info(f"  Sharpe(edge):  {sig['sharpe_of_edge']:.3f}")
        log.info(f"  Cohen's d:     {sig['cohens_d']:.3f}")
        log.info(f"  → {sig['significance']}")

        # Fix 2 — walk-forward stability
        stab = check_walk_forward_stability(results_df, ticker)
        if stab:
            log.info(f"  Stable years:  {stab['positive_years']}/{stab['total_years']} ({stab['stability_pct']}%)")
            log.info(f"  Best year:     {stab['best_year']} — {stab['best_year_win_rate']}% win rate")
            log.info(f"  Worst year:    {stab['worst_year']} — {stab['worst_year_win_rate']}% win rate")
            log.info(f"  Win rate std:  {stab['win_rate_std']}% (lower = more stable)")
            log.info(f"  → {stab['stability_label']}")

        full_report[ticker] = {
            "significance":  sig,
            "stability":     stab,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}")
    log.info("  SUMMARY")
    log.info("=" * 60)
    log.info(f"  {'Ticker':<8} {'Win%':>6} {'p(binom)':>10} {'p(t)':>8} {'Sharpe':>8} {'Stability':>12} {'Significance'}")
    log.info(f"  {'─' * 70}")

    for ticker, report in full_report.items():
        sig  = report["significance"]
        stab = report.get("stability", {})
        log.info(
            f"  {ticker:<8} "
            f"{sig['win_rate_pct']:>5.1f}% "
            f"{sig['p_value_binomial']:>10.4f} "
            f"{sig['p_value_ttest']:>8.4f} "
            f"{sig['sharpe_of_edge']:>8.3f} "
            f"{stab.get('stability_label', 'N/A'):>12}  "
            f"{sig['significance']}"
        )

    # Save report
    MODELS_DIR.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        # Remove yearly_breakdown from JSON to keep it clean
        clean = {}
        for t, r in full_report.items():
            clean[t] = {
                "significance": r["significance"],
                "stability": {
                    k: v for k, v in r.get("stability", {}).items()
                    if k != "yearly_breakdown"
                },
            }
        json.dump(clean, f, indent=2)

    log.info(f"\n  Report saved to: {REPORT_PATH}")
    log.info("=" * 60)

    # ── Interview-ready answer ─────────────────────────────────────────────────
    log.info("\n  INTERVIEW-READY ANSWER:")
    log.info("  " + "─" * 58)
    sig_tickers  = [t for t, r in full_report.items()
                    if "SIGNIFICANT" in r["significance"]["significance"]]
    stab_tickers = [t for t, r in full_report.items()
                    if r.get("stability", {}).get("stability_pct", 0) >= 75]
    avg_sharpe   = np.mean([r["significance"]["sharpe_of_edge"]
                            for r in full_report.values()])
    log.info(f"  Statistically significant tickers: {sig_tickers}")
    log.info(f"  Stable across years (≥75%): {stab_tickers}")
    log.info(f"  Average Sharpe of timing edge: {avg_sharpe:.3f}")

    return full_report


if __name__ == "__main__":
    run_validation()
