"""
TemporalEdge — Phase 3 Runner
Trains all 11 LightGBM models, runs walk-forward backtest,
logs to MLflow, and generates a results report.

Usage:
  python run_phase3.py              # full training run
  python run_phase3.py --ticker NVDA  # single ticker (faster for testing)
  python run_phase3.py --no-save    # don't save model files
"""

import sys
import time
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import pandas as pd
import numpy as np

from src.config import PORTFOLIO, YOUR_CURRENT_DAY, ROOT
from src.models.trainer import (
    load_ticker_data,
    train_final_model,
    walk_forward_train,
    evaluate_backtest,
)
from src.models.feature_selector import get_feature_list
from src.models.experiment import tracker
from src.utils.logger import get_logger

log = get_logger("pipeline.phase3")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def print_backtest_report(backtest_summary: list[dict], monthly_usd: dict):
    """Print a detailed backtest report with dollar impact."""
    log.info("\n" + "=" * 70)
    log.info("  PHASE 3 BACKTEST REPORT")
    log.info("=" * 70)

    if not backtest_summary:
        log.warning("  No backtest results to report")
        return

    df = pd.DataFrame(backtest_summary)

    # Per-ticker results
    log.info(f"\n  {'Ticker':<8} {'Win%':>6} {'AvgSave':>9} {'Capture':>9} "
             f"{'$/mo':>6} {'Ann$':>8} {'Months':>7}")
    log.info(f"  {'─'*62}")

    total_monthly     = 0
    total_annual_gain = 0

    for _, row in df.sort_values("win_rate_pct", ascending=False).iterrows():
        t      = row["ticker"]
        mo_usd = monthly_usd.get(t, 0)
        # Annual dollar impact: saving% × monthly investment × 12
        ann_gain = row["avg_saving_pct"] / 100 * mo_usd * 12

        total_monthly     += mo_usd
        total_annual_gain += ann_gain

        log.info(
            f"  {t:<8} "
            f"{row['win_rate_pct']:>5.1f}% "
            f"{row['avg_saving_pct']:>+8.3f}% "
            f"{row['avg_capture_rate']:>8.1f}% "
            f"${mo_usd:>4} "
            f"${ann_gain:>+7.2f} "
            f"{row['n_months']:>6}"
        )

    log.info(f"  {'─'*62}")

    # Portfolio total
    avg_win    = df["win_rate_pct"].mean()
    avg_saving = df["avg_saving_pct"].mean()
    log.info(
        f"  {'PORTFOLIO':<8} "
        f"{avg_win:>5.1f}% "
        f"{avg_saving:>+8.3f}% "
        f"{'─':>9} "
        f"${total_monthly:>4} "
        f"${total_annual_gain:>+7.2f}"
    )

    # Compounding impact — three honest scenarios
    # Conservative: bear decade, model only works in high-VIX months (25% capture)
    # Base case:    15% market, model works ~45% of months (mixed VIX conditions)
    # Optimistic:   15% market, model captures full saving every month (upper bound)
    log.info(f"\n  {'─'*70}")
    log.info(f"  COMPOUNDING IMPACT — 3 SCENARIOS (${total_monthly}/month)")
    log.info(f"  {'─'*70}")
    log.info(f"  {'':>6}  {'Conservative':>16}  {'Base Case':>16}  {'Optimistic':>16}")
    log.info(f"  {'':>6}  {'10% mkt,25%cap':>16}  {'15% mkt,45%cap':>16}  {'15% mkt,100%cap':>16}")
    log.info(f"  {'─'*70}")

    scenarios = [
        ("Conservative", 0.10, 0.25),
        ("Base case",    0.15, 0.45),
        ("Optimistic",   0.15, 1.00),
    ]
    # Pre-compute all scenario rates
    rates = {}
    for label, base, capture in scenarios:
        opt = base + (avg_saving / 100 * capture * 12 * 0.3)
        rates[label] = (base, opt)

    for years in [1, 3, 5, 10, 20]:
        months = years * 12
        cols = []
        for label, base, capture in scenarios:
            base_r, opt_r = rates[label]
            r_b  = base_r / 12
            r_o  = opt_r  / 12
            fv_b = total_monthly * ((1 + r_b) ** months - 1) / r_b
            fv_o = total_monthly * ((1 + r_o) ** months - 1) / r_o
            cols.append(f"+${fv_o-fv_b:>6,.0f}")
        log.info(f"  {years:>2} yr:  {cols[0]:>16}  {cols[1]:>16}  {cols[2]:>16}")

    # Headline rates
    log.info(f"  {'─'*70}")
    rate_line = "  eff. rate: "
    for label, base, capture in scenarios:
        _, opt_r = rates[label]
        rate_line += f"  {opt_r*100:>5.1f}% p.a.    "
    log.info(rate_line)
    log.info(f"\n  NOTE: 'Base case' is the most realistic. Conservative assumes")
    log.info(f"  a low-return decade + model only activates in high-VIX months.")
    log.info(f"  Optimistic is the mathematical upper bound, not a prediction.")
    log.info("=" * 70)

    # VIX regime reminder
    log.info("\n  KEY INSIGHT FROM EDA (informs when to trust the model):")
    log.info("  VIX < 20  → model confidence reduced 40% (day 27 is fine)")
    log.info("  VIX 20-30 → model saves avg +0.17% per month")
    log.info("  VIX > 30  → model saves avg +0.63% per month (most valuable)")
    log.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="TemporalEdge Phase 3")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Train single ticker only (e.g. --ticker NVDA)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving model files")
    args = parser.parse_args()

    t0 = time.time()

    log.info("\n" + "🎯 " * 20)
    log.info("  TEMPORALEDGE — PHASE 3: MODEL TRAINING")
    log.info("🎯 " * 20 + "\n")

    # Optionally train single ticker for faster testing
    if args.ticker:
        tickers_to_train = {args.ticker: PORTFOLIO[args.ticker]}
        log.info(f"  Single ticker mode: {args.ticker}")
    else:
        tickers_to_train = PORTFOLIO

    # Train
    results = {}
    backtest_summary = []
    monthly_usd = {t: PORTFOLIO[t]["monthly_usd"] for t in PORTFOLIO}

    for ticker, info in tickers_to_train.items():
        log.info(f"\n{'═'*50}")
        log.info(f"  {ticker} — {info['name']} (${info['monthly_usd']}/mo)")
        log.info(f"{'═'*50}")

        df = load_ticker_data(ticker)
        if df is None:
            continue

        features = get_feature_list(ticker, df.columns.tolist())

        # Walk-forward validation
        wf = walk_forward_train(ticker, df, features)
        if not wf or wf["predictions"].empty:
            continue

        # Backtest evaluation
        bt = evaluate_backtest(wf["predictions"], ticker)

        if bt:
            summary_row = {k: v for k, v in bt.items() if k != "results_df"}
            backtest_summary.append(summary_row)
            if not args.no_save and "results_df" in bt:
                bt["results_df"].to_csv(MODELS_DIR / f"{ticker}_backtest.csv", index=False)

        # Final production model
        final = train_final_model(ticker, df, features)
        if not final:
            continue

        bundle = {
            "ticker":          ticker,
            "classifier":      final["classifier"],
            "regressor":       final["regressor"],
            "features":        final["features"],
            "importance_clf":  final["importance_clf"],
            "importance_reg":  final["importance_reg"],
            "backtest":        {k: v for k, v in bt.items() if k != "results_df"} if bt else {},
            "wf_fold_metrics": wf["fold_metrics"],
            "training_period": final["training_period"],
            "n_training_rows": final["n_training_rows"],
        }
        results[ticker] = bundle

        # MLflow logging
        tracker.log_training_run(ticker, bundle, bt if bt else {})

        # Save model
        if not args.no_save:
            import pickle
            model_path = MODELS_DIR / f"{ticker}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(bundle, f)
            log.info(f"  ✅ Saved: {model_path.name}")

    # Portfolio summary
    if backtest_summary:
        tracker.log_portfolio_summary(backtest_summary)
        print_backtest_report(backtest_summary, monthly_usd)
        if not args.no_save:
            with open(MODELS_DIR / "backtest_summary.json", "w") as f:
                json.dump(backtest_summary, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n✅ Phase 3 complete in {elapsed:.1f}s")
    log.info(f"   Models saved to: {MODELS_DIR}")
    log.info(f"   View MLflow: mlflow ui --backend-store-uri ./mlruns")
    log.info(f"   Then open:   http://localhost:5000")
    log.info(f"\n   Ready for Phase 4: Monte Carlo simulation\n")

    return results


if __name__ == "__main__":
    main()