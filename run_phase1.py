"""
TemporalEdge — Phase 1 Pipeline Runner
Run this one script to execute the complete Phase 1 pipeline:
  1. Download all price + macro data
  2. Download FRED macro series
  3. Build calendar features
  4. Engineer all ticker features
  5. Print data quality report

Usage:
  python run_phase1.py              # normal run (uses cache)
  python run_phase1.py --refresh    # force re-download everything
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import argparse
import pandas as pd

from src.data.collector import run_collection
from src.features.engineer import build_all_features, load_macro_panel
from src.utils.logger import get_logger

log = get_logger("pipeline.phase1")


def print_quality_report(features: dict):
    """Print a clean data quality summary."""
    log.info("\n" + "=" * 70)
    log.info("  DATA QUALITY REPORT")
    log.info("=" * 70)
    log.info(f"  {'Ticker':<8} {'Rows':>8} {'Start':>12} {'End':>12} "
             f"{'Features':>10} {'Optimal%':>10} {'AvgSave%':>10}")
    log.info("  " + "-" * 68)

    for ticker, df in sorted(features.items()):
        rows     = len(df)
        start    = df.index.min().date()
        end      = df.index.max().date()
        n_feat   = df.shape[1]
        opt_pct  = df["is_optimal_buy_day"].mean() * 100 if "is_optimal_buy_day" in df else 0
        avg_save = df["vs_day27_pct"].mean() * 100 if "vs_day27_pct" in df else 0

        log.info(
            f"  {ticker:<8} {rows:>8,} {str(start):>12} {str(end):>12} "
            f"{n_feat:>10} {opt_pct:>9.1f}% {avg_save:>+9.2f}%"
        )

    log.info("=" * 70)
    log.info("\n  Legend:")
    log.info("  Optimal% = % of purchase-window days that were lowest price in month")
    log.info("  AvgSave% = avg % cheaper vs buying on day 27 (+ = you would save)")
    log.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="TemporalEdge Phase 1 Pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download all data (ignores cache)")
    args = parser.parse_args()

    t0 = time.time()

    log.info("\n" + "🚀 " * 20)
    log.info("  TEMPORALEDGE — PHASE 1 PIPELINE")
    log.info("  Optimal DCA Entry Timing System")
    log.info("🚀 " * 20 + "\n")

    # Step 1: Collect
    log.info("STEP 1/2 — DATA COLLECTION")
    collection = run_collection(force_refresh=args.refresh)

    # Step 2: Feature engineering
    log.info("\nSTEP 2/2 — FEATURE ENGINEERING")
    features = build_all_features()

    # Report
    print_quality_report(features)

    elapsed = time.time() - t0
    log.info(f"\n✅ Phase 1 complete in {elapsed:.1f}s")
    log.info(f"   Ready for Phase 2: EDA & model training\n")

    return features


if __name__ == "__main__":
    main()
