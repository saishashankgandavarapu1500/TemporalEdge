"""
TemporalEdge — MLflow Experiment Tracker  (Phase 3, Step 3)
Logs every training run so you can compare experiments over time.
This is what makes the project look professional on a resume.

Usage:
  from src.models.experiment import tracker
  tracker.log_training_run(ticker, bundle, backtest)

View runs:
  mlflow ui --backend-store-uri ./mlruns
  Then open http://localhost:5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import ROOT
from src.utils.logger import get_logger

log = get_logger("models.experiment")

MLRUNS_DIR = ROOT / "mlruns"


def _get_mlflow():
    """Lazy import mlflow — gracefully skip if not installed."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        log.warning("  mlflow not installed — skipping experiment tracking")
        log.warning("  Install with: pip install mlflow")
        return None


class ExperimentTracker:
    """Wraps MLflow for TemporalEdge experiment tracking."""

    def __init__(self):
        self.mlflow = _get_mlflow()
        self.experiment_name = "temporaledge_phase3"
        self._setup()

    def _setup(self):
        if not self.mlflow:
            return
        self.mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        try:
            exp = self.mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                self.mlflow.create_experiment(self.experiment_name)
            self.mlflow.set_experiment(self.experiment_name)
            log.info(f"  MLflow tracking: {MLRUNS_DIR}")
        except Exception as e:
            log.warning(f"  MLflow setup error: {e}")

    def log_training_run(
        self,
        ticker: str,
        bundle: dict,
        backtest: dict,
    ):
        """Log one ticker's training run to MLflow."""
        if not self.mlflow:
            return

        run_name = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        try:
            with self.mlflow.start_run(run_name=run_name):
                # Parameters
                self.mlflow.log_params({
                    "ticker":            ticker,
                    "n_features":        len(bundle.get("features", [])),
                    "training_period":   bundle.get("training_period", ""),
                    "n_training_rows":   bundle.get("n_training_rows", 0),
                    "walk_forward_folds":len(bundle.get("wf_fold_metrics", [])),
                })

                # Backtest metrics
                if backtest:
                    self.mlflow.log_metrics({
                        "win_rate_pct":      backtest.get("win_rate_pct", 0),
                        "avg_saving_pct":    backtest.get("avg_saving_pct", 0),
                        "median_saving_pct": backtest.get("median_saving_pct", 0),
                        "avg_capture_rate":  backtest.get("avg_capture_rate", 0),
                        "best_month":        backtest.get("best_month", 0),
                        "worst_month":       backtest.get("worst_month", 0),
                        "n_months":          backtest.get("n_months", 0),
                    })

                # Average AUC across walk-forward folds
                fold_metrics = bundle.get("wf_fold_metrics", [])
                if fold_metrics:
                    aucs = [f["auc"] for f in fold_metrics if "auc" in f]
                    if aucs:
                        self.mlflow.log_metric("mean_auc", round(np.mean(aucs), 4))
                        self.mlflow.log_metric("std_auc",  round(np.std(aucs), 4))

                # Top 10 features
                importance = bundle.get("importance_clf")
                if importance is not None and len(importance) > 0:
                    top10 = importance.head(10)
                    for feat, imp in top10.items():
                        safe_feat = feat.replace("(","").replace(")","").replace(" ","_")
                        self.mlflow.log_metric(f"feat_{safe_feat}", round(float(imp), 4))

                log.info(f"  MLflow run logged: {run_name}")

        except Exception as e:
            log.warning(f"  MLflow logging failed for {ticker}: {e}")

    def log_portfolio_summary(self, backtest_summary: list[dict]):
        """Log portfolio-level summary run."""
        if not self.mlflow:
            return

        try:
            with self.mlflow.start_run(run_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}"):
                if backtest_summary:
                    win_rates   = [r["win_rate_pct"] for r in backtest_summary]
                    avg_savings = [r["avg_saving_pct"] for r in backtest_summary]

                    self.mlflow.log_metrics({
                        "portfolio_avg_win_rate":    round(np.mean(win_rates), 2),
                        "portfolio_avg_saving_pct":  round(np.mean(avg_savings), 4),
                        "portfolio_best_ticker_win": round(max(win_rates), 1),
                        "n_tickers":                 len(backtest_summary),
                    })
                log.info("  MLflow portfolio summary logged")
        except Exception as e:
            log.warning(f"  MLflow portfolio summary failed: {e}")


# Singleton
tracker = ExperimentTracker()
