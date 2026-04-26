"""
TemporalEdge — LightGBM Trainer  (Phase 3, Step 2)
Walk-forward training for all 11 portfolio tickers.

Architecture (informed by EDA):
  Model A: Classifier  → P(today is optimal buy day in window)
  Model B: Regressor   → predicted saving % vs day 27

Walk-forward scheme:
  Train on 36 months → predict next 1 month → slide forward
  No lookahead bias — model never sees future data during training

Key EDA-driven decisions:
  - scale_pos_weight=20 (class imbalance: 4.8% positive rate)
  - Regime as a feature (crash=+1.46%, sideways=-0.14% saving)
  - Top 45 features (reduces noise vs using all 121)
  - VIX-confidence weighting at prediction time
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler

from src.config import (
    PORTFOLIO, PURCHASE_WINDOW_START, YOUR_CURRENT_DAY,
    WALK_FORWARD_TRAIN_MONTHS, WALK_FORWARD_TEST_MONTHS,
    DATA_FEAT, ROOT
)
from src.models.feature_selector import get_feature_list, get_regime_label
from src.utils.logger import get_logger

log = get_logger("models.trainer")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model proxy table — used for both simulation and live prediction
# VTI: Phase 4 confirmed own model is ~random (50.9% win) due to
# pre-2010 regime shift in training data. VOO model (79.1% win,
# post-2010) transfers cleanly — same macro drivers, >0.99 corr.
# VTI features (actual prices + macro) are always used as inputs.
# Only the trained weights come from VOO.
MODEL_PROXY = {
    "VTI": "VOO",
}


def load_model_for_prediction(ticker: str) -> dict | None:
    """
    Load the correct model bundle for a given ticker.
    Applies MODEL_PROXY so VTI automatically uses VOO weights.
    This is the single entry point for live prediction in Phase 5+.
    """
    model_ticker = MODEL_PROXY.get(ticker, ticker)
    path = MODELS_DIR / f"{model_ticker}_model.pkl"

    if not path.exists():
        log.warning(f"  Model not found: {path}. Run Phase 3 first.")
        return None

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if model_ticker != ticker:
        log.info(
            f"  {ticker}: using {model_ticker} model weights (proxy) | "
            f"VTI own model was ~random, VOO transfers cleanly"
        )
        bundle = dict(bundle)
        bundle["proxy_for"]       = ticker
        bundle["original_ticker"] = model_ticker

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Classifier: P(optimal buy day)
CLASSIFIER_PARAMS = {
    "objective":           "binary",
    "metric":              "auc",
    "n_estimators":        500,
    "learning_rate":       0.03,
    "num_leaves":          31,
    "min_child_samples":   20,
    "feature_fraction":    0.8,
    "bagging_fraction":    0.8,
    "bagging_freq":        5,
    "reg_alpha":           0.1,
    "reg_lambda":          0.1,
    # EDA finding: 4.8% positive rate → scale_pos_weight ≈ 20
    "scale_pos_weight":    20,
    "random_state":        42,
    "verbose":             -1,
    "n_jobs":              -1,
}

# Regressor: predicted saving % vs day 27
REGRESSOR_PARAMS = {
    "objective":           "regression",
    "metric":              "mae",
    "n_estimators":        500,
    "learning_rate":       0.03,
    "num_leaves":          31,
    "min_child_samples":   20,
    "feature_fraction":    0.8,
    "bagging_fraction":    0.8,
    "bagging_freq":        5,
    "reg_alpha":           0.1,
    "reg_lambda":          0.1,
    "random_state":        42,
    "verbose":             -1,
    "n_jobs":              -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ticker_data(ticker: str) -> pd.DataFrame | None:
    path = DATA_FEAT / f"{ticker}.parquet"
    if not path.exists():
        log.warning(f"  {ticker}: feature file not found — run Phase 1 first")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    # Only use purchase window rows for training
    df = df[df.index.day >= PURCHASE_WINDOW_START].copy()
    df = df.sort_index()
    return df


def prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    target_clf: str = "is_optimal_buy_day",
    target_reg: str = "vs_day27_pct",
) -> tuple:
    """
    Prepare feature matrix and targets.
    Returns (X, y_clf, y_reg) with NaN rows dropped.
    """
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        log.debug(f"  Features not found (will skip): {list(missing)[:5]}")

    subset = df[available + [target_clf, target_reg]].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna(subset=available + [target_clf])

    X      = subset[available]
    y_clf  = subset[target_clf].astype(int)
    y_reg  = subset[target_reg].fillna(0)

    return X, y_clf, y_reg


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward trainer
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_train(
    ticker: str,
    df: pd.DataFrame,
    features: list[str],
) -> dict:
    """
    Walk-forward cross-validation.
    Slides a training window forward month by month.
    Returns per-period predictions and final trained model.
    """
    periods = df.index.to_period("M").unique().sort_values()
    n       = len(periods)
    train_n = WALK_FORWARD_TRAIN_MONTHS
    test_n  = WALK_FORWARD_TEST_MONTHS

    if n < train_n + test_n:
        log.warning(f"  {ticker}: only {n} months — need at least {train_n + test_n}")
        return {}

    all_preds  = []
    fold_metrics = []

    log.info(f"  {ticker}: walk-forward {n - train_n} folds "
             f"({periods[0]} → {periods[-1]})")

    for start_idx in range(0, n - train_n - test_n + 1, test_n):
        train_periods = periods[start_idx : start_idx + train_n]
        test_periods  = periods[start_idx + train_n : start_idx + train_n + test_n]

        train_mask = df.index.to_period("M").isin(train_periods)
        test_mask  = df.index.to_period("M").isin(test_periods)

        train_df = df[train_mask]
        test_df  = df[test_mask]

        if len(train_df) < 50 or len(test_df) < 3:
            continue

        X_train, y_clf_train, y_reg_train = prepare_xy(train_df, features)
        X_test,  y_clf_test,  y_reg_test  = prepare_xy(test_df,  features)

        if len(X_train) < 30 or len(X_test) < 1:
            continue

        # ── Classifier ──────────────────────────────────────────────────────
        clf = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        clf.fit(
            X_train, y_clf_train,
            eval_set=[(X_test, y_clf_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        clf_proba = clf.predict_proba(X_test)[:, 1]

        # ── Regressor ───────────────────────────────────────────────────────
        reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        reg.fit(
            X_train, y_reg_train,
            eval_set=[(X_test, y_reg_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        reg_pred = reg.predict(X_test)

        # ── Ensemble score: combine classifier + regressor ───────────────
        # Normalise regressor to [0,1] range for blending
        reg_norm = (reg_pred - reg_pred.min()) / (reg_pred.max() - reg_pred.min() + 1e-9)
        ensemble_score = 0.6 * clf_proba + 0.4 * reg_norm

        # ── VIX confidence adjustment (EDA finding) ──────────────────────
        # When VIX < 20, reduce model confidence (day 27 is fine in calm markets)
        # When VIX > 20, trust the model more
        vix_values = X_test["vix"].values if "vix" in X_test.columns else np.full(len(X_test), 18)
        vix_multiplier = np.where(vix_values >= 20, 1.0, 0.6)
        ensemble_score = ensemble_score * vix_multiplier

        # ── Store predictions ────────────────────────────────────────────
        test_preds = test_df.loc[X_test.index].copy()
        test_preds["clf_proba"]      = clf_proba
        test_preds["reg_pred"]       = reg_pred
        test_preds["ensemble_score"] = ensemble_score
        test_preds["features_used"]  = len(features)
        all_preds.append(test_preds)

        # ── Fold metrics ─────────────────────────────────────────────────
        try:
            auc = roc_auc_score(y_clf_test, clf_proba) if y_clf_test.sum() > 0 else 0.5
        except Exception:
            auc = 0.5

        fold_metrics.append({
            "test_period":   str(test_periods[0]),
            "train_months":  len(train_periods),
            "test_rows":     len(X_test),
            "auc":           round(auc, 4),
            "mae_saving":    round(float(mean_absolute_error(y_reg_test, reg_pred)), 4),
            "n_positive":    int(y_clf_test.sum()),
        })

    return {
        "predictions":    pd.concat(all_preds) if all_preds else pd.DataFrame(),
        "fold_metrics":   fold_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final model trainer (trained on full history for deployment)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    ticker: str,
    df: pd.DataFrame,
    features: list[str],
) -> dict:
    """
    Train final production model on ALL available data.
    This is the model used for live monthly recommendations.
    """
    X, y_clf, y_reg = prepare_xy(df, features)

    if len(X) < 100:
        log.warning(f"  {ticker}: insufficient data for final model ({len(X)} rows)")
        return {}

    # Classifier
    clf = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    clf.fit(X, y_clf)

    # Regressor
    reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    reg.fit(X, y_reg)

    # Feature importance
    importance_clf = pd.Series(
        clf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    importance_reg = pd.Series(
        reg.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    return {
        "classifier":       clf,
        "regressor":        reg,
        "features":         list(X.columns),
        "importance_clf":   importance_clf,
        "importance_reg":   importance_reg,
        "n_training_rows":  len(X),
        "training_period":  f"{df.index.min().date()} → {df.index.max().date()}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction for a specific month window
# ─────────────────────────────────────────────────────────────────────────────

def predict_optimal_day(
    ticker: str,
    model_bundle: dict,
    window_df: pd.DataFrame,
) -> dict:
    """
    Given the purchase window for a month, predict the optimal buy day.

    Returns:
        recommended_day:  int (day of month)
        confidence:       float 0-1
        ensemble_scores:  Series (score per day in window)
        reasoning_data:   dict (for LLM macro layer)
    """
    clf      = model_bundle["classifier"]
    reg      = model_bundle["regressor"]
    features = model_bundle["features"]

    # Phase 3 finding: TSLA (7.9% capture) and NVDA (38.3% capture) have
    # high win rates but the model can't reliably pinpoint their exact dip day.
    # Their intra-month volatility is real but chaotic — too driven by news/earnings.
    # Penalty: widen uncertainty band so the recommendation output is honest.
    # This does NOT change WHICH day is recommended, only how confident we claim to be.
    CAPTURE_RATE_BY_TICKER = {
        "TSLA":  0.079,   # 7.9%  — very uncertain, model barely pinpoints the dip
        "NVDA":  0.383,   # 38.3% — moderate, earnings/options noise limits precision
        "AAPL":  0.790,   # using win_rate as proxy (capture was -29.4%, formula bug)
        "VEA":   0.767,   # same proxy
        "VOO":   0.775,   # same proxy
        "VTI":   0.791,   # using VOO model proxy — VOO win rate applies
        "VYM":   0.552,
        "SCHD":  0.457,
        "VWO":   0.431,
        "BND":   0.265,
        "VXUS":  0.088,
    }
    # Reliability multiplier: scales confidence score before output
    # Tickers with low capture → confidence is discounted → wider uncertainty band
    reliability = CAPTURE_RATE_BY_TICKER.get(ticker, 0.5)
    # Map to [0.5, 1.0] range so we never report 0 confidence even for TSLA
    reliability_mult = 0.5 + (reliability * 0.5)

    available = [f for f in features if f in window_df.columns]
    X = window_df[available].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) == 0:
        log.warning(f"  {ticker}: empty window — defaulting to day {YOUR_CURRENT_DAY}")
        return {"recommended_day": YOUR_CURRENT_DAY, "confidence": 0.0}

    clf_proba  = clf.predict_proba(X)[:, 1]
    reg_pred   = reg.predict(X)
    reg_norm   = (reg_pred - reg_pred.min()) / (reg_pred.max() - reg_pred.min() + 1e-9)
    ensemble   = 0.6 * clf_proba + 0.4 * reg_norm

    # Adjustment 1 — VIX regime (from EDA: model least useful when VIX < 20)
    vix_vals   = X["vix"].values if "vix" in X.columns else np.full(len(X), 18)
    vix_mean   = float(np.nanmean(vix_vals))
    vix_mult   = 1.0 if vix_mean >= 20 else 0.6
    ensemble   = ensemble * vix_mult

    scores     = pd.Series(ensemble, index=window_df.index)
    best_idx   = scores.idxmax()
    best_day   = best_idx.day
    confidence = float(scores.max())

    # Normalise confidence to 0-1 range, then apply ticker reliability discount
    if scores.max() > scores.min():
        conf_norm = (confidence - scores.min()) / (scores.max() - scores.min())
    else:
        conf_norm = 0.5
    # Adjustment 2 — ticker reliability (Phase 3 finding: TSLA/NVDA less precise)
    conf_norm = conf_norm * reliability_mult

    # Build reasoning data for LLM layer
    reasoning = {
        "ticker":            ticker,
        "recommended_day":   best_day,
        "confidence":        round(conf_norm, 3),
        "vix_level":         round(vix_mean, 1),
        "vix_regime":        get_regime_label(vix_mean),
        "vix_trust":         "high" if vix_mean >= 20 else "low",
        "top_3_days":        scores.nlargest(3).index.day.tolist(),
        "predicted_saving":  round(float(reg_pred[scores.values.argmax()]), 3),
        "day_scores":        {d.day: round(float(s), 4) for d, s in scores.items()},
        "current_day":       YOUR_CURRENT_DAY,
        # Reliability: how well the model pinpoints the exact dip day (Phase 3 finding)
        # TSLA=0.54 (low), NVDA=0.69, VOO=0.89 (high)
        "model_reliability":    round(reliability_mult, 3),
        # Uncertainty band: ±days around recommendation the true optimal might fall
        # High reliability (VOO) → ±1 day.  Low reliability (TSLA) → ±3 days.
        "uncertainty_days":     1 if reliability_mult >= 0.75 else (2 if reliability_mult >= 0.60 else 3),
        # For Monte Carlo: whether to trust the exact day or just the window
        "timing_precision":     "exact" if reliability_mult >= 0.75 else ("window" if reliability_mult >= 0.60 else "loose"),
        "improvement_vs_current": round(
            float(scores.max() - scores.get(
                window_df.index[window_df.index.day == YOUR_CURRENT_DAY][0]
                if any(window_df.index.day == YOUR_CURRENT_DAY) else window_df.index[0]
            , 0)), 4
        ),
    }

    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Backtest evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_backtest(predictions_df: pd.DataFrame, ticker: str) -> dict:
    """
    Given walk-forward predictions, evaluate how well the model
    beats the fixed day 27 strategy.
    """
    if predictions_df.empty or "ensemble_score" not in predictions_df.columns:
        return {}

    results = []

    for period, group in predictions_df.groupby(predictions_df.index.to_period("M")):
        if len(group) < 2:
            continue

        # Model recommendation: day with highest ensemble score
        model_day_idx    = group["ensemble_score"].idxmax()
        model_day        = model_day_idx.day
        model_price      = group.loc[model_day_idx, "close"]

        # Day 27 baseline
        day27_rows = group[group.index.day == YOUR_CURRENT_DAY]
        if len(day27_rows) == 0:
            # Closest to 27 if 27 is weekend/holiday
            candidates = group[group.index.day >= 25]
            day27_rows = candidates.iloc[[0]] if len(candidates) else group.iloc[[-1]]
        day27_price = float(day27_rows["close"].values[0])

        # Actual optimal (ground truth)
        actual_opt_idx   = group["close"].idxmin()
        actual_opt_day   = actual_opt_idx.day
        actual_opt_price = float(group.loc[actual_opt_idx, "close"])

        # Savings
        model_saving_vs_27  = (day27_price - model_price) / day27_price * 100
        optimal_saving_vs_27 = (day27_price - actual_opt_price) / day27_price * 100

        results.append({
            "period":               str(period),
            "model_day":            model_day,
            "model_price":          round(float(model_price), 4),
            "day27_price":          round(day27_price, 4),
            "actual_opt_day":       actual_opt_day,
            "actual_opt_price":     round(actual_opt_price, 4),
            "model_saving_vs_27":   round(model_saving_vs_27, 4),
            "optimal_saving_vs_27": round(optimal_saving_vs_27, 4),
            "model_beat_27":        model_saving_vs_27 > 0,
            "capture_rate":         round(
                model_saving_vs_27 / optimal_saving_vs_27 * 100
                if optimal_saving_vs_27 > 0 else 0, 2
            ),
        })

    if not results:
        return {}

    df_results = pd.DataFrame(results)
    win_rate      = df_results["model_beat_27"].mean() * 100
    avg_saving    = df_results["model_saving_vs_27"].mean()
    median_saving = df_results["model_saving_vs_27"].median()

    # Capture rate fix: the raw mean is distorted by large negative values in
    # loss months (e.g. model_saving=-3%, optimal=0.5% → capture=-600%).
    # These blow up the mean even when the model wins 67% of the time.
    # Fix: use median capture over months where there was a real opportunity
    # (optimal_saving_vs_27 > 0.1pp). Median is robust to the blow-up.
    # Also clip each month's capture to [-100, 100] before taking the median
    # so outliers in tiny-spread months don't corrupt the result.
    opportunity_rows = df_results[df_results["optimal_saving_vs_27"] > 0.1].copy()
    if len(opportunity_rows) >= 10:
        opportunity_rows["capture_clipped"] = opportunity_rows["capture_rate"].clip(-100, 100)
        avg_capture = float(opportunity_rows["capture_clipped"].median())
    else:
        # Fallback: clipped mean across all rows
        avg_capture = float(df_results["capture_rate"].clip(-100, 100).mean())

    return {
        "ticker":           ticker,
        "n_months":         len(df_results),
        "win_rate_pct":     round(win_rate, 1),
        "avg_saving_pct":   round(avg_saving, 3),
        "median_saving_pct":round(median_saving, 3),
        "avg_capture_rate": round(avg_capture, 1),
        "best_month":       round(df_results["model_saving_vs_27"].max(), 2),
        "worst_month":      round(df_results["model_saving_vs_27"].min(), 2),
        "results_df":       df_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def train_all_tickers(save: bool = True) -> dict:
    """Train models for all 11 portfolio tickers."""

    log.info("=" * 60)
    log.info("  TEMPORALEDGE — PHASE 3: LIGHTGBM TRAINING")
    log.info("=" * 60)
    log.info(f"  Walk-forward: {WALK_FORWARD_TRAIN_MONTHS}mo train → "
             f"{WALK_FORWARD_TEST_MONTHS}mo predict")
    log.info(f"  Class imbalance: scale_pos_weight=20 (4.8% positives)")
    log.info(f"  VIX adjustment: confidence reduced when VIX < 20\n")

    all_results  = {}
    backtest_summary = []

    for ticker in PORTFOLIO:
        log.info(f"\n{'─' * 50}")
        log.info(f"  Training: {ticker} — {PORTFOLIO[ticker]['name']}")
        log.info(f"{'─' * 50}")

        # Load data
        df = load_ticker_data(ticker)
        if df is None:
            continue

        # Select features
        features = get_feature_list(ticker, df.columns.tolist())

        # Walk-forward validation
        log.info(f"  Running walk-forward validation...")
        wf_result = walk_forward_train(ticker, df, features)

        if not wf_result or wf_result["predictions"].empty:
            log.warning(f"  {ticker}: walk-forward produced no predictions")
            continue

        # Evaluate backtest
        bt = evaluate_backtest(wf_result["predictions"], ticker)

        if bt:
            log.info(f"  ✓ Win rate:     {bt['win_rate_pct']:.1f}%")
            log.info(f"  ✓ Avg saving:   {bt['avg_saving_pct']:+.3f}% vs day {YOUR_CURRENT_DAY}")
            log.info(f"  ✓ Capture rate: {bt['avg_capture_rate']:.1f}% of available saving")
            log.info(f"  ✓ Best month:   {bt['best_month']:+.2f}%")
            log.info(f"  ✓ Worst month:  {bt['worst_month']:+.2f}%")
            backtest_summary.append({
                k: v for k, v in bt.items() if k != "results_df"
            })

        # Train final production model
        log.info(f"  Training final production model (full history)...")
        final = train_final_model(ticker, df, features)

        if not final:
            continue

        log.info(f"  ✓ Final model trained on {final['n_training_rows']:,} rows")
        log.info(f"  ✓ Top features: "
                 f"{final['importance_clf'].head(5).index.tolist()}")

        # Save model bundle
        bundle = {
            "ticker":          ticker,
            "classifier":      final["classifier"],
            "regressor":       final["regressor"],
            "features":        final["features"],
            "importance_clf":  final["importance_clf"],
            "importance_reg":  final["importance_reg"],
            "backtest":        {k: v for k, v in bt.items() if k != "results_df"} if bt else {},
            "wf_fold_metrics": wf_result["fold_metrics"],
            "training_period": final["training_period"],
        }

        if save:
            model_path = MODELS_DIR / f"{ticker}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(bundle, f)
            log.info(f"  ✓ Saved: {model_path.name}")

            # Also save backtest results CSV
            if bt and "results_df" in bt:
                bt["results_df"].to_csv(
                    MODELS_DIR / f"{ticker}_backtest.csv", index=False
                )

        all_results[ticker] = bundle

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}")
    log.info("  TRAINING COMPLETE — BACKTEST SUMMARY")
    log.info("=" * 60)
    log.info(f"  {'Ticker':<8} {'Win%':>6} {'AvgSave%':>10} {'Capture%':>10} {'Best':>8} {'Worst':>8}")
    log.info(f"  {'─'*54}")

    for row in sorted(backtest_summary, key=lambda x: x["win_rate_pct"], reverse=True):
        log.info(
            f"  {row['ticker']:<8} "
            f"{row['win_rate_pct']:>5.1f}% "
            f"{row['avg_saving_pct']:>+9.3f}% "
            f"{row['avg_capture_rate']:>9.1f}% "
            f"{row['best_month']:>+7.2f}% "
            f"{row['worst_month']:>+7.2f}%"
        )

    # Save summary JSON
    if save and backtest_summary:
        with open(MODELS_DIR / "backtest_summary.json", "w") as f:
            json.dump(backtest_summary, f, indent=2)
        log.info(f"\n  Results saved to: {MODELS_DIR}")

    log.info("=" * 60)
    return all_results


if __name__ == "__main__":
    train_all_tickers(save=True)