"""
TemporalEdge — On-Demand Ticker Pipeline
=========================================
Full pipeline for any ticker not in the portfolio.
Runs in ~1-3 minutes on M2 Mac. Results cached for 30 days.

Steps:
  1. Download max history from yfinance
  2. Build 121 features (same pipeline as Phase 1)
  3. Train LightGBM classifier + regressor (same as Phase 3)
  4. Monte Carlo simulation 1000 runs (same as Phase 4)
  5. LLM advisory via Groq (same as monthly_refresh)

Usage:
  from src.pipeline.on_demand import run_on_demand
  result = run_on_demand("ARKK", monthly_usd=150, horizon_years=15)
"""

import sys
import time
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    DATA_RAW, DATA_FEAT, DATA_MACRO, ROOT,
    PURCHASE_WINDOW_START, WALK_FORWARD_TRAIN_MONTHS,
)
from src.features.engineer import build_ticker_features, load_macro_panel
from src.models.feature_selector import get_feature_list
from src.models.trainer import (
    walk_forward_train,
    train_final_model,
    evaluate_backtest,
    CLASSIFIER_PARAMS,
    REGRESSOR_PARAMS,
)
from src.simulation.scenario_sampler import ScenarioSampler
from src.simulation.agents import agent_a, agent_b, compare_agents
from src.utils.logger import get_logger

log = get_logger("pipeline.on_demand")

MODELS_DIR   = ROOT / "models"
CACHE_DIR    = ROOT / "dashboard" / "cache" / "on_demand"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_AGE_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_result.json"


def _model_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker}_model.pkl"


def _cache_valid(ticker: str) -> bool:
    p = _cache_path(ticker)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=CACHE_MAX_AGE_DAYS)


def _model_valid(ticker: str) -> bool:
    p = _model_path(ticker)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=CACHE_MAX_AGE_DAYS)


def _load_cache(ticker: str) -> dict | None:
    try:
        with open(_cache_path(ticker)) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(ticker: str, result: dict):
    try:
        with open(_cache_path(ticker), "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"  Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download
# ─────────────────────────────────────────────────────────────────────────────

def _download_ticker(ticker: str, progress_cb=None) -> tuple[bool, dict]:
    """
    Download max history for ticker.
    Returns (success, metadata).
    """
    if progress_cb:
        progress_cb(0.05, f"Downloading {ticker} history from Yahoo Finance...")

    try:
        raw = yf.download(
            ticker,
            start="2000-01-01",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty or len(raw) < 200:
            return False, {"error": f"Insufficient data for {ticker} ({len(raw)} rows)"}

        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        df.index.name = "date"
        df["ticker"] = ticker

        raw_path = DATA_RAW / f"{ticker}.parquet"
        df.to_parquet(raw_path)

        # Get basic metadata
        info = {}
        try:
            t_info = yf.Ticker(ticker).info
            info = {
                "name":   t_info.get("longName", ticker),
                "sector": t_info.get("sector", "Unknown"),
                "type":   t_info.get("quoteType", "EQUITY"),
                "beta":   t_info.get("beta", 1.0),
            }
        except Exception:
            info = {"name": ticker, "sector": "Unknown", "type": "EQUITY", "beta": 1.0}

        log.info(
            f"  ✓ Downloaded {ticker}: {len(df):,} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return True, {
            "rows":  len(df),
            "start": str(df.index[0].date()),
            "end":   str(df.index[-1].date()),
            **info,
        }

    except Exception as e:
        return False, {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(ticker: str, progress_cb=None) -> pd.DataFrame | None:
    """Build full 121-col feature store for ticker."""
    if progress_cb:
        progress_cb(0.20, f"Building features for {ticker}...")

    try:
        macro = load_macro_panel()
        df    = build_ticker_features(ticker, macro, save=True)
        if df is None or len(df) < 100:
            return None
        log.info(f"  ✓ Features: {len(df):,} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        log.warning(f"  Feature build error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def _get_scale_pos_weight(feat_df: pd.DataFrame) -> float:
    """
    Calculate class imbalance weight from actual data.
    Higher beta/vol tickers have lower optimal day frequency.
    """
    window = feat_df[feat_df.index.day >= PURCHASE_WINDOW_START]
    if "is_optimal_buy_day" not in window.columns:
        return 20.0
    pos_rate = window["is_optimal_buy_day"].mean()
    if pos_rate > 0:
        return round((1 - pos_rate) / pos_rate, 1)
    return 20.0


def _train_model(
    ticker: str,
    feat_df: pd.DataFrame,
    ticker_meta: dict,
    progress_cb=None,
) -> dict | None:
    """
    Train LightGBM classifier + regressor.
    Adjusts hyperparameters based on ticker volatility.
    """
    if progress_cb:
        progress_cb(0.35, f"Training LightGBM for {ticker} (walk-forward ~100 folds)...")

    # Window data only
    window = feat_df[feat_df.index.day >= PURCHASE_WINDOW_START].copy()

    # Feature selection
    features = get_feature_list(ticker, feat_df.columns.tolist())

    # Adjust scale_pos_weight for this ticker's class imbalance
    spw = _get_scale_pos_weight(window)

    # Override params for this ticker
    clf_params = dict(CLASSIFIER_PARAMS)
    reg_params = dict(REGRESSOR_PARAMS)
    clf_params["scale_pos_weight"] = spw

    # High-volatility tickers: more regularization to avoid overfitting noise
    beta = float(ticker_meta.get("beta", 1.0) or 1.0)
    if beta > 1.5:
        clf_params["reg_alpha"]      = 0.3
        clf_params["reg_lambda"]     = 0.3
        clf_params["num_leaves"]     = 20   # smaller model for noisy tickers
        clf_params["min_child_samples"] = 30
        reg_params["reg_alpha"]      = 0.3
        reg_params["reg_lambda"]     = 0.3
        reg_params["num_leaves"]     = 20
        log.info(f"  {ticker}: high-beta ({beta:.1f}) — increased regularization")

    # Monkey-patch params into trainer module temporarily
    import src.models.trainer as _trainer
    orig_clf = _trainer.CLASSIFIER_PARAMS.copy()
    orig_reg = _trainer.REGRESSOR_PARAMS.copy()
    _trainer.CLASSIFIER_PARAMS.update(clf_params)
    _trainer.REGRESSOR_PARAMS.update(reg_params)

    try:
        wf = walk_forward_train(ticker, window, features)
        if not wf or wf["predictions"].empty:
            return None

        bt    = evaluate_backtest(wf["predictions"], ticker)
        final = train_final_model(ticker, window, features)
        if not final:
            return None

        bundle = {
            "ticker":           ticker,
            "classifier":       final["classifier"],
            "regressor":        final["regressor"],
            "features":         final["features"],
            "importance_clf":   final["importance_clf"],
            "importance_reg":   final["importance_reg"],
            "backtest":         {k: v for k, v in bt.items() if k != "results_df"} if bt else {},
            "wf_fold_metrics":  wf["fold_metrics"],
            "training_period":  final["training_period"],
            "n_training_rows":  final["n_training_rows"],
            "ticker_meta":      ticker_meta,
            "scale_pos_weight": spw,
        }

        with open(_model_path(ticker), "wb") as f:
            pickle.dump(bundle, f)

        if bt:
            log.info(
                f"  ✓ Model trained: win={bt.get('win_rate_pct',0):.1f}% | "
                f"save={bt.get('avg_saving_pct',0):+.3f}% | "
                f"capture={bt.get('avg_capture_rate',0):.1f}%"
            )
        return bundle

    finally:
        _trainer.CLASSIFIER_PARAMS.update(orig_clf)
        _trainer.REGRESSOR_PARAMS.update(orig_reg)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────

# ── Clipping bounds by beta — prevents model noise from inflating distributions
def _clip_bounds(beta: float) -> tuple[float, float]:
    """
    Returns (low_clip, high_clip) for monthly saving% based on asset volatility.
    Meme/extreme stocks get tighter clips because model output is noise there.
    """
    if beta > 3.0:   return (-8.0, 12.0)   # meme/extreme — cap the noise
    if beta > 2.0:   return (-10.0, 18.0)  # very high vol
    if beta > 1.5:   return (-8.0, 15.0)   # high vol (NVDA, ARKK)
    if beta > 1.0:   return (-4.0, 8.0)    # moderate vol (AAPL)
    if beta > 0.6:   return (-2.0, 5.0)    # broad market (VOO)
    return            (-0.5, 2.5)          # low vol (BND)


# ── Trust scoring — 5 factors → 0-100 score + plain-English reasons ──────────
def _compute_trust(
    win_rate: float,
    avg_save: float,
    capture_rate: float,
    beta: float,
    n_folds: int,
    p10: float,
    p90: float,
) -> dict:
    """
    Compute a trust score (0-100) and tier (A/B/C) with explanation.

    Five factors, each 0-20 points:
    F1  Win rate consistency       — how reliably does model beat day 27?
    F2  Capture rate               — does it find the ACTUAL best day?
    F3  Asset predictability       — is beta low enough for patterns to hold?
    F4  Distribution tightness     — are outcomes clustered (reliable) or wide?
    F5  Training data adequacy     — enough folds to trust the model?
    """
    reasons   = []
    penalties = []
    score     = 0

    # F1: Win rate (0-20)
    if win_rate >= 75:
        score += 20; reasons.append(f"high win rate ({win_rate:.0f}%)")
    elif win_rate >= 65:
        score += 14; reasons.append(f"moderate win rate ({win_rate:.0f}%)")
    elif win_rate >= 55:
        score += 7;  penalties.append(f"low win rate ({win_rate:.0f}%) — barely above random")
    else:
        score += 0;  penalties.append(f"win rate {win_rate:.0f}% ≈ random — no reliable signal")

    # F2: Capture rate (0-20)
    if capture_rate >= 45:
        score += 20; reasons.append(f"strong capture ({capture_rate:.0f}%)")
    elif capture_rate >= 30:
        score += 12; reasons.append(f"moderate capture ({capture_rate:.0f}%)")
    elif capture_rate >= 15:
        score += 5;  penalties.append(f"low capture ({capture_rate:.0f}%) — wins but misses exact dip")
    else:
        score += 0;  penalties.append(f"very low capture ({capture_rate:.0f}%) — timing not learnable")

    # F3: Asset predictability (0-20)
    if beta <= 0.8:
        score += 20; reasons.append(f"low beta ({beta:.1f}) — stable patterns")
    elif beta <= 1.2:
        score += 15; reasons.append(f"moderate beta ({beta:.1f}) — predictable enough")
    elif beta <= 1.8:
        score += 8;  penalties.append(f"high beta ({beta:.1f}) — volatile, patterns weaker")
    elif beta <= 2.5:
        score += 3;  penalties.append(f"very high beta ({beta:.1f}) — macro-disconnected")
    else:
        score += 0;  penalties.append(f"extreme beta ({beta:.1f}) — likely meme/news-driven, model unreliable")

    # F4: Distribution tightness (0-20)
    spread = p90 - p10
    if spread < 3:
        score += 20; reasons.append(f"tight distribution (p10-p90 range {spread:.1f}pp)")
    elif spread < 6:
        score += 12; reasons.append(f"moderate spread ({spread:.1f}pp)")
    elif spread < 12:
        score += 5;  penalties.append(f"wide spread ({spread:.1f}pp) — outcomes vary a lot month to month")
    else:
        score += 0;  penalties.append(f"very wide spread ({spread:.1f}pp) — high variance, results unreliable")

    # F5: Training adequacy (0-20)
    if n_folds >= 150:
        score += 20; reasons.append(f"well-trained ({n_folds} folds)")
    elif n_folds >= 80:
        score += 13; reasons.append(f"adequately trained ({n_folds} folds)")
    elif n_folds >= 40:
        score += 6;  penalties.append(f"limited training ({n_folds} folds) — newer ticker, less history")
    else:
        score += 0;  penalties.append(f"insufficient training ({n_folds} folds) — results provisional")

    score = max(0, min(100, score))

    # Tier from score
    if score >= 65:
        tier = "A"
        tier_label = "High precision — act on model recommendation"
    elif score >= 40:
        tier = "B"
        tier_label = "Medium precision — use as context, not instruction"
    else:
        tier = "C"
        tier_label = "Low precision — timing not reliably learnable for this asset"

    return {
        "trust_score":  score,
        "tier":         tier,
        "tier_label":   tier_label,
        "reasons":      reasons,
        "penalties":    penalties,
        "summary_text": (
            f"Trust {score}/100. " +
            (f"Strengths: {'; '.join(reasons[:2])}. " if reasons else "") +
            (f"Concerns: {'; '.join(penalties[:2])}." if penalties else "")
        ),
    }


# ── Tier-aware compounding projections ───────────────────────────────────────
def _tier_adjusted_projections(
    monthly_usd: float,
    avg_save: float,
    trust: dict,
    beta: float,
) -> dict:
    """
    Compounding projections adjusted for trust level.

    Tier A: standard 15% base, 45% capture
    Tier B: uncertain base (10-15%), lower capture (25%)
    Tier C: show conservative only + explicit warning

    High beta (>2.0): base rate itself uncertain — show range not point
    """
    tier = trust["tier"]

    # Base rate scenarios by tier
    if tier == "A":
        scenarios = [
            ("Conservative", 0.10, 0.25),
            ("Base case",    0.15, 0.45),
            ("Optimistic",   0.15, 1.00),
        ]
        show_optimistic = True
    elif tier == "B":
        scenarios = [
            ("Conservative", 0.08, 0.15),
            ("Base case",    0.12, 0.25),
            ("Optimistic",   0.15, 0.45),
        ]
        show_optimistic = True
    else:  # Tier C
        scenarios = [
            ("Conservative", 0.05, 0.10),
            ("Base case",    0.10, 0.15),
        ]
        show_optimistic = False  # don't show optimistic for Tier C

    projections_by_scenario = {}
    for label, base_r, capture in scenarios:
        opt_r = base_r + (avg_save / 100 * capture * 12 * 0.3)
        scenario_proj = {}
        for years in [1, 3, 5, 10, 20]:
            n    = years * 12
            r_b  = base_r / 12
            r_o  = opt_r  / 12
            fv_b = monthly_usd * ((1 + r_b) ** n - 1) / r_b
            fv_o = monthly_usd * ((1 + r_o) ** n - 1) / r_o
            scenario_proj[years] = {
                "fixed":     round(fv_b),
                "optimised": round(fv_o),
                "extra":     round(fv_o - fv_b),
                "lift_pct":  round((fv_o - fv_b) / fv_b * 100, 1),
            }
        projections_by_scenario[label] = {
            "base_rate":    round(base_r * 100, 1),
            "capture_rate": round(capture * 100, 0),
            "opt_rate":     round(opt_r * 100, 2),
            "projections":  scenario_proj,
        }

    # Keep backward-compatible "projections" key using base case
    base_case = projections_by_scenario.get("Base case", {})
    base_proj  = base_case.get("projections", {})
    simple_proj = {yr: {
        "fixed":     base_proj[yr]["fixed"],
        "optimised": base_proj[yr]["optimised"],
        "extra":     base_proj[yr]["extra"],
    } for yr in base_proj}

    return {
        "projections":             simple_proj,          # backward compat
        "projections_by_scenario": projections_by_scenario,
        "show_optimistic":         show_optimistic,
        "projection_note": (
            "⚠️  Tier C: projections use conservative assumptions. "
            "This asset's timing signal is weak — treat with caution."
            if tier == "C" else
            "⚠️  Tier B: base rate assumption is uncertain for volatile assets."
            if tier == "B" else ""
        ),
    }


def _run_simulation(
    ticker: str,
    model_bundle: dict,
    monthly_usd: float,
    n_runs: int = 1000,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """
    Run Monte Carlo simulation using real ticker data + trained model.
    Applies clipping, trust scoring, and tier-adjusted projections.
    """
    if progress_cb:
        progress_cb(0.65, f"Running {n_runs} Monte Carlo scenarios for {ticker}...")

    local_sampler = ScenarioSampler()
    scenarios     = local_sampler.sample(ticker, n=n_runs, seed=seed)
    rng           = np.random.default_rng(seed)
    outcomes      = []

    for scenario in scenarios:
        try:
            result_a   = agent_a(scenario)
            result_b   = agent_b(scenario, model_bundle, rng)
            comparison = compare_agents(result_a, result_b, monthly_usd)
            outcomes.append(comparison)
        except Exception:
            continue

    if not outcomes:
        return {"error": "No valid outcomes"}

    df   = pd.DataFrame(outcomes)
    beta = float(model_bundle.get("ticker_meta", {}).get("beta", 1.0) or 1.0)

    # ── Clip extreme saving values ────────────────────────────────────────
    # Prevents model noise from inflating distribution for volatile assets
    lo, hi = _clip_bounds(beta)
    raw_savings   = df["saving_pct"].copy()
    df["saving_pct"] = df["saving_pct"].clip(lo, hi)
    n_clipped     = (raw_savings != df["saving_pct"]).sum()
    if n_clipped > 0:
        log.info(f"  {ticker}: clipped {n_clipped} extreme saving values "
                 f"to [{lo:.1f}%, +{hi:.1f}%] bounds")

    win_rate = df["agent_b_wins"].mean() * 100
    avg_save = df["saving_pct"].mean()

    p = {
        "p10": round(float(np.percentile(df["saving_pct"], 10)), 3),
        "p25": round(float(np.percentile(df["saving_pct"], 25)), 3),
        "p50": round(float(np.percentile(df["saving_pct"], 50)), 3),
        "p75": round(float(np.percentile(df["saving_pct"], 75)), 3),
        "p90": round(float(np.percentile(df["saving_pct"], 90)), 3),
    }

    # ── Trust scoring ─────────────────────────────────────────────────────
    bt      = model_bundle.get("backtest", {})
    n_folds = len(model_bundle.get("wf_fold_metrics", []))

    # avg_capture_rate from trainer.py is now the median capture over
    # positive-opportunity months (> 0.1pp spread), clipped to [-100,100].
    # Guard: if the stored value is still negative or near-zero (old pkl),
    # treat as 0 so the trust score doesn't unfairly penalise the ticker.
    raw_cap = bt.get("avg_capture_rate", 0) or 0
    cap     = max(0.0, float(raw_cap))   # floor at 0 — negative = no credit, not penalty
    trust        = _compute_trust(
        win_rate, avg_save, cap, beta, n_folds, p["p10"], p["p90"]
    )

    # ── Tier-adjusted projections ─────────────────────────────────────────
    proj_data = _tier_adjusted_projections(monthly_usd, avg_save, trust, beta)

    # ── Regime breakdown ──────────────────────────────────────────────────
    regime_stats = {}
    for regime in ["calm", "normal", "elevated", "fear"]:
        sub = df[df["vix_regime"] == regime]
        if len(sub) >= 5:
            regime_stats[regime] = {
                "n_runs":      len(sub),
                "win_rate":    round(sub["agent_b_wins"].mean() * 100, 1),
                "avg_saving":  round(sub["saving_pct"].mean(), 3),
                "pct_of_runs": round(len(sub) / len(df) * 100, 1),
            }

    summary = {
        "ticker":               ticker,
        "monthly_usd":          monthly_usd,
        "n_runs":               len(df),
        # Trust and tier
        "tier":                 trust["tier"],
        "tier_label":           trust["tier_label"],
        "trust_score":          trust["trust_score"],
        "trust_reasons":        trust["reasons"],
        "trust_penalties":      trust["penalties"],
        "trust_summary":        trust["summary_text"],
        # Performance
        "win_rate_pct":         round(win_rate, 1),
        "avg_saving_pct":       round(avg_save, 3),
        "median_saving_pct":    round(float(p["p50"]), 3),
        "percentiles":          p,
        "clipped_values":       int(n_clipped),
        "clip_bounds":          {"low": lo, "high": hi},
        # Dollar impact
        "avg_dollar_per_month": round(float(df["dollar_advantage"].mean()), 4),
        "avg_dollar_per_year":  round(float(df["dollar_advantage"].mean()) * 12, 2),
        # Projections
        "projections":             proj_data["projections"],
        "projections_by_scenario": proj_data["projections_by_scenario"],
        "show_optimistic":         proj_data["show_optimistic"],
        "projection_note":         proj_data["projection_note"],
        # Regime
        "regime_stats":         regime_stats,
        "confidence_stats":     {},
        # Metadata
        "backtest":             bt,
        "ticker_meta":          model_bundle.get("ticker_meta", {}),
        "is_portfolio_ticker":  False,
        "is_proxy_data":        False,
    }

    log.info(
        f"  ✓ Simulation: win={win_rate:.1f}% | p50={p['p50']:+.2f}% | "
        f"tier={trust['tier']} | trust={trust['trust_score']}/100"
    )
    return {"summary": summary, "outcomes": df}


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: LLM advisory
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_advisory(
    ticker: str,
    model_bundle: dict,
    feat_df: pd.DataFrame,
    progress_cb=None,
) -> dict:
    """Get LLM advisory for current month."""
    if progress_cb:
        progress_cb(0.88, f"Fetching LLM advisory for {ticker}...")

    try:
        from src.llm.groq_advisor import (
            build_macro_context,
            build_ticker_context,
            generate_advisory,
        )
        from src.models.trainer import predict_optimal_day

        macro_ctx  = build_macro_context()
        ticker_ctx = build_ticker_context(ticker)

        # Get current window
        today  = pd.Timestamp.today()
        window = feat_df[
            (feat_df.index >= today.replace(day=1)) &
            (feat_df.index.day >= PURCHASE_WINDOW_START)
        ]
        if len(window) < 3:
            window = feat_df[
                feat_df.index >= today - pd.Timedelta(days=30)
            ]
            window = window[window.index.day >= PURCHASE_WINDOW_START]

        lgbm_rec = {}
        if len(window) > 0:
            lgbm_rec = predict_optimal_day(ticker, model_bundle, window)

        advisory = generate_advisory(ticker, lgbm_rec, macro_ctx, ticker_ctx)
        return {
            "lgbm_rec":  lgbm_rec,
            "advisory":  advisory,
            "macro_ctx": macro_ctx,
        }
    except Exception as e:
        log.warning(f"  LLM advisory failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_on_demand(
    ticker: str,
    monthly_usd: float,
    horizon_years: int = 10,
    n_runs: int = 1000,
    force_retrain: bool = False,
    progress_cb=None,   # callable(fraction: float, message: str)
) -> dict:
    """
    Run the full pipeline for any ticker.

    Args:
        ticker:        any Yahoo Finance ticker
        monthly_usd:   monthly investment amount
        horizon_years: projection horizon for compounding chart
        n_runs:        Monte Carlo runs (default 1000)
        force_retrain: ignore cache and retrain from scratch
        progress_cb:   optional callback(fraction, message) for Streamlit progress

    Returns:
        Full result dict compatible with the simulator UI
    """
    ticker = ticker.upper().strip()
    t0     = time.time()

    log.info(f"\n{'=' * 55}")
    log.info(f"  ON-DEMAND PIPELINE: {ticker}")
    log.info(f"  ${monthly_usd}/mo | {n_runs} runs | {horizon_years}yr horizon")
    log.info(f"{'=' * 55}")

    # Check cache
    if not force_retrain and _cache_valid(ticker):
        log.info(f"  Using cached result (< {CACHE_MAX_AGE_DAYS}d old)")
        cached = _load_cache(ticker)
        if cached:
            if progress_cb:
                progress_cb(1.0, f"Loaded from cache ✓")
            return cached

    # Step 1: Download
    if progress_cb:
        progress_cb(0.02, f"Starting pipeline for {ticker}...")

    raw_path = DATA_RAW / f"{ticker}.parquet"
    if force_retrain or not raw_path.exists():
        success, ticker_meta = _download_ticker(ticker, progress_cb)
        if not success:
            return {
                "error":   ticker_meta.get("error", "Download failed"),
                "ticker":  ticker,
                "summary": None,
            }
    else:
        log.info(f"  Raw data cached, loading...")
        ticker_meta = {"name": ticker, "sector": "Unknown",
                       "type": "EQUITY", "beta": 1.0}
        try:
            t_info = yf.Ticker(ticker).info
            ticker_meta = {
                "name":   t_info.get("longName", ticker),
                "sector": t_info.get("sector", "Unknown"),
                "type":   t_info.get("quoteType", "EQUITY"),
                "beta":   t_info.get("beta", 1.0),
            }
        except Exception:
            pass

    # Step 2: Features
    feat_df = None
    feat_path = DATA_FEAT / f"{ticker}.parquet"
    if not force_retrain and feat_path.exists():
        log.info(f"  Features cached, loading...")
        feat_df = pd.read_parquet(feat_path)
        feat_df.index = pd.to_datetime(feat_df.index)
        if progress_cb:
            progress_cb(0.30, f"Features loaded from cache ({len(feat_df):,} rows)")
    else:
        feat_df = _build_features(ticker, progress_cb)
        if feat_df is None:
            return {
                "error":   f"Could not build features for {ticker}",
                "ticker":  ticker,
                "summary": None,
            }

    # Step 3: Train model
    model_bundle = None
    if not force_retrain and _model_valid(ticker):
        log.info(f"  Model cached, loading...")
        with open(_model_path(ticker), "rb") as f:
            model_bundle = pickle.load(f)
        if progress_cb:
            progress_cb(0.60, f"Model loaded from cache ✓")
    else:
        model_bundle = _train_model(ticker, feat_df, ticker_meta, progress_cb)
        if model_bundle is None:
            return {
                "error":   f"Model training failed for {ticker}",
                "ticker":  ticker,
                "summary": None,
            }

    # Step 4: Monte Carlo
    sim_result = _run_simulation(
        ticker, model_bundle, monthly_usd,
        n_runs=n_runs, progress_cb=progress_cb,
    )
    if "error" in sim_result:
        return {
            "error":   sim_result["error"],
            "ticker":  ticker,
            "summary": None,
        }

    # Step 5: LLM advisory
    llm_data = _get_llm_advisory(ticker, model_bundle, feat_df, progress_cb)

    if progress_cb:
        progress_cb(0.95, "Assembling results...")

    # Enrich summary
    s = sim_result["summary"]
    s["ticker_meta"]       = ticker_meta
    s["llm_advisory"]      = llm_data.get("advisory", {})
    s["lgbm_rec"]          = llm_data.get("lgbm_rec", {})
    s["projection_years"]  = horizon_years

    elapsed = time.time() - t0

    final_result = {
        "ticker":     ticker,
        "summary":    s,
        "outcomes":   sim_result.get("outcomes", pd.DataFrame()).to_dict(orient="records"),
        "elapsed_s":  round(elapsed, 1),
        "generated":  datetime.now().isoformat(),
        "cached":     False,
    }

    _save_cache(ticker, {k: v for k, v in final_result.items() if k != "outcomes"})

    if progress_cb:
        progress_cb(1.0, f"Done in {elapsed:.0f}s ✓")

    log.info(f"\n  Pipeline complete in {elapsed:.1f}s")
    log.info(f"  Result cached at: {_cache_path(ticker)}")
    return final_result
