"""
TemporalEdge — Monthly Refresh Script
Runs on the 18th of each month via GitHub Actions.
Generates the execution plan and caches it as JSON.
Dashboard reads from cache — near-instant load.

Run manually:
  python scripts/monthly_refresh.py

Env vars required:
  GROQ_API_KEY — Groq API key (free tier)
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PORTFOLIO, PURCHASE_WINDOW_START, ROOT
from src.models.trainer import load_model_for_prediction, predict_optimal_day
from src.features.engineer import load_macro_panel
from src.llm.groq_advisor import build_macro_context, generate_portfolio_plan
from src.utils.logger import get_logger

log = get_logger("scripts.monthly_refresh")

CACHE_DIR  = ROOT / "dashboard" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "execution_plan.json"


def get_current_window(ticker: str) -> pd.DataFrame | None:
    """
    Get the purchase window for the current month, projected to the full
    18-31 range even if we are mid-month.

    Problem: running on April 22nd means only days 18-22 exist.
    The model scores only those days and clusters around today.
    Day 21 (yesterday) always wins because it has the latest feature values.

    Solution: project the remaining window days (23-31) using the
    equivalent days from LAST month as feature proxies.
    This gives the model a full 14-day window to score, producing
    realistic recommendations that include future days.

    This is conceptually sound: the model learned patterns from
    historical monthly windows. Last month's day-25 features are a
    reasonable proxy for this month's day-25 features.
    """
    from src.config import DATA_FEAT
    path = DATA_FEAT / f"{ticker}.parquet"
    if not path.exists():
        log.warning(f"  {ticker}: feature file not found")
        return None

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    today       = pd.Timestamp.today()
    month_start = today.replace(day=1)
    month_end   = month_start + pd.offsets.MonthEnd(1)

    # Current month days we actually have
    current = df[
        (df.index >= month_start) &
        (df.index <= today) &
        (df.index.day >= PURCHASE_WINDOW_START)
    ].copy()

    # Days still missing (today+1 through month end)
    existing_days = set(current.index.day)
    all_window_days = set(range(PURCHASE_WINDOW_START, int(month_end.day) + 1))
    missing_days = sorted(all_window_days - existing_days)

    if missing_days and len(current) > 0:
        # Pull equivalent days from last month as proxies for future days
        last_month_start = (month_start - pd.DateOffset(months=1))
        last_month_end   = month_start - pd.Timedelta(days=1)

        last_month = df[
            (df.index >= last_month_start) &
            (df.index <= last_month_end) &
            (df.index.day >= PURCHASE_WINDOW_START)
        ].copy()

        if len(last_month) > 0:
            proxies = []
            for day in missing_days:
                proxy_rows = last_month[last_month.index.day == day]
                if len(proxy_rows) == 0:
                    # Try closest available day from last month
                    for offset in [1, -1, 2, -2, 3]:
                        proxy_rows = last_month[last_month.index.day == day + offset]
                        if len(proxy_rows) > 0:
                            break
                if len(proxy_rows) > 0:
                    proxy = proxy_rows.iloc[[-1]].copy()
                    # Restamp the index to this month's date
                    try:
                        new_date = today.replace(day=day)
                        proxy.index = pd.DatetimeIndex([new_date])
                    except Exception:
                        continue
                    proxies.append(proxy)

            if proxies:
                projected = pd.concat(proxies)
                current = pd.concat([current, projected]).sort_index()
                log.info(
                    f"  {ticker}: window projected {len(proxies)} future days "
                    f"(days {missing_days[0]}-{missing_days[-1]}) from last month"
                )

    if len(current) >= 3:
        return current

    # Final fallback: last 30 days of data
    log.info(f"  {ticker}: using last 30 days as fallback window")
    window = df[df.index >= today - pd.Timedelta(days=30)].copy()
    return window[window.index.day >= PURCHASE_WINDOW_START] if len(window) > 0 else None


def compute_execution_tier(
    ticker: str,
    lgbm_rec: dict,
    macro_ctx: dict,
    advisory: dict,
) -> dict:
    """
    Compute the final execution tier for a ticker this month.
    Combines: model tier + VIX regime + confidence + LLM action.

    Returns:
        tier:    'green' / 'amber' / 'grey'
        score:   0-10 (for display)
        reason:  one-line explanation
    """
    from src.simulation.agents import TICKER_TIERS

    model_tier  = TICKER_TIERS.get(ticker, "B")
    conf        = lgbm_rec.get("confidence", 0.5)
    vix         = macro_ctx.get("vix", 18)
    llm_action  = advisory.get("action", "consider")
    precision   = lgbm_rec.get("timing_precision", "window")

    # Base score from model tier
    score = {"A": 5, "B": 3, "C": 1}[model_tier]

    # VIX adjustment
    if vix >= 25:
        score += 2    # high volatility = more timing value
    elif vix >= 20:
        score += 1
    elif vix < 15:
        score -= 1    # calm market = timing adds less

    # Confidence adjustment
    if conf >= 0.70:
        score += 2
    elif conf >= 0.55:
        score += 1
    else:
        score -= 1

    # LLM alignment
    if llm_action == "act":
        score += 1
    elif llm_action == "skip":
        score -= 2

    # Hard overrides
    if ticker == "TSLA" and vix < 20:
        score = min(score, 3)   # TSLA only useful in high-VIX
    if ticker == "VTI":
        score = min(score, 6)   # VTI cap — proxy model, not native

    score = max(0, min(10, score))

    # Map to tier
    if score >= 7:
        tier   = "green"
        reason = f"Strong signal — model tier {model_tier}, {conf:.0%} confidence, {macro_ctx.get('vix_regime','')}"
    elif score >= 4:
        tier   = "amber"
        reason = f"Mixed signal — review LLM context before acting"
    else:
        tier   = "grey"
        reason = f"Weak signal this month — keep day 27 schedule"

    return {"tier": tier, "score": score, "reason": reason}


def run_monthly_refresh() -> dict:
    """
    Main function: compute full execution plan for all portfolio tickers.
    Saves to cache/execution_plan.json.
    """
    log.info("=" * 60)
    log.info("  TEMPORALEDGE — MONTHLY REFRESH")
    log.info(f"  {datetime.now().strftime('%B %d, %Y %H:%M')}")
    log.info("=" * 60)

    t0 = time.time()

    # Step 0: Refresh feature store so current month data is available
    # This ensures days 18-22 of this month are in the parquet files
    # Without this, the window filter falls back to "last 30 days"
    log.info("\n[0/4] Refreshing feature store (fetching latest prices)...")
    try:
        from src.data.collector import run_collection
        from src.features.engineer import build_all_features
        run_collection(force_refresh=False)   # only downloads new data since last run
        build_all_features()
        log.info("  ✓ Feature store refreshed")
    except Exception as e:
        log.warning(f"  Feature refresh failed ({e}) — using existing data")

    # Step 1: Current macro context
    log.info("\n[1/4] Fetching macro context...")
    macro_ctx = build_macro_context()
    log.info(f"  VIX={macro_ctx.get('vix','?')} ({macro_ctx.get('vix_regime','?')})")
    log.info(f"  S&P500 5d={macro_ctx.get('sp500_5d_chg',0):+.1f}%  "
             f"USD 5d={macro_ctx.get('usd_index_5d_chg',0):+.1f}%  "
             f"Yield curve={macro_ctx.get('yield_curve','?')}")

    # Step 2: LightGBM recommendations for each ticker
    log.info("\n[2/4] Running LightGBM models...")

    # Load Monte Carlo avg_save% to display as expected saving
    # (live regressor predicted_saving is ~0% mid-month because day 27 is in the future)
    mc_avg_save = {}
    try:
        import json as _json
        mc_path = ROOT / "results" / "portfolio_simulation.json"
        if mc_path.exists():
            with open(mc_path) as f:
                mc_data = _json.load(f)
            for t, d in mc_data.items():
                mc_avg_save[t] = d.get("summary", {}).get("avg_saving_pct", 0.0)
    except Exception:
        pass

    lgbm_recs = {}
    for ticker in PORTFOLIO:
        model  = load_model_for_prediction(ticker)
        window = get_current_window(ticker)

        if model is None or window is None or len(window) == 0:
            log.warning(f"  {ticker}: skipping — no model or window data")
            continue

        rec = predict_optimal_day(ticker, model, window)

        # Replace live predicted_saving (near-zero mid-month) with
        # historically-validated Monte Carlo avg_save%
        rec["predicted_saving"] = mc_avg_save.get(ticker, rec.get("predicted_saving", 0.0))

        lgbm_recs[ticker] = rec
        log.info(
            f"  {ticker:6s}: day {rec['recommended_day']:2d} | "
            f"conf={rec['confidence']:.0%} | "
            f"vix_trust={rec.get('vix_trust','?')} | "
            f"top3={rec.get('top_3_days',[])} | "
            f"precision={rec.get('timing_precision','?')} | "
            f"mc_save={rec['predicted_saving']:+.2f}%"
        )

    # Step 3: LLM advisories
    log.info("\n[3/4] Generating LLM advisories (Groq)...")
    plan_raw = generate_portfolio_plan(lgbm_recs, macro_ctx)

    # Step 4: Compute execution tiers
    log.info("\n[4/4] Computing execution tiers...")
    execution_plan = {}
    tier_counts    = {"green": 0, "amber": 0, "grey": 0}

    for ticker, data in plan_raw.items():
        exec_tier = compute_execution_tier(
            ticker,
            data["lgbm"],
            macro_ctx,
            data["advisory"],
        )
        monthly_usd = PORTFOLIO[ticker]["monthly_usd"]

        execution_plan[ticker] = {
            "ticker":        ticker,
            "name":          PORTFOLIO[ticker]["name"],
            "monthly_usd":   monthly_usd,
            "model_tier":    data["lgbm"].get("timing_precision", "window"),
            "recommended_day": data["lgbm"].get("recommended_day", 27),
            "top_3_days":    data["lgbm"].get("top_3_days", []),
            "confidence":    data["lgbm"].get("confidence", 0.5),
            "vix_trust":     data["lgbm"].get("vix_trust", "low"),
            "predicted_saving": data["lgbm"].get("predicted_saving", 0.0),
            "advisory":      data["advisory"].get("advisory", ""),
            "suggested_window": data["advisory"].get("suggested_window", ""),
            "llm_action":    data["advisory"].get("action", "consider"),
            "key_factor":    data["advisory"].get("key_factor", ""),
            "exec_tier":     exec_tier["tier"],
            "exec_score":    exec_tier["score"],
            "exec_reason":   exec_tier["reason"],
            "ticker_ctx":    data.get("ticker_ctx", {}),
        }
        tier_counts[exec_tier["tier"]] += 1

    # Build final cache object
    cache = {
        "generated_at":  datetime.now().isoformat(),
        "month":         datetime.now().strftime("%B %Y"),
        "macro_context": macro_ctx,
        "execution_plan": execution_plan,
        "summary": {
            "green_count":  tier_counts["green"],
            "amber_count":  tier_counts["amber"],
            "grey_count":   tier_counts["grey"],
            "vix":          macro_ctx.get("vix", 0),
            "vix_regime":   macro_ctx.get("vix_regime", "normal"),
        },
    }

    # Save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"  REFRESH COMPLETE in {elapsed:.1f}s")
    log.info(f"  Green: {tier_counts['green']} | Amber: {tier_counts['amber']} | Grey: {tier_counts['grey']}")
    log.info(f"  Cached: {CACHE_FILE}")
    log.info("=" * 60)

    return cache


if __name__ == "__main__":
    run_monthly_refresh()