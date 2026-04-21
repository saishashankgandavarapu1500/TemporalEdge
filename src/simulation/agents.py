"""
TemporalEdge — Simulation Agents  (Phase 4, Step 2)

Agent A: Fixed Day 27  — your current Robinhood recurring schedule
Agent B: AI-Optimised  — model recommendation with tier-aware precision

The controlled experiment: same scenario, same macro context,
same market environment. Only the entry date changes.
This isolates the pure timing effect.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config import YOUR_CURRENT_DAY, PURCHASE_WINDOW_START
from src.utils.logger import get_logger

log = get_logger("simulation.agents")

# Tier definitions from Phase 3 capture rate analysis
# Tier A ≥45% capture → exact day recommendation (±1 day uncertainty)
# Tier B 30-45%       → window recommendation   (±2 day uncertainty)
# Tier C <30%         → loose recommendation    (±3 day uncertainty, random within window)
TICKER_TIERS = {
    "VOO":  "A",  # 57.8% capture
    "VYM":  "A",  # 55.2%
    "BND":  "A",  # 49.9%
    "SCHD": "A",  # 49.6%
    "AAPL": "A",  # 48.7%
    "VEA":  "A",  # 48.6%
    "VWO":  "B",  # 43.1%
    "NVDA": "B",  # 38.3%
    "VXUS": "B",  # 36.9%
    "VTI":  "C",  # 28.7%  (post-2010 data issue)
    "TSLA": "C",  #  7.9%  (chaotic intra-month moves)
}

TIER_UNCERTAINTY_DAYS = {"A": 1, "B": 2, "C": 3}


def get_day27_price(window_df: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    """
    Get the price on day 27 (or closest available trading day).
    Returns (price, actual_date).
    """
    day27_rows = window_df[window_df.index.day == YOUR_CURRENT_DAY]

    if len(day27_rows) == 0:
        # Day 27 is weekend/holiday — use closest day >= 25
        candidates = window_df[window_df.index.day >= 25]
        if len(candidates) == 0:
            candidates = window_df
        day27_rows = candidates.iloc[[0]]

    ts  = day27_rows.index[0]
    px  = float(day27_rows["close"].values[0])
    return px, ts


def agent_a(scenario: dict) -> dict:
    """
    Agent A: Always buys on day 27 regardless of conditions.
    This is your current Robinhood recurring investment.
    """
    window_df = scenario["window_df"]
    price, date = get_day27_price(window_df)

    return {
        "agent":          "A",
        "strategy":       "fixed_day27",
        "entry_day":      date.day,
        "entry_date":     str(date.date()),
        "entry_price":    round(price, 4),
        "period":         scenario["period"],
        "ticker":         scenario["ticker"],
    }


def agent_b(
    scenario: dict,
    model_bundle: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Agent B: Uses LightGBM model to recommend the optimal entry day.
    Applies tier-aware precision from Phase 3 capture rate analysis.

    Tier A (VOO, AAPL etc.): trusts the exact recommended day ±1 day
    Tier B (NVDA, VWO etc.): treats recommendation as ±2 day window
    Tier C (TSLA, VTI):      samples randomly from early purchase window
    """
    window_df = scenario["window_df"]
    ticker    = scenario["ticker"]
    tier      = TICKER_TIERS.get(ticker, "B")
    uncertainty = TIER_UNCERTAINTY_DAYS[tier]

    # ── Get model recommendation ──────────────────────────────────────────
    try:
        from src.models.trainer import predict_optimal_day
        reasoning = predict_optimal_day(ticker, model_bundle, window_df)
        recommended_day = reasoning["recommended_day"]
        confidence      = reasoning["confidence"]
        vix_regime      = reasoning.get("vix_regime", "normal")
        predicted_saving = reasoning.get("predicted_saving", 0.0)
    except Exception as e:
        log.debug(f"  Model prediction failed ({e}), using day 27 fallback")
        recommended_day  = YOUR_CURRENT_DAY
        confidence       = 0.0
        vix_regime       = scenario.get("regime", "normal")
        predicted_saving = 0.0

    # ── Tier-aware entry selection ─────────────────────────────────────────
    if tier == "C":
        # TSLA / VTI: model can't pinpoint exact day
        # Sample randomly from days earlier than day 27 in the window
        early_window = window_df[window_df.index.day < YOUR_CURRENT_DAY]
        if len(early_window) == 0:
            early_window = window_df
        chosen_row = early_window.iloc[rng.integers(0, len(early_window))]

    elif tier == "B":
        # NVDA, VWO, VXUS: recommendation is directionally right but ±2 days
        # Sample from days within ±uncertainty of recommendation
        lo = max(PURCHASE_WINDOW_START, recommended_day - uncertainty)
        hi = min(31, recommended_day + uncertainty)
        candidates = window_df[
            (window_df.index.day >= lo) & (window_df.index.day <= hi)
        ]
        if len(candidates) == 0:
            candidates = window_df
        chosen_row = candidates.iloc[rng.integers(0, len(candidates))]

    else:
        # Tier A: trust the exact recommended day ±1 day
        lo = max(PURCHASE_WINDOW_START, recommended_day - uncertainty)
        hi = min(31, recommended_day + uncertainty)
        candidates = window_df[
            (window_df.index.day >= lo) & (window_df.index.day <= hi)
        ]
        if len(candidates) == 0:
            candidates = window_df
        # Weight candidates by ensemble score so best day is most likely chosen
        if "ensemble_score" in window_df.columns:
            scores = candidates["ensemble_score"].values
            scores = np.clip(scores, 0, None)
            total  = scores.sum()
            probs  = scores / total if total > 0 else np.ones(len(candidates)) / len(candidates)
        else:
            probs = np.ones(len(candidates)) / len(candidates)
        idx = rng.choice(len(candidates), p=probs)
        chosen_row = candidates.iloc[idx]

    entry_price = float(chosen_row["close"])
    entry_day   = chosen_row.name.day
    entry_date  = str(chosen_row.name.date())

    return {
        "agent":            "B",
        "strategy":         "ai_optimised",
        "tier":             tier,
        "entry_day":        entry_day,
        "entry_date":       entry_date,
        "entry_price":      round(entry_price, 4),
        "recommended_day":  recommended_day,
        "confidence":       round(confidence, 3),
        "vix_regime":       vix_regime,
        "predicted_saving": round(predicted_saving, 3),
        "period":           scenario["period"],
        "ticker":           ticker,
    }


def compare_agents(
    result_a: dict,
    result_b: dict,
    monthly_usd: float,
) -> dict:
    """
    Compare Agent A vs Agent B for one scenario.
    Returns the outcome: who won, by how much, in dollars.
    """
    price_a = result_a["entry_price"]
    price_b = result_b["entry_price"]

    # Saving: positive = Agent B bought cheaper
    saving_pct = (price_a - price_b) / price_a * 100

    # Extra shares bought with same dollar amount
    shares_a = monthly_usd / price_a
    shares_b = monthly_usd / price_b
    extra_shares = shares_b - shares_a

    # Dollar value of extra shares (at entry price — forward returns apply equally)
    dollar_advantage = extra_shares * price_b

    return {
        "period":          result_a["period"],
        "ticker":          result_a["ticker"],
        "monthly_usd":     monthly_usd,
        "price_a":         round(price_a, 4),
        "price_b":         round(price_b, 4),
        "entry_day_a":     result_a["entry_day"],
        "entry_day_b":     result_b["entry_day"],
        "saving_pct":      round(saving_pct, 4),
        "extra_shares":    round(extra_shares, 6),
        "dollar_advantage":round(dollar_advantage, 4),
        "agent_b_wins":    saving_pct > 0,
        "confidence":      result_b.get("confidence", 0),
        "vix_regime":      result_b.get("vix_regime", "normal"),
        "tier":            result_b.get("tier", "B"),
        "recommended_day": result_b.get("recommended_day", YOUR_CURRENT_DAY),
    }
