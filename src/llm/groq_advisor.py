"""
TemporalEdge — Groq LLM Advisor  (Phase 5)
Calls Groq API (Llama 3, free tier) with current macro context
and LightGBM recommendation to produce a plain-English advisory.

Design principle: LLM is an advisor, not a decision-maker.
Output is always framed as context for the user's own judgement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import json
from datetime import datetime
import yfinance as yf
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("llm.advisor")

# Free tier model — fast, capable enough for macro reasoning
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client():
    """Lazy-load Groq client. Fails gracefully if key not set."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            log.warning("GROQ_API_KEY not set — LLM advisory unavailable")
            return None
        return Groq(api_key=api_key)
    except ImportError:
        log.warning("groq not installed — run: pip install groq")
        return None


def build_macro_context() -> dict:
    """
    Fetch current macro snapshot from yfinance + derived features.
    Returns structured dict used both for LLM prompt and display.
    """
    tickers = {
        "^VIX":    "vix",
        "^TNX":    "us10y_yield",
        "^IRX":    "us3m_yield",
        "GC=F":    "gold",
        "CL=F":    "oil_wti",
        "HG=F":    "copper",
        "DX-Y.NYB":"usd_index",
        "^GSPC":   "sp500",
    }

    ctx = {}
    for yf_ticker, name in tickers.items():
        try:
            data = yf.Ticker(yf_ticker).history(period="10d")
            if len(data) >= 2:
                ctx[name]             = round(float(data["Close"].iloc[-1]), 2)
                ctx[f"{name}_5d_chg"] = round(
                    (data["Close"].iloc[-1] / data["Close"].iloc[-5] - 1) * 100, 2
                ) if len(data) >= 5 else 0.0
        except Exception as e:
            log.debug(f"  Could not fetch {yf_ticker}: {e}")

    # Derived
    if "us10y_yield" in ctx and "us3m_yield" in ctx:
        ctx["yield_curve"] = round(ctx["us10y_yield"] - ctx["us3m_yield"], 3)
        ctx["yield_curve_signal"] = (
            "inverted (recession warning)" if ctx["yield_curve"] < 0
            else "flat" if ctx["yield_curve"] < 0.5
            else "normal"
        )

    if "vix" in ctx:
        ctx["vix_regime"] = (
            "fear (VIX>30)"      if ctx["vix"] >= 30 else
            "elevated (VIX20-30)" if ctx["vix"] >= 20 else
            "normal (VIX15-20)"  if ctx["vix"] >= 15 else
            "calm (VIX<15)"
        )

    if "sp500_5d_chg" in ctx:
        ctx["market_momentum"] = (
            "strong rally" if ctx["sp500_5d_chg"] > 3 else
            "rally"        if ctx["sp500_5d_chg"] > 1 else
            "flat"         if ctx["sp500_5d_chg"] > -1 else
            "pullback"     if ctx["sp500_5d_chg"] > -3 else
            "sharp selloff"
        )

    ctx["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return ctx


# Tickers that are individual stocks — have earnings, earnings IV, etc.
# ETFs don't have earnings calendars — fetching them causes 404 errors
STOCK_TICKERS = {"NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "NFLX"}


def _is_stock(ticker: str) -> bool:
    """Quick check: is this an individual stock vs an ETF/fund?"""
    if ticker in STOCK_TICKERS:
        return True
    # Heuristic: ETFs are usually 3-4 chars without numbers
    # Stocks with < 5 chars might be either — default to trying info.quoteType
    return False


def build_ticker_context(ticker: str) -> dict:
    """
    Fetch ticker-specific context: recent price action,
    earnings proximity (stocks only), ex-dividend date if available.

    ETFs: only fetch price/return data — no earnings calendar
    Stocks: fetch earnings + ex-div normally
    """
    ctx = {"ticker": ticker}
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="30d")

        if len(hist) >= 5:
            ctx["price"]    = round(float(hist["Close"].iloc[-1]), 2)
            ctx["ret_5d"]   = round(
                (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100, 2
            )
            ctx["ret_21d"]  = round(
                (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2
            ) if len(hist) >= 21 else 0.0

            sma20 = hist["Close"].tail(20).mean()
            ctx["vs_sma20_pct"] = round(
                (hist["Close"].iloc[-1] / sma20 - 1) * 100, 2
            )
            ctx["sma20_signal"] = (
                "above SMA20 (momentum)" if ctx["vs_sma20_pct"] > 1
                else "near SMA20 (neutral)" if ctx["vs_sma20_pct"] > -1
                else "below SMA20 (dip)"
            )

        # Only fetch company-specific data for stocks, not ETFs
        # ETFs don't have earnings calendars — this causes 404 errors
        is_stock = _is_stock(ticker)

        if is_stock:
            # Earnings date — stocks only
            try:
                cal = t.calendar
                if cal is not None and not cal.empty:
                    earnings_date = cal.iloc[0, 0]
                    if hasattr(earnings_date, "date"):
                        days_to = (earnings_date.date() - datetime.today().date()).days
                        if 0 <= days_to <= 30:
                            ctx["earnings_days"]   = days_to
                            ctx["earnings_signal"] = (
                                "earnings THIS WEEK — elevated IV" if days_to <= 5
                                else f"earnings in {days_to}d"
                            )
            except Exception:
                pass  # calendar unavailable — not an error

            # Ex-dividend — stocks only
            try:
                info = t.info
                ctx["name"] = info.get("longName", ticker)
                ctx["type"] = info.get("quoteType", "STOCK")
                ex_div = info.get("exDividendDate")
                if ex_div:
                    ex_div_dt = pd.Timestamp(ex_div, unit="s").date()
                    days_to   = (ex_div_dt - datetime.today().date()).days
                    if -5 <= days_to <= 20:
                        ctx["ex_div_days"]   = days_to
                        ctx["ex_div_signal"] = (
                            "ex-div PASSED (post-dip opportunity)" if days_to < 0
                            else f"ex-div in {days_to}d (price may dip after)"
                        )
            except Exception:
                ctx.setdefault("name", ticker)
                ctx.setdefault("type", "STOCK")
        else:
            # ETF — just name and type, no earnings/ex-div attempt
            try:
                # Use fast_info which is lighter and works for ETFs
                fi = t.fast_info
                ctx["name"] = getattr(fi, "exchange", ticker)
                ctx["type"] = "ETF"
            except Exception:
                ctx["name"] = ticker
                ctx["type"] = "ETF"

    except Exception as e:
        log.debug(f"  Ticker context error for {ticker}: {e}")
        ctx.setdefault("name", ticker)
        ctx.setdefault("type", "UNKNOWN")

    return ctx


def generate_advisory(
    ticker: str,
    lgbm_recommendation: dict,
    macro_ctx: dict,
    ticker_ctx: dict,
) -> dict:
    """
    Call Groq LLM to generate a plain-English advisory for one ticker.

    Args:
        ticker:              e.g. 'VOO'
        lgbm_recommendation: output from predict_optimal_day()
        macro_ctx:           from build_macro_context()
        ticker_ctx:          from build_ticker_context()

    Returns dict with:
        advisory:   plain-English paragraph (2-3 sentences)
        suggested_window: e.g. "days 20-23"
        confidence_context: brief reason for confidence level
        action:     "act" / "consider" / "skip"
    """
    client = get_groq_client()
    if client is None:
        return _fallback_advisory(ticker, lgbm_recommendation, macro_ctx)

    # Build the prompt
    rec_day  = lgbm_recommendation.get("recommended_day", 27)
    top3     = lgbm_recommendation.get("top_3_days", [rec_day])
    conf     = lgbm_recommendation.get("confidence", 0.5)
    tier     = lgbm_recommendation.get("timing_precision", "window")
    pred_save = lgbm_recommendation.get("predicted_saving", 0.0)

    prompt = f"""You are a quantitative investment advisor helping a retail investor decide 
when to make their monthly DCA (dollar-cost averaging) purchase.

CURRENT MACRO ENVIRONMENT:
- VIX: {macro_ctx.get('vix', 'N/A')} ({macro_ctx.get('vix_regime', 'unknown')})
- VIX 5-day change: {macro_ctx.get('vix_5d_chg', 0):+.1f}%
- S&P 500 5-day: {macro_ctx.get('sp500_5d_chg', 0):+.1f}% ({macro_ctx.get('market_momentum', 'flat')})
- Yield curve: {macro_ctx.get('yield_curve', 'N/A')} ({macro_ctx.get('yield_curve_signal', '')})
- Gold 5-day: {macro_ctx.get('gold_5d_chg', 0):+.1f}% (risk-off signal)
- Oil 5-day: {macro_ctx.get('oil_wti_5d_chg', 0):+.1f}% (inflation signal)
- USD 5-day: {macro_ctx.get('usd_index_5d_chg', 0):+.1f}%

TICKER: {ticker} ({ticker_ctx.get('name', ticker)})
- Current price: ${ticker_ctx.get('price', 'N/A')}
- 5-day return: {ticker_ctx.get('ret_5d', 0):+.1f}%
- vs 20-day SMA: {ticker_ctx.get('vs_sma20_pct', 0):+.1f}% ({ticker_ctx.get('sma20_signal', '')})
{f"- Earnings: {ticker_ctx.get('earnings_signal', '')}" if 'earnings_signal' in ticker_ctx else ""}
{f"- Ex-dividend: {ticker_ctx.get('ex_div_signal', '')}" if 'ex_div_signal' in ticker_ctx else ""}

MODEL RECOMMENDATION:
- Recommended day: {rec_day}th of the month
- Top 3 days by score: {top3}
- Model confidence: {conf:.0%}
- Model precision tier: {tier} (exact/window/loose)
- Predicted saving vs day 27: {pred_save:+.2f}%

TASK: Write a 2-3 sentence advisory for this investor. 
Rules:
1. Do NOT tell them what to do. Frame as context and considerations.
2. Note any macro events that support or conflict with the model's recommendation.
3. If earnings or ex-dividend dates matter, mention them specifically.
4. End with the suggested purchase window (e.g. "Consider days 20-23").
5. Be specific and factual. No generic advice.
6. Maximum 60 words.

Also provide:
- suggested_window: a day range string like "days 20-23"  
- action: one of "act" (strong signal), "consider" (mixed signal), "skip" (weak/risky)

Respond in JSON only:
{{"advisory": "...", "suggested_window": "...", "action": "act|consider|skip", "key_factor": "one phrase explaining the main driver"}}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw_clean)

        # Validate required fields
        result.setdefault("advisory",        "No advisory available.")
        result.setdefault("suggested_window", f"days {top3[0]-1}-{top3[0]+1}")
        result.setdefault("action",          "consider")
        result.setdefault("key_factor",      "model signal")

        log.info(f"  ✓ {ticker} LLM advisory: action={result['action']}")
        return result

    except json.JSONDecodeError as e:
        log.warning(f"  LLM JSON parse error for {ticker}: {e}")
        return _fallback_advisory(ticker, lgbm_recommendation, macro_ctx)
    except Exception as e:
        log.warning(f"  LLM call failed for {ticker}: {e}")
        return _fallback_advisory(ticker, lgbm_recommendation, macro_ctx)


def _fallback_advisory(
    ticker: str,
    lgbm_recommendation: dict,
    macro_ctx: dict,
) -> dict:
    """
    Rule-based fallback when Groq is unavailable.
    Uses macro context to generate a basic advisory.
    """
    rec_day = lgbm_recommendation.get("recommended_day", 27)
    conf    = lgbm_recommendation.get("confidence", 0.5)
    vix     = macro_ctx.get("vix", 18)
    regime  = macro_ctx.get("vix_regime", "normal")
    sp5d    = macro_ctx.get("sp500_5d_chg", 0)

    if vix >= 25:
        advisory = (
            f"Elevated volatility (VIX={vix:.0f}) means larger intra-month swings "
            f"and more timing opportunity. Model recommends day {rec_day} with "
            f"{conf:.0%} confidence. High-VIX months historically show strongest model performance."
        )
        action = "act" if conf > 0.6 else "consider"
    elif vix < 15:
        advisory = (
            f"Calm market (VIX={vix:.0f}) — day-to-day price differences are small. "
            f"Model suggests day {rec_day} but the gain over day 27 will be modest. "
            f"Timing adds limited value this month."
        )
        action = "consider" if conf > 0.65 else "skip"
    else:
        momentum = "recovering" if sp5d > 1 else "pulling back" if sp5d < -1 else "flat"
        advisory = (
            f"Normal volatility environment (VIX={vix:.0f}), market {momentum} "
            f"({sp5d:+.1f}% past 5 days). Model recommends day {rec_day} with "
            f"{conf:.0%} confidence. Standard timing conditions."
        )
        action = "act" if conf > 0.65 else "consider"

    return {
        "advisory":        advisory,
        "suggested_window": f"days {max(18, rec_day-1)}-{min(31, rec_day+1)}",
        "action":          action,
        "key_factor":      f"VIX={vix:.0f} ({regime})",
    }


def generate_portfolio_plan(
    portfolio_recommendations: dict,
    macro_ctx: dict,
) -> dict:
    """
    Generate the full monthly execution plan for all tickers.

    Args:
        portfolio_recommendations: {ticker: lgbm_recommendation_dict}
        macro_ctx: shared macro context

    Returns: {ticker: advisory_dict} for all tickers
    """
    plan = {}
    for ticker, lgbm_rec in portfolio_recommendations.items():
        ticker_ctx = build_ticker_context(ticker)
        advisory   = generate_advisory(ticker, lgbm_rec, macro_ctx, ticker_ctx)
        plan[ticker] = {
            "lgbm":         lgbm_rec,
            "ticker_ctx":   ticker_ctx,
            "advisory":     advisory,
        }
    return plan