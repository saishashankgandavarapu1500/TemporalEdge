"""
TemporalEdge — Scenario Sampler  (Phase 4, Step 1)
Draws correlated macro scenarios from actual historical data.

Key design principle: we never generate random macro combinations.
We sample REAL historical months so correlations are always realistic.
(VIX=40 never appears with a steep yield curve and surging dollar —
 because that never happened historically either.)

For VTI specifically: only samples from post-2010 months to avoid
dot-com/GFC era patterns that the model never learned.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.config import DATA_FEAT, DATA_MACRO, PURCHASE_WINDOW_START
from src.utils.logger import get_logger

log = get_logger("simulation.sampler")

# Tickers where we restrict to post-2010 to avoid structural regime shift
# VTI: pre-2010 dot-com/GFC patterns are structurally different — excluded
# Note: VTI now uses VOO model proxy, so post-2010 restriction also ensures
# the scenario period matches VOO's training regime (VOO started Sep 2010)
POST_2010_ONLY = {"VTI"}

# Macro columns that define the scenario context
# These are fed to the LLM reasoning layer AND used to label each run
SCENARIO_MACRO_COLS = [
    "vix",                  # fear gauge
    "vix_regime",           # 0=calm 1=normal 2=elevated 3=fear
    "vix_change_5d",        # VIX direction
    "sp500_ret_5d",         # market momentum
    "sp500_ret_21d",        # market trend
    "gold_ret_5d",          # risk-off signal
    "oil_ret_5d",           # inflation proxy
    "copper_ret_21d",       # global growth (Dr. Copper)
    "usd_ret_5d",           # USD direction
    "yield_curve_10y3m",    # recession signal
    "yield_change_5d",      # rate direction
    "fed_funds_rate",       # rate level
    "near_options_expiry",  # options expiry proximity
    "is_fomc_month",        # Fed meeting month
    "days_to_month_end",    # calendar position
    "day_of_week",          # trading day context
]


class ScenarioSampler:
    """
    Samples historical purchase windows from the feature store.
    Each 'scenario' is one real historical month's worth of data
    for days >= PURCHASE_WINDOW_START.
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}
        self._scenario_pool: dict[str, list[pd.DataFrame]] = {}

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        path = DATA_FEAT / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}. Run Phase 1 first.")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        self._cache[ticker] = df
        return df

    def build_scenario_pool(self, ticker: str) -> list[dict]:
        """
        Build a pool of historical monthly windows for this ticker.
        Each entry is one month's purchase window (days >= 18).
        Returns list of dicts, one per month.
        Works for both portfolio tickers and on-the-fly unknown tickers.
        """
        if ticker in self._scenario_pool:
            return self._scenario_pool[ticker]

        df = self._load_ticker(ticker)

        # VTI fix: skip pre-2010 (dot-com/GFC era pattern shift)
        # Only applies to known portfolio tickers, not on-the-fly
        if ticker in POST_2010_ONLY:
            df = df[df.index >= "2010-01-01"]
            log.info(f"  {ticker}: restricted to post-2010 data ({len(df):,} rows)")

        # For on-the-fly tickers: always use full available history
        # (no regime restriction — we don't know their history yet)

        # Filter to purchase window
        window = df[df.index.day >= PURCHASE_WINDOW_START].copy()

        pool = []
        for period, group in window.groupby(window.index.to_period("M")):
            if len(group) < 3:
                continue

            # Build macro snapshot for this month (use first row of window)
            macro_row = group.iloc[0]
            macro_context = {
                col: float(macro_row[col])
                if col in group.columns and pd.notna(macro_row[col])
                else 0.0
                for col in SCENARIO_MACRO_COLS
            }

            # Classify this month's regime
            vix = macro_context.get("vix", 18)
            regime = (
                "fear"      if vix >= 30 else
                "elevated"  if vix >= 20 else
                "normal"    if vix >= 15 else
                "calm"
            )

            # Classify momentum
            sp5d = macro_context.get("sp500_ret_5d", 0)
            momentum = (
                "strong_bull" if sp5d >  0.04 else
                "bull"        if sp5d >  0.01 else
                "sideways"    if sp5d > -0.01 else
                "bear"        if sp5d > -0.04 else
                "crash"
            )

            pool.append({
                "period":        str(period),
                "ticker":        ticker,
                "window_df":     group,       # actual price + feature data
                "macro":         macro_context,
                "regime":        regime,
                "momentum":      momentum,
                "vix":           vix,
                "n_days":        len(group),
            })

        self._scenario_pool[ticker] = pool
        log.info(f"  {ticker}: {len(pool)} historical scenarios built "
                 f"({pool[0]['period']} → {pool[-1]['period']})")
        return pool

    def sample(
        self,
        ticker: str,
        n: int = 1000,
        regime_filter: str | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """
        Sample n scenarios with replacement from the historical pool.

        Args:
            ticker:        portfolio ticker
            n:             number of scenarios to sample
            regime_filter: if set, only sample from this VIX regime
                           ('calm', 'normal', 'elevated', 'fear')
            seed:          random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        pool = self.build_scenario_pool(ticker)

        if regime_filter:
            pool = [s for s in pool if s["regime"] == regime_filter]
            if not pool:
                log.warning(f"  No scenarios for regime '{regime_filter}', using all")
                pool = self.build_scenario_pool(ticker)

        indices = rng.integers(0, len(pool), size=n)
        return [pool[i] for i in indices]

    def get_regime_distribution(self, ticker: str) -> dict:
        """Returns the empirical frequency of each VIX regime for this ticker."""
        pool = self.build_scenario_pool(ticker)
        from collections import Counter
        counts = Counter(s["regime"] for s in pool)
        total = len(pool)
        return {r: round(counts[r] / total * 100, 1) for r in counts}

    def get_scenario_stats(self, ticker: str) -> dict:
        """Summary statistics of the scenario pool."""
        pool = self.build_scenario_pool(ticker)
        vix_vals = [s["vix"] for s in pool]
        regimes  = [s["regime"] for s in pool]
        from collections import Counter
        return {
            "n_scenarios":    len(pool),
            "vix_mean":       round(float(np.mean(vix_vals)), 1),
            "vix_median":     round(float(np.median(vix_vals)), 1),
            "regime_counts":  dict(Counter(regimes)),
            "regime_pct":     self.get_regime_distribution(ticker),
            "date_range":     f"{pool[0]['period']} → {pool[-1]['period']}",
        }


# Singleton for reuse across simulation runs
sampler = ScenarioSampler()
