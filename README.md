# TemporalEdge — AI-Powered DCA Timing

> *Stop buying on day 27 out of habit. Buy on the day the market gives you an edge.*

TemporalEdge is a portfolio-aware dollar-cost averaging (DCA) optimiser that uses machine learning to identify the statistically best day of each month to invest — ticker by ticker, regime by regime.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-green.svg)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io)

---

## The Problem

Most investors DCA on a fixed calendar day — payday, end of month, whatever is convenient. That day is arbitrary. Markets are not.

Volatility clusters. Macro regimes shift. Some days of the month are statistically cheaper to buy than others — and that pattern is learnable.

TemporalEdge learns it.

---

## What It Does

For each ticker, TemporalEdge:

1. Downloads 26 years of price history (yfinance + FRED)
2. Builds 121 features — momentum, volatility, VIX regime, yield curve, macro indicators
3. Trains a LightGBM model to predict whether buying today vs. the fixed 27th saves money
4. Runs 1,000 Monte Carlo simulations across market regimes
5. Outputs a trust-scored recommendation with compounding projections
6. Generates an LLM advisory (Groq / Llama 3) for plain-English context

---

## Key Results

From the EDA across 11 portfolio tickers (full historical analysis):

- **Day-of-month timing matters** — buying on the model-selected day beats a fixed day 27 in 65–79% of months across all portfolio tickers
- **Regime-aware performance** — the model outperforms most in fear/elevated VIX regimes, where entry timing has the largest impact
- **Macro correlations** — yield curve shape, Fed rate direction, and VIX level are the strongest predictors of optimal entry days
- **Compounding edge** — even a 0.8% average monthly saving compounds to meaningful extra wealth over 10–20 years

---

## Architecture

```
yfinance + FRED
      ↓
  data/raw/          ← price history + macro panel
      ↓
  data/features/     ← 121 engineered features (parquet)
      ↓
  models/            ← per-ticker LightGBM (.pkl)
      ↓
  Monte Carlo        ← 1,000 regime-aware simulations
      ↓
  Groq / Llama 3     ← LLM advisory layer
      ↓
  Streamlit          ← interactive dashboard
```

**Precomputed results** (`data/precomputed/`) store 491 S&P 500 + ETF results as lightweight JSON (~3KB each), so the dashboard loads instantly without re-running the pipeline.

---

## Dashboard

The Streamlit dashboard has three pages:

**Portfolio Simulator** — run DCA timing analysis for any ticker. Portfolio tickers and all 491 precomputed S&P 500 stocks load instantly from cache. Unknown tickers trigger a live pipeline run (~3–5 min) with explicit user consent.

**Execution Plan** — shows the model's recommended buy windows for the current month across your portfolio, ranked by signal strength.

**Historical Evidence** — the EDA report embedded inline, covering 6 sections: portfolio data quality, day-of-month patterns, VIX regime analysis, macro factor correlations, historical simulation results, and feature importance.

---

## Ticker Coverage

| Type | Count | Load time |
|------|-------|-----------|
| Portfolio tickers | 11 | Instant |
| S&P 500 + major ETFs | 491 | Instant |
| Any other Yahoo Finance ticker | unlimited | ~3–5 min (live pipeline) |

The 18 tickers that failed precompute (ANSS, CDAY, CMA, DFS, FI, FLT, HES, IPG, JNPR, K, KVUE, MMC, MRO, SOLV, WBA, BRK-B, etc.) are delisted or have insufficient history (<37 months) for model training.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data | yfinance, FRED API, pandas |
| Features | 121 engineered features — momentum, volatility, macro |
| Model | LightGBM (per-ticker, regime-aware) |
| Simulation | NumPy Monte Carlo (1,000 runs) |
| LLM | Groq API (Llama 3.1-8B) |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |
| Testing | pytest (89 tests) |

---

## Project Structure

```
temporaledge/
├── data/
│   ├── precomputed/       ← 491 × _result.json (~1.5 MB total, committed)
│   ├── raw/               ← price history (not committed)
│   └── features/          ← parquet feature matrices (not committed)
├── models/                ← trained LightGBM pkl files (not committed)
├── src/
│   ├── pipeline/
│   │   └── on_demand.py   ← full pipeline: download → features → train → simulate
│   ├── simulation/
│   │   └── monte_carlo.py ← 1,000-run Monte Carlo engine
│   ├── models/
│   │   ├── features.py    ← 121-feature engineering
│   │   └── trainer.py     ← LightGBM training + trust scoring
│   └── llm/
│       └── groq_advisor.py ← Llama 3 advisory generation
├── dashboard/
│   ├── app.py             ← Streamlit entry point
│   ├── cache/on_demand/   ← 30-day result cache
│   └── components/
│       └── simulator.py   ← simulator tab with precomputed gate
├── scripts/
│   ├── precompute_sp500.py ← batch precompute runner
│   └── validate_precomputed.py ← JSON validator with dry-run
└── tests/
    └── test_simulator_full.py ← 89-test pytest suite
```

---

## Running Locally

```bash
# Clone and install
git clone https://github.com/yourusername/temporaledge.git
cd temporaledge
pip install -r requirements.txt

# Run the dashboard (precomputed results load instantly)
streamlit run dashboard/app.py

# Run the full precompute pipeline (3–4 hours, needs yfinance access)
ulimit -n 4096 && python scripts/precompute_sp500.py --workers 2

# Validate precomputed files
python scripts/validate_precomputed.py --dry-run

# Run tests
pytest tests/ -v
```

---

## EDA Highlights

The exploratory analysis covers 11 portfolio tickers across full historical data. Key findings:

**Section 1 — Portfolio Data Quality**: All tickers have 20+ years of clean price history. VOO, VTI, and SCHD show the highest data coverage; VXUS and VWO have slightly shorter histories due to fund inception dates.

**Section 2 — Does Day-of-Month Matter?**: Yes. Statistically significant price differences exist across days of the month for all 11 tickers. The effect is largest for high-volatility tickers (TSLA, NVDA) and smallest for bond ETFs (BND, AGG).

**Section 3 — Market Regime Analysis**: VIX regime (calm / normal / elevated / fear) is the strongest single predictor of optimal entry timing. In fear regimes, optimal entry days shift earlier in the month.

**Section 4 — Macro Factor Correlation**: Yield curve shape (10Y–3M spread), Fed funds rate direction, and CPI momentum are the top macro predictors. These are incorporated as FRED features in all models.

**Section 5 — Cost of the 27th**: Buying on day 27 costs an average of 0.6–1.1% more per month than the model-selected optimal day, depending on the ticker and regime.

**Section 6 — Feature Importance**: The top 10 features across all tickers are dominated by short-term momentum (3–10 day), VIX level, yield curve slope, and month-of-year seasonality.

The full interactive EDA report is embedded in the dashboard's Historical Evidence page.

---

## Trust Score System

Each ticker result includes a trust score (0–100) and tier:

| Tier | Score | Label | Meaning |
|------|-------|-------|---------|
| A | 65–100 | ACT ON MODEL | High confidence — long history, strong signal |
| B | 40–64 | USE WITH CAUTION | Moderate confidence — some data limitations |
| C | 0–39 | INFORMATIONAL ONLY | Low confidence — short history or weak signal |

Trust is penalised for: short price history (<5 years), low win rate (<55%), high regime instability, or proxy data usage.

---

## Deployment Notes

Streamlit Cloud free tier has a 1GB repo limit. This project keeps only the lightweight precomputed JSONs (~1.5MB) in the repo. Large files are excluded:

```gitignore
data/raw/
data/features/
**/*.parquet
**/*.pkl
!data/precomputed/*_result.json
!dashboard/cache/on_demand/*_result.json
```

Unknown tickers (not in the 491 precomputed) run the live pipeline on Streamlit Cloud on user demand — downloading fresh data, building features, and training a model in the session. Results are cached for 30 days.

---

## Author

Built by Shashank Gandavarapu as a full-stack ML project combining quantitative finance, machine learning, and production deployment.

- Portfolio: [https://myportfolio-ssg.vercel.app/]
- LinkedIn: [https://www.linkedin.com/in/saishashankgandavarapu1500/]
- GitHub: [[your-github](https://github.com/saishashankgandavarapu1500)]
