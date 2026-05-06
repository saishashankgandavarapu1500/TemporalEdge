# TemporalEdge — AI-Powered DCA Timing
 
> *Stop buying on day 27 out of habit. Buy on the day the market gives you an edge.*
 
TemporalEdge is a portfolio-aware dollar-cost averaging (DCA) optimiser that uses machine learning to identify the statistically best day of each month to invest — ticker by ticker, regime by regime.
 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-green.svg)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-89%20passing-brightgreen.svg)]()
 
---
 
## The Problem
 
Most investors DCA on a fixed calendar day — payday, end of month, whatever is convenient. That day is arbitrary. Markets are not.
 
Volatility clusters. Macro regimes shift. Some days of the month are statistically cheaper to buy than others — and that pattern is learnable.
 
TemporalEdge learns it.
 
The model doesn't predict prices. It predicts **relative entry quality** — a subtler and more tractable problem.
 
---
 
## What It Does
 
For each ticker, TemporalEdge:
 
1. Downloads 26 years of price history (yfinance + FRED)
2. Builds 121 features — momentum, volatility, VIX regime, yield curve, macro indicators
3. Trains a per-ticker LightGBM model using walk-forward validation (out-of-sample only)
4. Runs 1,000 Monte Carlo simulations across market regimes
5. Outputs a trust-scored recommendation with compounding projections
6. Generates an LLM advisory (Groq / Llama 3) for plain-English interpretation
---
 
## Validated Results
 
All results are **out-of-sample** from walk-forward validation across 139–280 months per ticker.
 
### Win Rate vs Fixed Day 27
 
| Ticker | Win Rate | Avg Monthly Saving | Out-of-Sample Months |
|--------|----------|--------------------|----------------------|
| VOO    | 82.2%    | +1.12%             | 152                  |
| SCHD   | 80.6%    | +1.00%             | 139                  |
| VYM    | 78.3%    | +0.96%             | 198                  |
| VTI    | 78.2%    | +1.16%             | 225                  |
| AAPL   | 77.3%    | +2.05%             | 225                  |
| VEA    | 76.8%    | +1.11%             | 190                  |
| BND    | 74.6%    | +0.26%             | 193                  |
| NVDA   | 73.8%    | +3.10%             | 225                  |
| VWO    | 73.4%    | +1.30%             | 218                  |
| TSLA   | 72.3%    | +2.28%             | 155                  |
| VXUS   | 70.3%    | +0.80%             | 148                  |
 
### Statistical Significance
 
- **All 11 tickers**: binomial p < 0.01, t-statistics ranging 4.0–10.7
- **Average Sharpe of timing edge**: 2.09 annualised
- **Year-by-year stability**: 9 of 11 tickers positive in >90% of individual years tested
### Model vs Naive Baseline
 
A naive "always buy on the first available day" strategy beats day 27 ~57% of the time — purely from the structural effect of end-of-month rebalancing flows. The model adds **+13–24 percentage points on top of that**, confirming it is learning genuine signal beyond a simple calendar effect.
 
| Ticker | Model Win Rate | Naive Win Rate | Model Lift |
|--------|---------------|----------------|------------|
| VYM    | 79.3%         | 55.6%          | +23.7%     |
| VWO    | 75.7%         | 52.0%          | +23.7%     |
| AAPL   | 79.1%         | 58.9%          | +20.2%     |
| VEA    | 77.4%         | 58.3%          | +19.0%     |
| NVDA   | 70.7%         | 52.5%          | +18.2%     |
| VOO    | 78.3%         | 61.1%          | +17.2%     |
| VTI    | 76.4%         | 59.4%          | +17.0%     |
| TSLA   | 71.6%         | 54.9%          | +16.7%     |
| BND    | 76.2%         | 61.4%          | +14.8%     |
| SCHD   | 77.7%         | 63.9%          | +13.8%     |
| VXUS   | 70.9%         | 57.5%          | +13.5%     |
 
### Regime Analysis
 
The edge holds across all four VIX regimes — it is not purely a fear-market effect.
 
| Regime   | VIX Level | Avg Win Rate | Notes                              |
|----------|-----------|--------------|-------------------------------------|
| Calm     | < 15      | 73.3%        | Significant across most tickers    |
| Normal   | 15–20     | 76.5%        | Significant across all tickers     |
| Elevated | 20–30     | 76.3%        | BND loses significance here        |
| Fear     | > 30      | 87.4%        | Strongest edge, as expected        |
 
### Compounding Impact
 
On $100/month invested, the timing edge compounds to meaningful extra wealth over 20 years (assuming 8% annual returns):
 
| Ticker | Annual Edge | 20yr Compounded Value |
|--------|-------------|----------------------|
| NVDA   | $37.22      | $1,703               |
| TSLA   | $27.30      | $1,249               |
| AAPL   | $24.62      | $1,127               |
| VWO    | $15.58      | $713                 |
| VTI    | $13.93      | $638                 |
| VOO    | $13.42      | $614                 |
| VEA    | $13.32      | $610                 |
| SCHD   | $12.04      | $551                 |
| VYM    | $11.48      | $525                 |
| VXUS   | $9.58       | $438                 |
| BND    | $3.09       | $141                 |
 
All 11 tickers survive a 0.1% round-trip transaction cost assumption.
 
---
 
## Validation Methodology
 
### Lookahead Bias
 
All FRED macro features are shifted forward by one business day before joining price data — the model only sees information publicly available before the trading day. This was tested explicitly: applying the shift produced **zero change** in win rates across all 11 tickers, confirming no leakage in the original pipeline.
 
### Walk-Forward Validation
 
The model uses expanding walk-forward validation: train on months 1–36, predict month 37; train on months 1–37, predict month 38; and so on. The model is **never evaluated on data it was trained on**.
 
### Year-by-Year Stability
 
| Ticker | Positive Years | Total Years | Worst Year Win Rate |
|--------|---------------|-------------|---------------------|
| VTI    | 20/20         | 20          | 66.7% (2010)        |
| NVDA   | 20/20         | 20          | 58.3% (2016)        |
| AAPL   | 19/20         | 20          | 50.0% (2026 partial)|
| VYM    | 17/17         | 17          | 58.3% (2024)        |
| VEA    | 16/17         | 17          | 50.0% (2018)        |
| VOO    | 14/14         | 14          | 66.7% (2014)        |
| TSLA   | 13/14         | 14          | 50.0% (2017)        |
| BND    | 14/17         | 17          | 44.4% (2010)        |
 
---
 
## Known Limitations
 
Documented honestly — these are real constraints:
 
1. **Structural baseline**: ~57% of the win rate comes from a structural "buy earlier is cheaper" effect from end-of-month rebalancing flows. The model adds 13–24% on top — but both components are real.
2. **BND in elevated VIX**: Bond timing loses statistical significance when VIX is 20–30. Rate volatility during Fed uncertainty makes bond entry harder to predict.
3. **US large-cap scope**: All models are validated on US large-cap equities and major ETFs. Generalisation to international markets, small caps, or crypto is untested.
4. **Relative metric only**: The model optimises entry timing vs. a fixed day — not absolute portfolio returns, drawdowns, or risk-adjusted outcomes. It is an entry timing tool, not a portfolio strategy.
5. **LLM layer is interpretability only**: The Groq/Llama 3 advisory does not improve model signal — it translates model output into plain English for non-technical users.
6. **Short-history tickers**: 18 tickers failed precompute due to delisting or insufficient history (<37 months) for model training.
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
 
**Portfolio Simulator** — run DCA timing analysis for any ticker. Portfolio tickers and all 491 precomputed S&P 500 stocks load instantly. Unknown tickers trigger a live pipeline run (~3–5 min) with explicit user consent.
 
**Execution Plan** — shows the model's recommended buy windows for the current month across your portfolio, ranked by signal strength.
 
**Historical Evidence** — the full EDA report embedded inline, covering data quality, day-of-month patterns, VIX regime analysis, macro correlations, simulation results, and feature importance.
 
---
 
## Ticker Coverage
 
| Type | Count | Load time |
|------|-------|-----------|
| Portfolio tickers | 11 | Instant |
| S&P 500 + major ETFs | 491 | Instant |
| Any Yahoo Finance ticker | Unlimited | ~3–5 min (live pipeline) |
 
---
 
## Stack
 
| Layer | Technology |
|-------|-----------|
| Data | yfinance, FRED API, pandas |
| Features | 121 engineered features — momentum, volatility, macro |
| Model | LightGBM (per-ticker, walk-forward validated) |
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
│   ├── precomputed/              ← 491 × _result.json (~1.5 MB, committed)
│   ├── raw/                      ← price history (not committed)
│   └── features/                 ← parquet feature matrices (not committed)
├── models/
│   ├── robustness_report.json    ← statistical validation results
│   ├── reasons_3_4_report.json   ← regime + Sharpe analysis
│   └── {ticker}_backtest.csv     ← per-ticker walk-forward results
├── src/
│   ├── data/
│   │   ├── collector.py          ← yfinance + FRED downloader
│   │   └── engineer.py           ← 121-feature engineering
│   ├── features/
│   │   └── feature_selector.py   ← EDA-informed feature selection
│   ├── models/
│   │   └── trainer.py            ← LightGBM walk-forward training
│   ├── pipeline/
│   │   └── on_demand.py          ← full pipeline: download → train → simulate
│   ├── simulation/
│   │   └── monte_carlo.py        ← 1,000-run Monte Carlo engine
│   └── llm/
│       └── groq_advisor.py       ← Llama 3 advisory generation
├── dashboard/
│   ├── app.py                    ← Streamlit entry point
│   ├── cache/on_demand/          ← 30-day result cache
│   └── components/
│       └── simulator.py          ← simulator with precomputed gate
├── scripts/
│   ├── precompute_sp500.py       ← batch precompute runner
│   ├── validate_precomputed.py   ← JSON validator with dry-run
│   ├── validate_robustness.py    ← walk-forward stability + significance tests
│   └── check_reasons_3_4.py      ← regime dependence + Sharpe analysis
└── tests/
    └── test_simulator_full.py    ← 89-test pytest suite
```
 
---
 
## Running Locally
 
```bash
# Clone and install
git clone https://github.com/yourusername/temporaledge.git
cd temporaledge
pip install -r requirements.txt
 
# Set your Groq API key
export GROQ_API_KEY=your_key_here
 
# Run the dashboard (precomputed results load instantly)
streamlit run dashboard/app.py
 
# Rebuild data and models from scratch
python src/data/collector.py --refresh
python src/features/engineer.py
python src/models/trainer.py
 
# Run robustness validation
python scripts/validate_robustness.py
python scripts/check_reasons_3_4.py
 
# Run precompute for all S&P 500 tickers (~3–4 hours)
ulimit -n 4096 && python scripts/precompute_sp500.py --workers 2
 
# Run tests
pytest tests/ -v
```
 
---
 
## Trust Score System
 
| Tier | Score | Label | Meaning |
|------|-------|-------|---------|
| A | 65–100 | ACT ON MODEL | High confidence — long history, strong signal |
| B | 40–64 | USE WITH CAUTION | Moderate confidence — some data limitations |
| C | 0–39 | INFORMATIONAL ONLY | Low confidence — short history or weak signal |
 
Trust is penalised for: short price history (<5 years), low win rate (<55%), high regime instability, or proxy data usage.
 
---
 
## Deployment Notes
 
Streamlit Cloud free tier has a 1GB repo limit. Only lightweight precomputed JSONs (~1.5MB) are committed. Large files are excluded:
 
```gitignore
data/raw/
data/features/
**/*.parquet
**/*.pkl
!data/precomputed/*_result.json
!dashboard/cache/on_demand/*_result.json
```
 
---

## Author

Built by Shashank Gandavarapu as a full-stack ML project combining quantitative finance, machine learning, and production deployment.

- Portfolio: [https://myportfolio-ssg.vercel.app/]
- LinkedIn: [https://www.linkedin.com/in/saishashankgandavarapu1500/]
- GitHub: [[your-github](https://github.com/saishashankgandavarapu1500)]
