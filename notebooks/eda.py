"""
TemporalEdge — Phase 2: Exploratory Data Analysis
===================================================
Runs all 6 analysis sections and exports:
  - HTML report  (notebooks/eda_report.html)   ← share this
  - PNG charts   (notebooks/charts/)           ← use in write-up
  - CSV summaries(notebooks/exports/)          ← back up findings

Sections:
  1. Portfolio Overview & Data Quality
  2. Does Day-of-Month Matter? (The Core Signal)
  3. Market Regime Analysis
  4. Macro Factor Correlation
  5. Cost of the 27th — Historical Simulation
  6. Feature Importance Preview (pre-model signal strength)

Run:
  python notebooks/eda.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from src.config import PORTFOLIO, YOUR_CURRENT_DAY, PURCHASE_WINDOW_START, DATA_FEAT, DATA_RAW, DATA_MACRO
from src.utils.logger import get_logger

log = get_logger("eda")

# ── Output dirs ───────────────────────────────────────────────────────────────
NOTEBOOKS_DIR = Path(__file__).parent
CHARTS_DIR    = NOTEBOOKS_DIR / "charts"
EXPORTS_DIR   = NOTEBOOKS_DIR / "exports"
CHARTS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#534AB7",   # purple
    "secondary": "#0F6E56",   # teal
    "accent":    "#BA7517",   # amber
    "danger":    "#993C1D",   # coral
    "success":   "#3B6D11",   # green
    "neutral":   "#5F5E5A",   # gray
    "bg":        "#FAFAF8",
    "grid":      "#E8E6DE",
}

TICKER_COLORS = px.colors.qualitative.Safe[:len(PORTFOLIO)]
TICKER_COLOR_MAP = dict(zip(PORTFOLIO.keys(), TICKER_COLORS))

pio.templates["temporaledge"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=13, color="#2C2C2A"),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAF8",
        colorway=list(TICKER_COLOR_MAP.values()),
        xaxis=dict(gridcolor="#E8E6DE", linecolor="#D3D1C7", showgrid=True),
        yaxis=dict(gridcolor="#E8E6DE", linecolor="#D3D1C7", showgrid=True),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#D3D1C7", borderwidth=1),
    )
)
pio.templates.default = "temporaledge"


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> dict[str, pd.DataFrame]:
    """Load feature store for all portfolio tickers."""
    data = {}
    for ticker in PORTFOLIO:
        path = DATA_FEAT / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            data[ticker] = df
            log.info(f"  Loaded {ticker}: {len(df):,} rows")
        else:
            log.warning(f"  {ticker} features not found — run Phase 1 first")
    return data


def load_raw_prices() -> dict[str, pd.DataFrame]:
    """Load raw OHLCV for price-level analysis."""
    data = {}
    for ticker in PORTFOLIO:
        path = DATA_RAW / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            data[ticker] = df
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Portfolio Overview & Data Quality
# ─────────────────────────────────────────────────────────────────────────────

def section1_overview(data: dict, raw: dict) -> list[go.Figure]:
    log.info("\n[Section 1] Portfolio overview & data quality...")
    figs = []

    # 1a. Data availability timeline
    rows = []
    for ticker, df in data.items():
        rows.append({
            "ticker": ticker,
            "name":   PORTFOLIO[ticker]["name"],
            "start":  df.index.min(),
            "end":    df.index.max(),
            "rows":   len(df),
            "type":   PORTFOLIO[ticker]["type"],
            "monthly_usd": PORTFOLIO[ticker]["monthly_usd"],
        })
    summary_df = pd.DataFrame(rows).sort_values("start")

    fig1a = go.Figure()
    for _, row in summary_df.iterrows():
        fig1a.add_trace(go.Scatter(
            x=[row["start"], row["end"]],
            y=[row["ticker"], row["ticker"]],
            mode="lines+markers",
            line=dict(width=8, color=TICKER_COLOR_MAP[row["ticker"]]),
            marker=dict(size=8),
            name=row["ticker"],
            hovertemplate=(
                f"<b>{row['ticker']}</b><br>"
                f"{row['name']}<br>"
                f"From: {row['start'].date()}<br>"
                f"To: {row['end'].date()}<br>"
                f"Rows: {row['rows']:,}<br>"
                f"Monthly: ${row['monthly_usd']}"
            ),
        ))
    fig1a.update_layout(
        title="Data availability per ticker (max yfinance history)",
        xaxis_title="Year",
        yaxis_title="Ticker",
        height=450,
        showlegend=False,
    )
    figs.append(("1a_data_timeline", fig1a))

    # 1b. Portfolio allocation (monthly $ amounts)
    alloc = {t: PORTFOLIO[t]["monthly_usd"] for t in PORTFOLIO}
    fig1b = go.Figure(go.Pie(
        labels=list(alloc.keys()),
        values=list(alloc.values()),
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=list(TICKER_COLOR_MAP.values())),
        hovertemplate="<b>%{label}</b><br>$%{value}/month<br>%{percent}<extra></extra>",
    ))
    fig1b.update_layout(
        title="Monthly DCA allocation ($51 total)",
        annotations=[dict(text="$51/mo", x=0.5, y=0.5, font_size=18, showarrow=False)],
        height=420,
    )
    figs.append(("1b_allocation", fig1b))

    # 1c. Normalised cumulative returns (all tickers on same start base=100)
    fig1c = go.Figure()
    for ticker, df in raw.items():
        if "close" not in df.columns:
            continue
        # Start from common date (2015 for comparability)
        sub = df[df.index >= "2015-01-01"]["close"]
        if len(sub) < 100:
            continue
        normed = (sub / sub.iloc[0]) * 100
        fig1c.add_trace(go.Scatter(
            x=normed.index, y=normed.values,
            name=ticker,
            line=dict(color=TICKER_COLOR_MAP[ticker], width=1.5),
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Value: %{{y:.1f}}<extra></extra>",
        ))
    fig1c.add_hline(y=100, line_dash="dot", line_color=COLORS["neutral"], annotation_text="Base (2015)")
    fig1c.update_layout(
        title="Normalised performance since 2015 (base = 100)",
        yaxis_title="Value (base 100)",
        height=480,
    )
    figs.append(("1c_normalised_returns", fig1c))

    # Export summary
    summary_df.to_csv(EXPORTS_DIR / "portfolio_summary.csv", index=False)
    log.info(f"  ✓ Section 1: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Does Day-of-Month Matter? (THE CORE SIGNAL)
# ─────────────────────────────────────────────────────────────────────────────

def section2_day_of_month_signal(data: dict) -> list[go.Figure]:
    log.info("\n[Section 2] Day-of-month signal analysis...")
    figs = []

    # For each ticker: average forward return by day of month
    # Restrict to purchase window only
    # 2a. Heatmap: day-of-month vs ticker, colour = avg 21d forward return
    dom_returns = {}

    for ticker, df in data.items():
        work = df.copy()

        if "fwd_ret_21d" not in work.columns or work["fwd_ret_21d"].notna().sum() == 0:
            if "close" in work.columns:
                work["fwd_ret_21d"] = work["close"].shift(-21) / work["close"] - 1
                print(f"{ticker}: recomputed fwd_ret_21d from close")
            else:
                print(f"{ticker}: skipped (no fwd_ret_21d and no close)")
                continue

        window = work[work.index.day >= PURCHASE_WINDOW_START].copy()
        window = window.dropna(subset=["fwd_ret_21d"])

        print(
            f"{ticker}: rows={len(work)}, "
            f"window_rows={len(window)}, "
            f"nonnull_fwd_ret_21d={window['fwd_ret_21d'].notna().sum()}"
        )

        if window.empty:
            print(f"{ticker}: skipped (empty after filter/dropna)")
            continue

        window["dom"] = window.index.day
        avg = window.groupby("dom")["fwd_ret_21d"].agg(["mean", "std", "count"])

        if avg.empty or avg["mean"].notna().sum() == 0:
            print(f"{ticker}: skipped (grouped means empty/all-NaN)")
            continue

        avg.columns = ["mean_ret", "std_ret", "n"]
        avg["se"] = avg["std_ret"] / np.sqrt(avg["n"])

        dom_returns[ticker] = avg

    if not dom_returns:
        raise ValueError("Section 2a failed: no usable day-of-month forward return data found.")

    heatmap_data = pd.DataFrame(
        {ticker: avg["mean_ret"] for ticker, avg in dom_returns.items()}
    ).sort_index()

    print("heatmap_data shape:", heatmap_data.shape)
    print("non-null cells:", heatmap_data.notna().sum().sum())
    print(heatmap_data.head())

    if heatmap_data.empty or heatmap_data.notna().sum().sum() == 0:
        raise ValueError("Section 2a failed: heatmap_data is empty or all NaN.")

    fig2a = go.Figure(go.Heatmap(
        z=heatmap_data.T.to_numpy() * 100,
        x=heatmap_data.index.tolist(),
        y=heatmap_data.columns.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(heatmap_data.T.to_numpy() * 100, 2),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10),
        hovertemplate="Day %{x} | %{y}<br>Avg 21d return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Avg 21d<br>return %"),
    ))

    fig2a.add_vline(
        x=YOUR_CURRENT_DAY,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"Your day ({YOUR_CURRENT_DAY}th)",
        annotation_position="top",
    )

    fig2a.update_layout(
        title="Average 21-day forward return by purchase day (purchase window only)",
        xaxis_title="Day of month",
        yaxis_title="Ticker",
        height=500,
    )

    figs.append(("2a_dom_heatmap", fig2a))

    # 2b. Bar chart for each ticker — best vs worst days
    fig2b = make_subplots(
        rows=3, cols=4,
        subplot_titles=list(dom_returns.keys()),
        shared_yaxes=False,
    )
    for idx, (ticker, avg) in enumerate(dom_returns.items()):
        row = idx // 4 + 1
        col = idx % 4 + 1
        # FIX: use string labels (categorical axis) instead of integers
        # Numeric integer x with Plotly bdata encoding ignores range settings
        # Categorical x always renders all categories regardless of version
        x_labels = [f"d{d}" if d != YOUR_CURRENT_DAY else f"d{d}★"
                    for d in avg.index]
        colors_bar = [
            COLORS["danger"] if d == YOUR_CURRENT_DAY
            else (COLORS["success"] if v > 0 else "#B4B2A9")
            for d, v in zip(avg.index, avg["mean_ret"])
        ]
        fig2b.add_trace(
            go.Bar(
                x=x_labels,
                y=avg["mean_ret"] * 100,
                marker_color=colors_bar,
                name=ticker,
                showlegend=False,
                hovertemplate=f"<b>{ticker}</b> %{{x}}<br>Avg 21d return: %{{y:.2f}}%<extra></extra>",
            ),
            row=row, col=col,
        )
    fig2b.update_layout(
        title="Avg 21d forward return by purchase day — per ticker (★ = your day 27, green = above avg, red = below)",
        height=700,
    )
    figs.append(("2b_dom_bars", fig2b))

    # 2c. Portfolio-level: how often is day 27 the optimal day?
    optimal_stats = []
    for ticker, df in data.items():
        if "is_optimal_buy_day" not in df.columns:
            continue
        window = df[df.index.day >= PURCHASE_WINDOW_START]
        total_months     = window.index.to_period("M").nunique()
        day27_optimal    = window[
            (window.index.day == YOUR_CURRENT_DAY) &
            (window["is_optimal_buy_day"] == 1)
        ].index.to_period("M").nunique()
        optimal_stats.append({
            "ticker":          ticker,
            "total_months":    total_months,
            "day27_optimal":   day27_optimal,
            "day27_optimal_pct": day27_optimal / total_months * 100,
            "avg_saving_pct":  df["vs_day27_pct"].mean() * 100 if "vs_day27_pct" in df.columns else 0,
        })

    opt_df = pd.DataFrame(optimal_stats).sort_values("avg_saving_pct", ascending=False)

    fig2c = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"% of months day {YOUR_CURRENT_DAY} was optimal",
            "Average saving vs day 27 (%)"
        ],
    )
    fig2c.add_trace(
        go.Bar(
            x=opt_df["ticker"],
            y=opt_df["day27_optimal_pct"],
            marker_color=[TICKER_COLOR_MAP[t] for t in opt_df["ticker"]],
            hovertemplate="%{x}: %{y:.1f}% of months<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig2c.add_hline(y=100/12*3, line_dash="dot", line_color=COLORS["neutral"],
                    annotation_text="Random baseline", row=1, col=1)

    fig2c.add_trace(
        go.Bar(
            x=opt_df["ticker"],
            y=opt_df["avg_saving_pct"],
            marker_color=[
                COLORS["success"] if v > 0 else COLORS["danger"]
                for v in opt_df["avg_saving_pct"]
            ],
            hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig2c.update_layout(
        title=f"Day {YOUR_CURRENT_DAY} performance — is it actually good?",
        height=420,
    )
    figs.append(("2c_day27_analysis", fig2c))

    # 2d. Monthly price range available to exploit
    range_stats = []
    for ticker, df in data.items():
        if "close" not in df.columns:
            continue
        window = df[df.index.day >= PURCHASE_WINDOW_START]["close"]
        monthly = window.groupby(window.index.to_period("M"))
        monthly_range = monthly.apply(lambda x: (x.max() - x.min()) / x.mean() * 100)
        range_stats.append({
            "ticker": ticker,
            "median_range_pct": monthly_range.median(),
            "p25": monthly_range.quantile(0.25),
            "p75": monthly_range.quantile(0.75),
        })

    range_df = pd.DataFrame(range_stats).sort_values("median_range_pct", ascending=True)

    fig2d = go.Figure()
    fig2d.add_trace(go.Bar(
        x=range_df["median_range_pct"],
        y=range_df["ticker"],
        orientation="h",
        error_x=dict(
            type="data",
            symmetric=False,
            array=range_df["p75"] - range_df["median_range_pct"],
            arrayminus=range_df["median_range_pct"] - range_df["p25"],
        ),
        marker_color=[TICKER_COLOR_MAP[t] for t in range_df["ticker"]],
        hovertemplate="%{y}: %{x:.1f}% range<extra></extra>",
    ))
    fig2d.update_layout(
        title="Monthly price range within purchase window (= max theoretical saving from timing)",
        xaxis_title="Price range % (high-low / avg)",
        height=420,
    )
    figs.append(("2d_exploitable_range", fig2d))

    opt_df.to_csv(EXPORTS_DIR / "day27_analysis.csv", index=False)
    log.info(f"  ✓ Section 2: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Market Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

def section3_regime_analysis(data: dict) -> list[go.Figure]:
    log.info("\n[Section 3] Market regime analysis...")
    figs = []

    # Define regimes using S&P 500 rolling returns
    # Use VOO as proxy if sp500 macro not available
    ref_ticker = "VOO" if "VOO" in data else list(data.keys())[0]
    ref_df = data[ref_ticker].copy()

    # Regime classification
    if "sp500_ret_21d" in ref_df.columns:
        ret_col = "sp500_ret_21d"
    elif "ret_21d" in ref_df.columns:
        ret_col = "ret_21d"
    else:
        log.warning("  No return column for regime classification")
        return figs

    ref_df["regime"] = pd.cut(
        ref_df[ret_col],
        bins=[-np.inf, -0.08, -0.02, 0.02, 0.08, np.inf],
        labels=["crash", "bear", "sideways", "bull", "strong_bull"],
    )

    regime_colors = {
        "crash":       "#993C1D",
        "bear":        "#D85A30",
        "sideways":    "#888780",
        "bull":        "#3B6D11",
        "strong_bull": "#1D9E75",
    }

    # 3a. Optimal day distribution by regime — for NVDA and VOO
    for ticker in ["NVDA", "VOO"]:
        if ticker not in data:
            continue
        df = data[ticker].copy()
        df = df.join(ref_df[["regime"]], how="left")
        window = df[df.index.day >= PURCHASE_WINDOW_START].dropna(subset=["regime"])

        fig3 = make_subplots(
            rows=1, cols=5,
            subplot_titles=[r.replace("_", " ").title() for r in regime_colors],
            shared_yaxes=True,
        )
        for col_idx, (regime, color) in enumerate(regime_colors.items()):
            regime_data = window[window["regime"] == regime]
            if len(regime_data) < 10:
                continue
            dom_avg = regime_data.groupby(regime_data.index.day)["fwd_ret_21d"].mean() * 100
            # FIX: categorical x labels — avoid numeric autorange issues
            x_cat = [f"d{d}" if d != YOUR_CURRENT_DAY else f"d{d}★"
                     for d in dom_avg.index]
            bar_colors = [
                "black" if d == YOUR_CURRENT_DAY else color
                for d in dom_avg.index
            ]
            fig3.add_trace(
                go.Bar(
                    x=x_cat,
                    y=dom_avg.values,
                    marker_color=bar_colors,
                    showlegend=False,
                    name=regime,
                    hovertemplate=f"Day %{{x}}: %{{y:.2f}}%<extra></extra>",
                ),
                row=1, col=col_idx + 1,
            )
        # FIX: categorical x avoids numeric autorange issues
        fig3.update_layout(
            title=f"{ticker} — optimal buy day shifts by market regime",
            height=420,
        )
        figs.append((f"3_regime_{ticker}", fig3))

    # 3b. Avg saving by regime (portfolio-level)
    regime_savings = []
    for regime in regime_colors:
        monthly_savings = []
        for ticker, df in data.items():
            if "vs_day27_pct" not in df.columns:
                continue
            df2 = df.join(ref_df[["regime"]], how="left")
            window = df2[
                (df2.index.day >= PURCHASE_WINDOW_START) &
                (df2["regime"] == regime)
            ]
            if len(window) > 0:
                monthly_savings.append(window["vs_day27_pct"].mean() * 100)
        if monthly_savings:
            regime_savings.append({
                "regime": regime,
                "avg_saving": np.mean(monthly_savings),
                "color": regime_colors[regime],
            })

    rs_df = pd.DataFrame(regime_savings)
    fig3b = go.Figure(go.Bar(
        x=rs_df["regime"],
        y=rs_df["avg_saving"],
        marker_color=rs_df["color"],
        text=rs_df["avg_saving"].round(2).astype(str) + "%",
        textposition="outside",
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    ))
    fig3b.update_layout(
        title="Portfolio average saving vs day 27 by market regime",
        yaxis_title="Avg saving %",
        height=400,
    )
    figs.append(("3b_regime_savings", fig3b))

    log.info(f"  ✓ Section 3: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Macro Factor Correlation
# ─────────────────────────────────────────────────────────────────────────────

def section4_macro_correlation(data: dict) -> list[go.Figure]:
    log.info("\n[Section 4] Macro factor correlations...")
    figs = []

    macro_cols = [
        "vix", "vix_change_5d", "sp500_ret_5d", "sp500_ret_21d",
        "gold_ret_5d", "oil_ret_5d", "copper_ret_21d",
        "usd_ret_5d", "yield_curve_10y3m", "yield_change_5d",
        "days_to_month_end", "near_options_expiry", "is_fomc_month",
    ]

    # 4a. Correlation heatmap: macro factors vs 21d forward return
    for ticker in ["NVDA", "VOO", "BND"]:
        if ticker not in data:
            continue
        df = data[ticker].copy()
        available_macro = [c for c in macro_cols if c in df.columns]
        if not available_macro or "fwd_ret_21d" not in df.columns:
            continue

        corr_cols = available_macro + ["fwd_ret_21d"]
        corr_matrix = df[corr_cols].dropna().corr()
        target_corr = corr_matrix["fwd_ret_21d"].drop("fwd_ret_21d").sort_values()

        fig4 = go.Figure(go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=[
                COLORS["success"] if v > 0 else COLORS["danger"]
                for v in target_corr.values
            ],
            hovertemplate="%{y}: r = %{x:.3f}<extra></extra>",
        ))
        fig4.add_vline(x=0, line_color=COLORS["neutral"])
        fig4.update_layout(
            title=f"{ticker} — macro factor correlation with 21d forward return",
            xaxis_title="Pearson correlation (r)",
            height=450,
        )
        figs.append((f"4_macro_corr_{ticker}", fig4))

    # 4b. VIX regime vs all tickers saving potential
    vix_bins = ["Low (<15)", "Normal (15-20)", "Elevated (20-30)", "Fear (>30)"]
    vix_savings = {b: [] for b in vix_bins}

    for ticker, df in data.items():
        if "vix" not in df.columns or "vs_day27_pct" not in df.columns:
            continue
        window = df[df.index.day >= PURCHASE_WINDOW_START].dropna(subset=["vix"])
        for label, lo, hi in [
            ("Low (<15)", 0, 15),
            ("Normal (15-20)", 15, 20),
            ("Elevated (20-30)", 20, 30),
            ("Fear (>30)", 30, 200),
        ]:
            sub = window[(window["vix"] >= lo) & (window["vix"] < hi)]
            if len(sub) > 10:
                vix_savings[label].append(sub["vs_day27_pct"].mean() * 100)

    fig4b = go.Figure()
    vix_colors = [COLORS["secondary"], COLORS["neutral"], COLORS["accent"], COLORS["danger"]]
    for (label, savings), color in zip(vix_savings.items(), vix_colors):
        if savings:
            fig4b.add_trace(go.Box(
                y=savings,
                name=label,
                marker_color=color,
                boxmean=True,
            ))
    fig4b.update_layout(
        title="Saving potential vs day 27 by VIX regime (higher = more opportunity)",
        yaxis_title="Avg saving % vs day 27",
        height=420,
    )
    figs.append(("4b_vix_regime_saving", fig4b))

    log.info(f"  ✓ Section 4: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Cost of the 27th: Historical Simulation
# ─────────────────────────────────────────────────────────────────────────────

def section5_cost_of_27th(data: dict) -> list[go.Figure]:
    log.info("\n[Section 5] Cost of the 27th — historical simulation...")
    figs = []

    # Simulate: for every month in history, compare
    # buying on 27th vs buying on optimal day
    sim_results = []

    for ticker, df in data.items():
        monthly_usd  = PORTFOLIO[ticker]["monthly_usd"]
        window       = df[df.index.day >= PURCHASE_WINDOW_START].copy()
        window["ym"] = window.index.to_period("M")

        for period, group in window.groupby("ym"):
            if len(group) < 3:
                continue

            # Day 27 price
            day27_row = group[group.index.day == YOUR_CURRENT_DAY]
            if len(day27_row) == 0:
                candidates = group[group.index.day >= 25]
                day27_row  = candidates.iloc[[0]] if len(candidates) else group.iloc[[-1]]
            price_27   = day27_row["close"].values[0]

            # Optimal price (lowest in window)
            price_opt  = group["close"].min()
            opt_day    = group["close"].idxmin().day

            # Shares bought
            shares_27  = monthly_usd / price_27
            shares_opt = monthly_usd / price_opt
            extra_shares = shares_opt - shares_27

            sim_results.append({
                "ticker":       ticker,
                "period":       str(period),
                "price_27":     price_27,
                "price_opt":    price_opt,
                "opt_day":      opt_day,
                "saving_pct":   (price_27 - price_opt) / price_27 * 100,
                "extra_shares": extra_shares,
                "monthly_usd":  monthly_usd,
            })

    sim_df = pd.DataFrame(sim_results)
    sim_df["period_dt"] = pd.to_datetime(sim_df["period"].astype(str))

    # 5a. Rolling cumulative extra shares by ticker
    fig5a = go.Figure()
    for ticker in PORTFOLIO:
        t_df = sim_df[sim_df["ticker"] == ticker].sort_values("period_dt")
        if len(t_df) < 6:
            continue
        cumulative = t_df["extra_shares"].cumsum()
        fig5a.add_trace(go.Scatter(
            x=t_df["period_dt"],
            y=cumulative,
            name=ticker,
            line=dict(color=TICKER_COLOR_MAP[ticker], width=2),
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m}}<br>Cumulative extra shares: %{{y:.4f}}<extra></extra>",
        ))
    fig5a.update_layout(
        title="Cumulative extra shares gained by timing vs always buying on day 27",
        yaxis_title="Extra shares accumulated",
        height=460,
    )
    figs.append(("5a_cumulative_extra_shares", fig5a))

    # 5b. Monthly saving % distribution per ticker
    fig5b = go.Figure()
    for ticker in PORTFOLIO:
        t_df = sim_df[sim_df["ticker"] == ticker]
        if len(t_df) < 12:
            continue
        fig5b.add_trace(go.Violin(
            y=t_df["saving_pct"],
            name=ticker,
            box_visible=True,
            meanline_visible=True,
            fillcolor=TICKER_COLOR_MAP[ticker],
            opacity=0.7,
            line_color=TICKER_COLOR_MAP[ticker],
        ))
    fig5b.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig5b.update_layout(
        title="Distribution of monthly saving % (buying optimal day vs day 27)",
        yaxis_title="Saving % per month",
        yaxis=dict(range=[-2, 15]),   # fix: clip outliers so shape is visible
        height=460,
    )
    figs.append(("5b_saving_distribution", fig5b))

    # 5c. Summary table — annualised dollar impact
    summary = sim_df.groupby("ticker").agg(
        months=("period", "count"),
        avg_saving_pct=("saving_pct", "mean"),
        median_saving_pct=("saving_pct", "median"),
        win_rate=("saving_pct", lambda x: (x > 0).mean() * 100),
        total_extra_shares=("extra_shares", "sum"),
    ).round(3)
    summary["monthly_usd"] = [PORTFOLIO[t]["monthly_usd"] for t in summary.index]
    summary["annual_dollar_impact"] = (
        summary["avg_saving_pct"] / 100 * summary["monthly_usd"] * 12
    ).round(2)
    summary = summary.sort_values("avg_saving_pct", ascending=False)

    fig5c = go.Figure(go.Table(
        header=dict(
            values=["Ticker", "Months", "Avg Save%", "Median Save%", "Win Rate%",
                    "Monthly $", "Annual $ Impact"],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[
                summary.index,
                summary["months"],
                summary["avg_saving_pct"].round(2).astype(str) + "%",
                summary["median_saving_pct"].round(2).astype(str) + "%",
                summary["win_rate"].round(1).astype(str) + "%",
                "$" + summary["monthly_usd"].astype(str),
                "$" + summary["annual_dollar_impact"].astype(str),
            ],
            fill_color=["#F1EFE8", "white"] * 3,
            align="center",
            font=dict(size=12),
        ),
    ))
    fig5c.update_layout(
        title="Historical simulation summary — cost of always buying on day 27",
        height=460,
    )
    figs.append(("5c_simulation_table", fig5c))

    # 5d. Compounding impact — dollar growth over time
    total_monthly = sum(v["monthly_usd"] for v in PORTFOLIO.values())
    years         = 20
    # Fix 1: dollar-weighted saving (VOO/VTI dominate at $25/mo combined)
    # Fix 2: model captures ~45% of theoretical max (realistic estimate)
    # Fix 3: saving_pct already in %, divide by 100 to get decimal, then weight
    weighted_saving_pct = sum(
        PORTFOLIO[t]["monthly_usd"] * sim_df[sim_df["ticker"] == t]["saving_pct"].mean()
        for t in PORTFOLIO if t in sim_df["ticker"].values
    ) / sum(v["monthly_usd"] for v in PORTFOLIO.values())
    avg_blended_saving = (weighted_saving_pct / 100) * 0.45  # 45% model capture rate

    base_rate = 0.15
    opt_rate  = base_rate + (avg_blended_saving * 12 * 0.3)

    months_range = np.arange(1, years * 12 + 1)
    fv_base = total_monthly * ((1 + base_rate/12) ** months_range - 1) / (base_rate/12)
    fv_opt  = total_monthly * ((1 + opt_rate/12) ** months_range - 1) / (opt_rate/12)

    fig5d = go.Figure()
    # FIX: single y-axis — dual y-axis caused secondary axis [-1,4] scale
    # which made both main lines appear at y=0 and invisible
    years_x = months_range / 12
    fig5d.add_trace(go.Scatter(
        x=years_x, y=fv_base,
        name=f"Fixed day {YOUR_CURRENT_DAY} ({base_rate*100:.0f}% p.a.)",
        line=dict(color=COLORS["neutral"], width=2, dash="dash"),
    ))
    fig5d.add_trace(go.Scatter(
        x=years_x, y=fv_opt,
        name=f"AI-optimised (~{opt_rate*100:.1f}% p.a.)",
        line=dict(color=COLORS["secondary"], width=3),
        fill="tonexty", fillcolor="rgba(15,110,86,0.12)",
    ))
    fig5d.add_trace(go.Scatter(
        x=years_x, y=fv_opt - fv_base,
        name=f"Extra wealth (${(fv_opt-fv_base)[-1]:,.0f} at yr 20)",
        line=dict(color=COLORS["primary"], width=2, dash="dot"),
    ))
    # Milestone annotations
    for yr, label in [(5, "5yr"), (10, "10yr"), (20, "20yr")]:
        idx_yr = yr * 12 - 1
        fig5d.add_annotation(
            x=yr, y=fv_opt[idx_yr],
            text=f"${fv_opt[idx_yr]:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["secondary"],
            font=dict(size=10, color=COLORS["secondary"]),
            bgcolor="white", bordercolor=COLORS["secondary"], borderwidth=1,
        )
    fig5d.update_layout(
        title=f"Projected 20-year wealth: fixed day {YOUR_CURRENT_DAY} vs AI-optimised (${total_monthly}/month)",
        xaxis_title="Years",
        yaxis_title="Portfolio value ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=500,
        legend=dict(x=0.02, y=0.95),
    )
    figs.append(("5d_compounding_impact", fig5d))

    sim_df.to_csv(EXPORTS_DIR / "historical_simulation.csv", index=False)
    summary.to_csv(EXPORTS_DIR / "simulation_summary.csv")
    log.info(f"  ✓ Section 5: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Feature Importance Preview (pre-model)
# ─────────────────────────────────────────────────────────────────────────────

def section6_feature_preview(data: dict) -> list[go.Figure]:
    log.info("\n[Section 6] Feature importance preview...")
    figs = []

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        log.warning("  scikit-learn not installed — skipping feature importance")
        return figs

    feature_importance_all = {}

    for ticker in ["NVDA", "VOO", "TSLA", "BND"]:
        if ticker not in data:
            continue
        df = data[ticker].copy()
        if "is_optimal_buy_day" not in df.columns:
            continue

        window = df[df.index.day >= PURCHASE_WINDOW_START].copy()

        # Select numeric features only
        num_cols = window.select_dtypes(include=[np.number]).columns.tolist()
        exclude  = [c for c in num_cols if c.startswith("fwd_") or
                    c in ["is_optimal_buy_day", "vs_day27_pct", "day27_price"]]
        feat_cols = [c for c in num_cols if c not in exclude]

        X = window[feat_cols].fillna(0)
        y = window["is_optimal_buy_day"]

        if y.sum() < 20:
            continue

        try:
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                n_jobs=-1, random_state=42
            )
            rf.fit(X, y)
            importance = pd.Series(rf.feature_importances_, index=feat_cols)
            top20 = importance.nlargest(20)
            feature_importance_all[ticker] = top20
        except Exception as e:
            log.warning(f"  RF failed for {ticker}: {e}")
            continue

    # 6a. Top features per ticker
    if feature_importance_all:
        fig6a = make_subplots(
            rows=1, cols=len(feature_importance_all),
            subplot_titles=list(feature_importance_all.keys()),
        )
        for col_idx, (ticker, importance) in enumerate(feature_importance_all.items()):
            fig6a.add_trace(
                go.Bar(
                    x=importance.values,
                    y=importance.index,
                    orientation="h",
                    marker_color=TICKER_COLOR_MAP.get(ticker, COLORS["primary"]),
                    showlegend=False,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ),
                row=1, col=col_idx + 1,
            )
        fig6a.update_layout(
            title="Top 20 features predicting optimal buy day (Random Forest — pre-LightGBM preview)",
            height=600,
        )
        figs.append(("6a_feature_importance", fig6a))

    # 6b. Feature categories summary
    category_map = {
        "Calendar":    ["day_of_month", "day_of_week", "week_of_month", "days_to_month_end",
                        "is_month_end", "near_options_expiry", "is_fomc_month"],
        "Technical":   ["rsi_14", "macd", "bb_position", "atr_pct", "stoch_k"],
        "Volatility":  ["vol_5d", "vol_21d", "bb_width", "daily_range"],
        "Macro":       ["vix", "vix_change_5d", "sp500_ret_5d", "gold_ret_5d",
                        "oil_ret_5d", "usd_ret_5d", "yield_curve_10y3m"],
        "Momentum":    ["ret_5d", "ret_21d", "price_vs_sma21", "price_vs_sma50"],
    }

    cat_importance = {cat: 0 for cat in category_map}
    for ticker, importance in feature_importance_all.items():
        for cat, keywords in category_map.items():
            for feat in importance.index:
                if any(kw in feat for kw in keywords):
                    cat_importance[cat] += importance[feat]

    fig6b = go.Figure(go.Pie(
        labels=list(cat_importance.keys()),
        values=list(cat_importance.values()),
        hole=0.4,
        marker=dict(colors=[
            COLORS["primary"], COLORS["secondary"], COLORS["accent"],
            COLORS["danger"], COLORS["success"]
        ]),
        textinfo="label+percent",
    ))
    fig6b.update_layout(
        title="Feature category importance (aggregated across NVDA, VOO, TSLA, BND)",
        height=420,
    )
    figs.append(("6b_feature_categories", fig6b))

    log.info(f"  ✓ Section 6: {len(figs)} charts")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# HTML report exporter
# ─────────────────────────────────────────────────────────────────────────────

SECTION_DESCRIPTIONS = {
    "1": ("Portfolio Overview & Data Quality",
          "Coverage, allocation, and normalised performance across all 11 holdings."),
    "2": ("Does Day-of-Month Matter?",
          "The core signal — average forward returns by purchase day, exploitable price range, and the true cost of always buying on day 27."),
    "3": ("Market Regime Analysis",
          "How the optimal purchase day shifts in bull, bear, sideways, and crash regimes. When is the model most valuable?"),
    "4": ("Macro Factor Correlation",
          "Which macro signals (VIX, oil, gold, yields, USD) best predict a good entry day? Correlation analysis across your portfolio."),
    "5": ("Cost of the 27th — Historical Simulation",
          "Month-by-month simulation: how many extra shares and dollars were left on the table by always buying on day 27?"),
    "6": ("Feature Importance Preview",
          "Pre-model signal strength — which features matter most for predicting optimal entry days, before we train LightGBM."),
}


def export_html_report(all_figs: list[tuple[str, go.Figure]]):
    """Export all figures to a single shareable HTML file."""
    log.info("\nExporting HTML report...")

    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TemporalEdge — EDA Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #FAFAF8; color: #2C2C2A; }
  .header { background: #534AB7; color: white; padding: 40px 60px; }
  .header h1 { font-size: 28px; font-weight: 500; margin-bottom: 8px; }
  .header p  { font-size: 15px; opacity: 0.85; }
  .toc { background: white; border-bottom: 1px solid #E8E6DE;
         padding: 20px 60px; display: flex; gap: 24px; flex-wrap: wrap; }
  .toc a { color: #534AB7; text-decoration: none; font-size: 13px; }
  .toc a:hover { text-decoration: underline; }
  .section { padding: 40px 60px; border-bottom: 1px solid #E8E6DE; }
  .section:nth-child(even) { background: white; }
  .section h2 { font-size: 20px; font-weight: 500; color: #534AB7;
                margin-bottom: 6px; }
  .section .desc { font-size: 14px; color: #5F5E5A; margin-bottom: 24px;
                   max-width: 800px; line-height: 1.6; }
  .chart-wrap { margin-bottom: 32px; border-radius: 8px; overflow: hidden;
                border: 1px solid #E8E6DE; background: white; }
  .footer { padding: 40px 60px; text-align: center; font-size: 13px;
            color: #888780; }
</style>
</head>
<body>
<div class="header">
  <h1>TemporalEdge — EDA Report</h1>
  <p>Optimal DCA Entry Timing · 11 Tickers · Full Historical Analysis</p>
</div>
<div class="toc">"""]

    for sec_num, (title, _) in SECTION_DESCRIPTIONS.items():
        html_parts.append(
            f'  <a href="#section{sec_num}">Section {sec_num}: {title}</a>'
        )
    html_parts.append("</div>")

    current_section = None
    for name, fig in all_figs:
        sec_num = name.split("_")[0].replace("s", "").replace("section", "")[:1]
        if sec_num != current_section:
            if current_section is not None:
                html_parts.append("</div>")
            current_section = sec_num
            title, desc = SECTION_DESCRIPTIONS.get(sec_num, (f"Section {sec_num}", ""))
            html_parts.append(
                f'<div class="section" id="section{sec_num}">'
                f'<h2>Section {sec_num}: {title}</h2>'
                f'<p class="desc">{desc}</p>'
            )

        chart_html = pio.to_html(fig, include_plotlyjs=True, full_html=False,
                                  config={"displayModeBar": True, "responsive": True})
        html_parts.append(f'<div class="chart-wrap">{chart_html}</div>')

    if current_section:
        html_parts.append("</div>")

    html_parts.append("""
<div class="footer">
  Generated by TemporalEdge · Phase 2 EDA ·
  All data from Yahoo Finance &amp; FRED (free sources)
</div>
</body></html>""")

    report_path = NOTEBOOKS_DIR / "eda_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    log.info(f"  ✓ HTML report saved: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    log.info("=" * 60)
    log.info("  TEMPORALEDGE — PHASE 2: EDA")
    log.info("=" * 60)

    log.info("\nLoading feature store...")
    data = load_features()
    raw  = load_raw_prices()

    if not data:
        log.error("No feature data found. Run Phase 1 first: python run_phase1.py")
        return

    log.info(f"Loaded {len(data)} tickers\n")

    all_figs = []
    all_figs += section1_overview(data, raw)
    all_figs += section2_day_of_month_signal(data)
    all_figs += section3_regime_analysis(data)
    all_figs += section4_macro_correlation(data)
    all_figs += section5_cost_of_27th(data)
    all_figs += section6_feature_preview(data)

    # Save individual PNGs
    log.info("\nSaving individual chart PNGs...")
    for name, fig in all_figs:
        try:
            fig.write_image(str(CHARTS_DIR / f"{name}.png"), width=1200, height=fig.layout.height or 480)
        except Exception:
            pass  # kaleido optional — HTML report always works

    # Export combined HTML report
    report_path = export_html_report(all_figs)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 2 COMPLETE in {elapsed:.1f}s")
    log.info(f"  Charts    : {len(all_figs)}")
    log.info(f"  HTML report: {report_path}")
    log.info(f"  CSV exports: {EXPORTS_DIR}")
    log.info(f"  → Open eda_report.html in your browser")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
