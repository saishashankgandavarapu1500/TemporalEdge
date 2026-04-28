"""
TemporalEdge — Evidence tab renderer.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from .utils import get_active_portfolio, load_simulation_results, PLOTLY_LAYOUT, AXIS_DEFAULTS, TIER_COLORS

def render_evidence():
    st.markdown('<div class="section-label">historical evidence — walk-forward backtest + monte carlo</div>',
                unsafe_allow_html=True)

    sim = load_simulation_results()
    if sim is None:
        st.info("Run Phase 4 first: `python run_phase4.py`")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    phase3 = {
        "VOO":77.5,"VTI":76.3,"NVDA":71.0,"AAPL":79.0,"SCHD":77.5,
        "VXUS":70.1,"TSLA":71.4,"BND":74.5,"VYM":78.2,"VEA":76.7,"VWO":77.0,
    }
    rows = []
    for ticker, data in sim.items():
        s    = data.get("summary", {})
        p3   = phase3.get(ticker, 0)
        p4   = s.get("win_rate_pct", 0)
        delta = p4 - p3
        save  = s.get("avg_saving_pct", 0)
        p     = s.get("percentiles", {})
        tier  = s.get("tier", "B")
        ann   = s.get("avg_dollar_per_year", 0)
        rows.append({
            "Ticker": ticker,
            "Tier": tier,
            "P3 Win%": f"{p3:.1f}%",
            "P4 Win%": f"{p4:.1f}%",
            "Δ": f"{delta:+.1f}pp",
            "Avg Save": f"{save:+.3f}%",
            "P10": f"{p.get('p10',0):+.2f}%",
            "P50": f"{p.get('p50',0):+.2f}%",
            "P90": f"{p.get('p90',0):+.2f}%",
            "$/yr": f"+${ann:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Phase 3 vs Phase 4 consistency chart ──────────────────────────────
    st.markdown('<div class="section-label">phase 3 vs phase 4 consistency — key reliability test</div>',
                unsafe_allow_html=True)

    tickers  = list(sim.keys())
    p3_wins  = [phase3.get(t, 0) for t in tickers]
    p4_wins  = [sim[t]["summary"].get("win_rate_pct", 0) for t in tickers]
    deltas   = [p4-p3 for p4, p3 in zip(p4_wins, p3_wins)]
    colors   = ["#2DB37A" if abs(d)<=5 else "#E8A020" if abs(d)<=10 else "#E05C5C" for d in deltas]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=tickers, y=deltas, marker_color=colors, name="P4 - P3 delta"))
    fig3.add_hline(y=5,  line_dash="dot", line_color="#2DB37A", opacity=0.4)
    fig3.add_hline(y=-5, line_dash="dot", line_color="#2DB37A", opacity=0.4)
    fig3.add_hrect(y0=-5, y1=5, fillcolor="rgba(45,179,122,0.04)", line_width=0)
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Win rate delta: Phase 4 vs Phase 3 (green band = consistent signal)",
                   font=dict(size=12)),
        xaxis=dict(**AXIS_DEFAULTS),
        yaxis=dict(**AXIS_DEFAULTS, title="Δ percentage points"),
        height=320,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── 3-scenario compounding chart ──────────────────────────────────────
    st.markdown('<div class="section-label">20-year compounding — 3 scenarios ($55/month)</div>',
                unsafe_allow_html=True)

    avg_save = np.mean([sim[t]["summary"].get("avg_saving_pct", 0) for t in sim])
    base     = 0.15
    total_mo = 55
    scenarios = [
        ("Conservative", 0.10, 0.25, "#5C6378"),
        ("Base case",    0.15, 0.45, "#2DB37A"),
        ("Optimistic",   0.15, 1.00, "#4D9EF5"),
    ]

    fig4 = go.Figure()
    months = np.arange(1, 241)
    for name, base_r, cap, color in scenarios:
        opt_r = base_r + (avg_save/100 * cap * 12 * 0.3)
        r     = opt_r/12
        fv    = total_mo * ((1+r)**months - 1) / r
        fig4.add_trace(go.Scatter(
            x=months/12, y=fv, name=f"{name} ({opt_r*100:.1f}%)",
            line=dict(color=color, width=2, dash="dash" if name=="Conservative" else "solid"),
        ))
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="20-year wealth projection — fixed day 27 baseline (grey dashed)",
                   font=dict(size=12)),
        xaxis=dict(**AXIS_DEFAULTS, title="Years"),
        yaxis=dict(**AXIS_DEFAULTS, tickprefix="$", tickformat=",.0f"),
        height=340,
    )
    # Add fixed-day-27 baseline
    r_b  = 0.15/12
    fv_b = total_mo * ((1+r_b)**months - 1) / r_b
    fig4.add_trace(go.Scatter(
        x=months/12, y=fv_b, name="Fixed day 27 (15%)",
        line=dict(color="#3A3F52", width=1.5, dash="dash"),
    ))
    st.plotly_chart(fig4, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────


# ── Sidebar: Portfolio Setup ──────────────────────────────────────────────────