"""
TemporalEdge — Evidence tab renderer.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
from .utils import get_active_portfolio, load_simulation_results, PLOTLY_LAYOUT, AXIS_DEFAULTS, TIER_COLORS


# ── AI Explanation helper ─────────────────────────────────────────────────────

def _ai_explanation(section_key: str, context: dict) -> str:
    """
    Call Claude via Anthropic API to generate a plain-English explanation
    of the evidence section. Results cached in session_state.
    """
    cache_key = f"evidence_ai_{section_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    prompts = {
        "summary_table": f"""You are explaining a DCA timing model to a retail investor.
Here is the backtest summary data:
- Average win rate (model beats fixed day 27): {context.get('avg_win', 0):.1f}%
- Average monthly saving: {context.get('avg_save', 0):+.3f}%
- Tickers with Tier A (high trust): {context.get('tier_a', [])}
- Tickers with Tier B (moderate trust): {context.get('tier_b', [])}

Write 3–4 sentences in plain English explaining:
1. What the win rate means in practice
2. What the average saving means in dollar terms for someone investing $100/month
3. Why some tickers have higher trust than others
Be honest about uncertainty. No bullet points. Conversational tone.""",

        "consistency": f"""You are explaining a model reliability check to a retail investor.
The chart shows how consistent the model's win rate is between two independent validation phases.
- Tickers within ±5pp (consistent): {context.get('consistent', [])}
- Tickers with larger delta: {context.get('inconsistent', [])}
- Average delta across all tickers: {context.get('avg_delta', 0):+.1f} percentage points

Write 3–4 sentences explaining:
1. Why consistency between phases matters (overfitting risk)
2. What it means when two independent tests agree
3. What the user should take away from this chart
No bullet points. Plain English. Honest about what this does and doesn't prove.""",

        "compounding": f"""You are explaining long-term compounding to a retail investor.
The chart shows 20-year wealth projections for $55/month invested.
- Average monthly saving from model timing: {context.get('avg_save', 0):+.3f}%
- Conservative scenario annual return: 10%
- Base case annual return: 15%
- Optimistic scenario annual return: higher with full timing capture

Write 3–4 sentences explaining:
1. Why even a small monthly edge compounds to meaningful wealth
2. What the gap between model-timed and fixed day 27 represents in dollars
3. The honest caveat — this is a projection, not a guarantee
No bullet points. Conversational. Grounded.""",
    }

    prompt = prompts.get(section_key, "Explain this chart in plain English.")

    # Use Groq — same as the rest of the pipeline
    import os
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "Add GROQ_API_KEY to Streamlit secrets or environment variables."

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "llama3-8b-8192",
                "max_tokens": 300,
                "temperature": 0.4,
                "messages": [
                    {"role": "system", "content": "You are a helpful financial educator explaining investment data to retail investors in plain English. Be concise, honest, and avoid hype."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=20,
        )
        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text[:150]}"
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        st.session_state[cache_key] = text
        return text
    except Exception as e:
        return f"Connection error: {str(e)[:100]}"


def _render_ai_panel(section_key: str, context: dict, label: str = "🤖 AI Explanation"):
    """Render an expandable AI explanation panel below a chart."""
    with st.expander(label, expanded=False):
        with st.spinner("Generating explanation..."):
            explanation = _ai_explanation(section_key, context)
        if explanation:
            st.markdown(
                f"""<div style="
                    background: rgba(77,158,245,0.06);
                    border-left: 3px solid #4D9EF5;
                    border-radius: 0 6px 6px 0;
                    padding: 0.85rem 1.1rem;
                    font-size: 0.83rem;
                    line-height: 1.65;
                    color: #C8CEDB;
                    font-family: var(--font-body, sans-serif);
                ">{explanation}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.caption("AI explanation unavailable — check API connection.")

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
    st.dataframe(df, hide_index=True)

    # AI explanation for summary table
    tier_a = [r["Ticker"] for r in rows if r["Tier"] == "A"]
    tier_b = [r["Ticker"] for r in rows if r["Tier"] == "B"]
    avg_win  = np.mean([sim[t]["summary"].get("win_rate_pct", 0) for t in sim])
    avg_save = np.mean([sim[t]["summary"].get("avg_saving_pct", 0) for t in sim])
    _render_ai_panel(
        "summary_table",
        {"avg_win": avg_win, "avg_save": avg_save, "tier_a": tier_a, "tier_b": tier_b},
        "🤖 What does this table mean?",
    )

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
    st.plotly_chart(fig3)

    # AI explanation for consistency chart
    consistent   = [t for t, d in zip(tickers, deltas) if abs(d) <= 5]
    inconsistent = [t for t, d in zip(tickers, deltas) if abs(d) > 5]
    avg_delta    = np.mean(deltas)
    _render_ai_panel(
        "consistency",
        {"consistent": consistent, "inconsistent": inconsistent, "avg_delta": avg_delta},
        "🤖 Why does this consistency check matter?",
    )

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
    st.plotly_chart(fig4)

    # AI explanation for compounding chart
    _render_ai_panel(
        "compounding",
        {"avg_save": avg_save},
        "🤖 What does this compounding chart mean for me?",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


# ── Sidebar: Portfolio Setup ──────────────────────────────────────────────────