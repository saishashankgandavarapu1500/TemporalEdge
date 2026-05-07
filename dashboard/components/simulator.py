"""
TemporalEdge — Simulator tab renderer.
"""
import sys
import time
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .utils import get_active_portfolio, load_execution_plan, load_simulation_results, MODEL_PROXY, PLOTLY_LAYOUT, AXIS_DEFAULTS, TIER_COLORS, TIER_LABELS

def _render_advisory(placeholder, advisory: str, key_factor: str, action: str):
    """Render the LLM advisory block into a placeholder."""
    action_colors = {"act": "#2DB37A", "consider": "#E8A020", "skip": "#6B7280"}
    color = action_colors.get(action, "#9AA0B4")
    kf_html = (f'<span style="font-family:var(--mono);font-size:0.68rem;'
               f'color:#5C6378;background:#252A38;padding:2px 7px;'
               f'border-radius:3px;margin-left:0.5rem;">↳ {key_factor}</span>'
               if key_factor else "")
    placeholder.markdown(f"""
    <div style="font-size:0.82rem;color:#9AA0B4;line-height:1.6;
                border-top:1px solid #252A38;padding-top:0.6rem;margin-top:0.4rem;">
        {advisory}{kf_html}
    </div>
    """, unsafe_allow_html=True)


def render_simulator():
    st.markdown('<div class="section-label">portfolio simulator</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker", value="VOO",
            help="Any ticker on Yahoo Finance — portfolio or any stock/ETF",
            placeholder="e.g. VOO, ARKK, MSFT, GME"
        ).upper().strip()
    with col2:
        default_mo = get_active_portfolio().get(ticker_input, {}).get("monthly_usd", 50)
        monthly_usd = st.number_input("Monthly $", min_value=1, max_value=50000,
                                       value=default_mo, step=5)
    with col3:
        horizon = st.slider("Years", min_value=1, max_value=30, value=10)

    is_portfolio = ticker_input in get_active_portfolio()

    # Check if ticker is in precomputed S&P 500 results
    from .utils import PRECOMPUTED_DIR
    _precomputed_tickers = {
        f.stem.replace("_result", "")
        for f in PRECOMPUTED_DIR.glob("*_result.json")
    }
    is_precomputed = ticker_input in _precomputed_tickers

    # Gate: unknown tickers need explicit user consent before live training
    _needs_live    = not is_portfolio and not is_precomputed
    _live_approved = st.session_state.get(f"live_approved_{ticker_input}", False)

    if _needs_live and not _live_approved:
        st.warning(f"**{ticker_input}** is not in our analysed database.")
        st.caption("Live analysis requires downloading data and training a model (~3-5 min first time, then cached for 30 days).")
        if st.button(f"🔬 Run Live Analysis for {ticker_input}"):
            st.session_state[f"live_approved_{ticker_input}"] = True
            st.rerun()
        return

    run_label = "▶  Run Simulation" if is_portfolio else f"▶  Run Simulation for {ticker_input}"
    run_sim   = st.button(run_label)

    if not is_portfolio and ticker_input:
        if _needs_live:
            st.markdown(f"""
            <div style="background:rgba(77,158,245,0.07);border:1px solid rgba(77,158,245,0.2);
                        border-radius:6px;padding:0.65rem 1rem;margin-bottom:0.75rem;
                        font-size:0.78rem;color:#4D9EF5;font-family:var(--mono)">
                {ticker_input} is not in your portfolio.
                Running will: download history → build 121 features → train LightGBM
                → 1,000 Monte Carlo runs → LLM advisory. ~1-3 min first time, cached after.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(45,179,122,0.07);border:1px solid rgba(45,179,122,0.2);
                        border-radius:6px;padding:0.65rem 1rem;margin-bottom:0.75rem;
                        font-size:0.78rem;color:#2DB37A;font-family:var(--mono)">
                {ticker_input} is pre-analysed — results load instantly from cache.
            </div>
            """, unsafe_allow_html=True)

    # Check portfolio cache first
    sim_results  = load_simulation_results()
    cached_result = None

    if is_portfolio and sim_results and ticker_input in sim_results:
        cached_result = sim_results[ticker_input]
    else:
        # Use unified loader — checks on-demand cache then precomputed JSON
        from .utils import load_ondemand_result
        cached_result = load_ondemand_result(ticker_input)

    # Auto-show cached results without needing to click Run
    if cached_result and not run_sim:
        st.markdown(f"""
        <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;margin-bottom:0.5rem">
            Showing cached result — click Run to refresh
        </div>
        """, unsafe_allow_html=True)

    if run_sim or cached_result:
        result = cached_result

        if run_sim:
            # If precomputed result exists, just use it — don't re-run pipeline
            if cached_result and is_precomputed and not is_portfolio:
                result = cached_result
                st.toast(f"{ticker_input} loaded from precomputed cache", icon="✅")
            elif is_portfolio:
                with st.spinner(f"Running Monte Carlo for {ticker_input}..."):
                    try:
                        from src.simulation.monte_carlo import run_simulation
                        r = run_simulation(ticker_input, float(monthly_usd))
                        result = r
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        return
            else:
                # Full on-demand pipeline — session_state prevents rerun kills
                _rkey = f"od_result_{ticker_input}"
                _running = f"od_running_{ticker_input}"

                if not st.session_state.get(_running, False):
                    st.session_state[_running] = True
                    st.session_state.pop(_rkey, None)

                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    def update_progress(fraction: float, message: str):
                        try:
                            progress_bar.progress(min(fraction, 1.0))
                            status_text.markdown(
                                f'<div style="font-family:var(--mono);font-size:0.75rem;'
                                f'color:#9AA0B4">{message}</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            pass

                    try:
                        from src.pipeline.on_demand import run_on_demand
                        _res = run_on_demand(
                            ticker_input,
                            monthly_usd=float(monthly_usd),
                            horizon_years=horizon,
                            n_runs=1000,
                            progress_cb=update_progress,
                        )
                        st.session_state[_rkey] = _res
                        progress_bar.progress(1.0)
                        status_text.empty()
                    except Exception as e:
                        st.session_state[_running] = False
                        st.error(f"Pipeline failed: {e}")
                        return
                    finally:
                        st.session_state[_running] = False

                if _rkey in st.session_state:
                    result = st.session_state[_rkey]
                else:
                    st.info("Pipeline running — please wait...")
                    return

        if result is None:
            st.info("No results. Click Run Simulation.")
            return

        if "error" in result and result["error"]:
            st.error(f"❌ {result['error']}")
            return

        summary = result.get("summary", {})
        if not summary:
            st.error("Invalid result structure.")
            return

        win    = summary.get("win_rate_pct", 0)
        save   = summary.get("avg_saving_pct", 0)
        p      = summary.get("percentiles", {})
        tier   = summary.get("tier", "B")
        proj   = summary.get("projections", {})
        regime = summary.get("regime_stats", {})
        meta   = summary.get("ticker_meta", {})

        # Trust score fields
        trust_score    = summary.get("trust_score", None)
        tier_label     = summary.get("tier_label", "")
        trust_reasons  = summary.get("trust_reasons", [])
        trust_penalties= summary.get("trust_penalties", [])
        trust_summary  = summary.get("trust_summary", "")
        proj_note      = summary.get("projection_note", "")
        show_optimistic= summary.get("show_optimistic", True)
        proj_scenarios = summary.get("projections_by_scenario", {})
        clip_bounds    = summary.get("clip_bounds", {})
        n_clipped      = summary.get("clipped_values", 0)

        # Info banner for non-portfolio tickers
        if not is_portfolio and meta:
            beta   = float(meta.get("beta", 1.0) or 1.0)
            sector = meta.get("sector", "Unknown")
            name   = meta.get("name", ticker_input)
            elapsed = result.get("elapsed_s", 0)
            st.markdown(f"""
            <div style="background:rgba(45,179,122,0.07);border:1px solid rgba(45,179,122,0.2);
                        border-radius:6px;padding:0.65rem 1rem;margin-bottom:0.75rem;
                        font-size:0.78rem;color:#2DB37A;font-family:var(--mono)">
                ✓ {name} | sector: {sector} | beta: {beta:.2f}
                | trained on real {ticker_input} data in {elapsed:.0f}s
            </div>
            """, unsafe_allow_html=True)

        # Trust score panel
        if trust_score is not None:
            tier_color = {"A":"#2DB37A","B":"#E8A020","C":"#E05C5C"}.get(tier,"#9AA0B4")
            trust_color = "#2DB37A" if trust_score>=65 else "#E8A020" if trust_score>=40 else "#E05C5C"
            reasons_html = "".join([
                f'<span style="color:#2DB37A;margin-right:0.75rem">✓ {r}</span>'
                for r in trust_reasons
            ])
            penalties_html = "".join([
                f'<span style="color:#E05C5C;margin-right:0.75rem">✗ {r}</span>'
                for r in trust_penalties
            ])
            clip_note = ""
            if n_clipped > 0 and clip_bounds:
                clip_note = (f'<div style="margin-top:0.4rem;font-size:0.70rem;color:#5C6378">'
                             f'{n_clipped} extreme values clipped to '
                             f'[{clip_bounds.get("low",0):.1f}%, '
                             f'+{clip_bounds.get("high",0):.1f}%] — realistic bounds for this asset type'
                             f'</div>')
            st.markdown(f"""
            <div style="background:#141720;border:1px solid #252A38;border-radius:8px;
                        padding:1rem 1.25rem;margin-bottom:1rem">
              <div style="display:flex;justify-content:space-between;align-items:baseline;
                          margin-bottom:0.6rem">
                <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                            letter-spacing:0.1em">CAN I TRUST THIS RESULT?</div>
                <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378">
                  Tier <span style="color:{tier_color};font-weight:600">{tier}</span>
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:0.5rem">
                <div>
                  <span style="font-family:var(--mono);font-size:2rem;font-weight:600;
                               color:{trust_color}">{trust_score}</span>
                  <span style="font-family:var(--mono);font-size:0.78rem;color:#5C6378">/100</span>
                </div>
                <div>
                  <div style="background:#252A38;border-radius:4px;height:8px;width:200px">
                    <div style="background:{trust_color};border-radius:4px;height:8px;
                                width:{trust_score*2}px"></div>
                  </div>
                  <div style="font-size:0.75rem;color:#9AA0B4;margin-top:4px">{tier_label}</div>
                </div>
              </div>
              <div style="font-size:0.75rem;line-height:1.6">
                {reasons_html}
              </div>
              {f'<div style="font-size:0.75rem;line-height:1.6;margin-top:0.3rem">{penalties_html}</div>' if penalties_html else ""}
              {clip_note}
            </div>
            """, unsafe_allow_html=True)

        # Projection note for Tier B/C
        if proj_note:
            note_color = "#E05C5C" if tier == "C" else "#E8A020"
            st.markdown(f"""
            <div style="background:rgba(224,92,92,0.06) if tier=='C' else rgba(232,160,32,0.06);
                        border:1px solid {note_color}40;border-radius:6px;
                        padding:0.6rem 1rem;margin-bottom:0.75rem;
                        font-size:0.75rem;color:{note_color}">
                {proj_note}
            </div>
            """, unsafe_allow_html=True)

        tier_color = {"A": "#2DB37A", "B": "#E8A020", "C": "#6B7280"}.get(tier, "#9AA0B4")
        tier_label = {"A": "High precision", "B": "Medium precision", "C": "Low precision"}.get(tier, "")

        # ── Summary stats ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-val" style="color:{'#2DB37A' if win>=65 else '#E8A020' if win>=55 else '#6B7280'}">{win:.1f}%</div>
            <div class="stat-lbl">Win rate vs day 27</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" style="color:{'#2DB37A' if save>0 else '#E05C5C'}">{save:+.3f}%</div>
            <div class="stat-lbl">Avg saving per month</div>
          </div>
          <div class="stat-box">
            <div class="stat-val stat-white" style="font-size:1rem">
              {p.get('p10',0):+.2f}% / {p.get('p50',0):+.2f}% / {p.get('p90',0):+.2f}%
            </div>
            <div class="stat-lbl">P10 / P50 / P90</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" style="color:{tier_color}">Tier {tier}</div>
            <div class="stat-lbl">{tier_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column charts ──────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            # Compounding fan chart
            base_rate = 0.15
            opt_rate  = base_rate + (save/100 * 0.45 * 12 * 0.3)
            months    = np.arange(1, horizon*12+1)
            r_b = base_rate/12
            r_o = opt_rate/12
            fv_b = monthly_usd * ((1+r_b)**months - 1) / r_b
            fv_o = monthly_usd * ((1+r_o)**months - 1) / r_o

            # Bug 4 fix: Tier C suppresses the optimistic rate in the chart
            # title to match the table (which already hides the Optimistic row).
            conservative_rate = proj_scenarios.get("Conservative", {}).get("opt_rate", base_rate*100) if proj_scenarios else base_rate*100
            if tier == "C":
                chart_rate_label = f"{conservative_rate:.1f}% effective rate (conservative)"
            else:
                chart_rate_label = f"{opt_rate*100:.1f}% effective rate"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months/12, y=fv_b,
                name=f"Fixed day 27 ({base_rate*100:.0f}%)",
                line=dict(color="#5C6378", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=months/12, y=fv_o,
                name=f"AI-optimised ({opt_rate*100:.1f}%)",
                line=dict(color="#2DB37A", width=2.5),
                fill="tonexty", fillcolor="rgba(45,179,122,0.08)",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"{ticker_input} — {horizon}yr compounding ({chart_rate_label})", font=dict(size=12)),
                xaxis=dict(**AXIS_DEFAULTS, title="Years"),
                yaxis=dict(**AXIS_DEFAULTS, tickprefix="$", tickformat=",.0f"),
                height=300,
            )
            st.plotly_chart(fig)

        with col_right:
            # Regime breakdown
            if regime:
                r_names  = list(regime.keys())
                r_wins   = [regime[r]["win_rate"] for r in r_names]
                r_saves  = [regime[r]["avg_saving"] for r in r_names]
                r_colors = {"fear":"#E05C5C","elevated":"#E8A020","normal":"#9AA0B4","calm":"#2DB37A"}
                bar_colors = [r_colors.get(r, "#5C6378") for r in r_names]

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Bar(
                    x=r_names, y=r_wins, name="Win %",
                    marker_color=bar_colors, opacity=0.8,
                ), secondary_y=False)
                fig2.add_trace(go.Scatter(
                    x=r_names, y=r_saves, name="Avg save %",
                    line=dict(color="#4D9EF5", width=2), mode="lines+markers",
                    marker=dict(size=7),
                ), secondary_y=True)
                fig2.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text="Performance by VIX regime", font=dict(size=12)),
                    height=300, showlegend=True,
                )
                fig2.update_yaxes(title_text="Win rate %", secondary_y=False,
                                  gridcolor="#252A38", linecolor="#252A38", showgrid=True)
                fig2.update_yaxes(title_text="Avg saving %", secondary_y=True,
                                  gridcolor="rgba(0,0,0,0)", showgrid=False)
                st.plotly_chart(fig2)

        # ── LLM Advisory — async, shows after MC results ─────────────
        advisory_text = result.get("advisory", "")
        key_factor    = result.get("key_factor", "")
        llm_action    = result.get("llm_action", "consider")

        adv_placeholder = st.empty()

        if advisory_text:
            # Advisory already in cache — show immediately
            _render_advisory(adv_placeholder, advisory_text, key_factor, llm_action)
        else:
            # No advisory yet — fetch async so MC results show first
            _llm_key = f"llm_{ticker_input}"

            if _llm_key not in st.session_state:
                st.session_state[_llm_key] = {"status": "running", "result": None}

                def _fetch_llm():
                    try:
                        from src.llm.groq_advisor import (
                            build_macro_context, build_ticker_context, generate_advisory
                        )
                        macro_ctx  = build_macro_context()
                        ticker_ctx = build_ticker_context(ticker_input)
                        adv = generate_advisory(
                            ticker_input,
                            result.get("lgbm_recommendation", {}),
                            macro_ctx,
                            ticker_ctx,
                        )
                        st.session_state[_llm_key] = {"status": "done", "result": adv}
                    except Exception as e:
                        st.session_state[_llm_key] = {
                            "status": "error",
                            "result": {"advisory": f"LLM unavailable ({e})",
                                       "key_factor": "model signal only",
                                       "action": "consider"}
                        }

                t = threading.Thread(target=_fetch_llm, daemon=True)
                t.start()

            llm_state = st.session_state.get(_llm_key, {})
            if llm_state.get("status") == "done" and llm_state.get("result"):
                adv = llm_state["result"]
                _render_advisory(adv_placeholder, adv.get("advisory",""),
                                 adv.get("key_factor",""), adv.get("action","consider"))
            else:
                adv_placeholder.markdown("""
                <div style="font-family:var(--mono);font-size:0.72rem;color:#5C6378;
                            padding:0.5rem 0;border-top:1px solid #252A38;margin-top:0.5rem;">
                    ⟳ Fetching LLM advisory context…
                </div>
                """, unsafe_allow_html=True)

        # ── Projection table — tier-aware scenarios ───────────────────
        if proj_scenarios:
            scenario_names = list(proj_scenarios.keys())
            if not show_optimistic and "Optimistic" in scenario_names:
                scenario_names = [s for s in scenario_names if s != "Optimistic"]

            label_str = " · ".join([
                f"{s} ({proj_scenarios[s].get('base_rate',0):.0f}% base, "
                f"{proj_scenarios[s].get('capture_rate',0):.0f}% capture)"
                for s in scenario_names
            ])
            st.markdown(f'<div class="section-label">compounding projections — {label_str}</div>',
                        unsafe_allow_html=True)

            # Collect all available years from the first scenario
            first_s = proj_scenarios[scenario_names[0]].get("projections", {})
            available_years = sorted([int(k) for k in first_s.keys()])
            if not available_years:
                available_years = [1, 3, 5, 10, 20]

            rows = []
            for yr in available_years:
                row = {"Years": f"{yr}yr"}
                for s_name in scenario_names:
                    s_data = proj_scenarios[s_name].get("projections", {})
                    yr_key = yr if yr in s_data else str(yr)
                    if yr_key in s_data:
                        extra = s_data[yr_key].get("extra", 0)
                        row[f"{s_name} extra"] = f"+${extra:,.0f}"
                        row[f"{s_name} lift"]  = f"+{s_data[yr_key].get('lift_pct',0):.1f}%"
                rows.append(row)

            df_proj = pd.DataFrame(rows)
            st.dataframe(df_proj, hide_index=True)

        elif proj:
            # Fallback for portfolio tickers using old format
            st.markdown('<div class="section-label">compounding projections — base case</div>',
                        unsafe_allow_html=True)
            rows = []
            for yr, p_data in sorted(proj.items()):
                extra = p_data.get("extra", 0)
                rows.append({
                    "Years": f"{yr}yr",
                    "Fixed day 27": f"${p_data.get('fixed',0):,.0f}",
                    "AI-optimised": f"${p_data.get('optimised',0):,.0f}",
                    "Extra wealth":  f"+${extra:,.0f}",
                    "Lift": f"+{extra/max(p_data.get('fixed',1),1)*100:.1f}%",
                })
            df_proj = pd.DataFrame(rows)
            st.dataframe(df_proj, hide_index=True)


# ── Page 3: Historical Evidence ───────────────────────────────────────────────