"""
TemporalEdge — Execution Plan tab renderer.
"""
import sys
from pathlib import Path
import streamlit as st
from .utils import (
    get_active_portfolio, _format_confidence,
    load_execution_plan, MODEL_PROXY,
    TIER_COLORS, TIER_LABELS, PLOTLY_LAYOUT, AXIS_DEFAULTS,
)

def render_execution_plan():
    plan_data = load_execution_plan()

    if plan_data is None:
        st.markdown("""
        <div style="text-align:center; padding:4rem; color:#5C6378;">
            <div style="font-family:var(--mono); font-size:0.85rem; margin-bottom:1rem;">
                NO EXECUTION PLAN CACHED
            </div>
            <div style="font-size:0.78rem;">
                Run <code>python scripts/monthly_refresh.py</code> to generate this month's plan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⟳  Generate Plan Now"):
            with st.spinner("Running models and fetching context..."):
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from scripts.monthly_refresh import run_monthly_refresh
                    run_monthly_refresh()
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
        return

    # ── Macro bar ──────────────────────────────────────────────────────────
    macro = plan_data.get("macro_context", {})
    vix   = macro.get("vix", 0)
    vix_color = (
        "#E05C5C" if vix >= 30 else
        "#E8A020" if vix >= 20 else
        "#9AA0B4" if vix >= 15 else "#2DB37A"
    )

    summary = plan_data.get("summary", {})
    # Stat bar placeholder — filled below with live counts after ep merge
    _stat_ph = st.empty()

    # ── Tier legend ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="tier-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#2DB37A"></div>ACT ON MODEL — high confidence, stable signal</div>
      <div class="legend-item"><div class="legend-dot" style="background:#E8A020"></div>USE LOOSELY — review context before acting</div>
      <div class="legend-item"><div class="legend-dot" style="background:#6B7280"></div>SKIP TIMING — keep day 27 schedule</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Ticker cards ───────────────────────────────────────────────────────
    ep = plan_data.get("execution_plan", {})
    active_portfolio = get_active_portfolio()

    # Show only tickers in the user's active portfolio.
    # Unknown tickers (not in cache) get a "queued" card.
    # Merge on-demand + precomputed cache into ep for tickers not in monthly plan
    from .utils import load_ondemand_result

    cached_tickers = set(ep.keys())
    user_tickers   = set(active_portfolio.keys())

    for t in user_tickers - cached_tickers:
        try:
            od = load_ondemand_result(t)
            if od and "summary" in od:
                s    = od["summary"]
                tier = s.get("tier", "B")
                ep[t] = {
                    "ticker":           t,
                    "name":             s.get("ticker_meta", {}).get("name", t),
                    "monthly_usd":      active_portfolio[t]["monthly_usd"],
                    "recommended_day":  od.get("recommended_day", s.get("lgbm_rec", {}).get("recommended_day", 27)),
                    "top_3_days":       od.get("top_3_days", s.get("lgbm_rec", {}).get("top_3_days", [27])),
                    "confidence":       od.get("confidence", s.get("lgbm_rec", {}).get("confidence", 0.5)),
                    "predicted_saving": s.get("avg_saving_pct", 0.0),
                    "suggested_window": od.get("suggested_window", ""),
                    "advisory":         od.get("advisory", s.get("llm_advisory", {}).get("advisory", "")),
                    "key_factor":       od.get("key_factor", s.get("llm_advisory", {}).get("key_factor", "")),
                    "llm_action":       od.get("llm_action", s.get("llm_advisory", {}).get("action", "consider")),
                    "exec_tier":        {"A":"green","B":"amber","C":"grey"}.get(tier,"amber"),
                    "exec_score":       {"A":7,"B":5,"C":3}.get(tier,5),
                    "exec_reason":      s.get("trust_summary",""),
                    "win_rate_pct":     s.get("win_rate_pct", 50),
                }
                cached_tickers.add(t)
        except Exception:
            pass

    # Patch monthly_usd from live portfolio (cache may have stale amounts)
    for t in cached_tickers & user_tickers:
        ep[t]["monthly_usd"] = active_portfolio[t]["monthly_usd"]

    known_tickers   = user_tickers & cached_tickers
    unknown_tickers = user_tickers - cached_tickers

    # Update g/a/gr with live merged counts
    g  = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "green")
    a  = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "amber")
    gr = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "grey")

    # Fill stat bar now with live counts
    g  = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "green")
    a  = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "amber")
    gr = sum(1 for t in known_tickers if ep[t].get("exec_tier") == "grey")
    _stat_ph.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-val" style="color:{vix_color}">{vix:.1f}</div>
        <div class="stat-lbl">VIX · {macro.get('vix_regime','')}</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('sp500_5d_chg',0):+.1f}%</div>
        <div class="stat-lbl">S&P 500 · 5-day return</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('yield_curve','—')}</div>
        <div class="stat-lbl">Yield curve · 10Y–3M</div>
      </div>
      <div class="stat-box">
        <div class="stat-val stat-white">{macro.get('usd_index_5d_chg',0):+.1f}%</div>
        <div class="stat-lbl">USD · 5-day change</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">
          <span style="color:#2DB37A">{g}</span>
          <span style="color:#5C6378; font-size:1rem"> / </span>
          <span style="color:#E8A020">{a}</span>
          <span style="color:#5C6378; font-size:1rem"> / </span>
          <span style="color:#6B7280">{gr}</span>
        </div>
        <div class="stat-lbl">Act · Consider · Skip</div>
      </div>
    </div>
    <div class="te-meta" style="margin-bottom:1.5rem;">
      Plan generated: {plan_data.get('generated_at','')[:16]}
    </div>
    """, unsafe_allow_html=True)

    # Sort: green → amber → grey, then by score desc
    tier_order = {"green": 0, "amber": 1, "grey": 2}
    sorted_tickers = sorted(
        known_tickers,
        key=lambda t: (tier_order.get(ep[t].get("exec_tier","grey"), 2),
                       -ep[t].get("exec_score", 0))
    )

    # Show warning cards for unknown tickers
    if unknown_tickers:
        unknown_list = ", ".join(sorted(unknown_tickers))
        st.markdown(f"""
        <div style="background:rgba(232,160,32,0.07);border:1px solid rgba(232,160,32,0.2);
                    border-radius:6px;padding:0.75rem 1rem;margin-bottom:1rem;
                    font-family:var(--mono);font-size:0.75rem;color:#E8A020;">
            ⏳ <strong>NOT YET ANALYSED:</strong> {unknown_list}<br>
            <span style="color:#9AA0B4;font-size:0.7rem;">
            Go to the Simulator tab → run each ticker once → come back here.
            Results are cached for future visits.
            </span>
        </div>
        """, unsafe_allow_html=True)

    for ticker in sorted_tickers:
        card = ep[ticker]
        tier        = card.get("exec_tier", "grey")
        conf        = card.get("confidence", 0.5)
        rec_day     = card.get("recommended_day", 27)
        top3        = card.get("top_3_days", [rec_day])
        advisory    = card.get("advisory", "")
        window_str  = card.get("suggested_window", "")
        key_factor  = card.get("key_factor", "")
        pred_save   = card.get("predicted_saving", 0.0)
        monthly     = card.get("monthly_usd", 0)
        score       = card.get("exec_score", 0)
        win_rate_pct = card.get("win_rate_pct", None)  # Phase 4 MC win rate

        # Bug 3 fix: use win rate as tiebreaker within each integer band so
        # green tickers don't all score exactly 7/10.
        # Band floors: A→7.0, B→5.0, C→3.0.  Win rate adds 0.0–0.9 within band.
        _TIER_BASE  = {"green": 7.0, "amber": 5.0, "grey": 3.0}
        _TIER_RANGE = {"green": (0.60, 0.85), "amber": (0.50, 0.75), "grey": (0.40, 0.70)}
        if win_rate_pct is not None and tier in _TIER_BASE:
            wr   = win_rate_pct / 100
            lo, hi = _TIER_RANGE[tier]
            clamped = max(lo, min(hi, wr))
            tiebreaker = 0.9 * (clamped - lo) / (hi - lo)
            score = round(_TIER_BASE[tier] + tiebreaker, 1)

        badge_cls = f"badge-{tier}"
        card_cls  = tier
        color     = TIER_COLORS[tier]
        label     = TIER_LABELS[tier]
        conf_pct  = int(conf * 100)
        conf_cls  = f"conf-{tier}"
        conf_display = _format_confidence(ticker, conf_pct)

        top3_str = "  ".join([
            f"<span style='color:{color};font-weight:600'>{d}</span>" if d == rec_day
            else f"<span style='color:#5C6378'>{d}</span>"
            for d in top3
        ])

        st.markdown(f"""
        <div class="ticker-card {card_cls}">
          <div class="card-header">
            <div>
              <div class="card-ticker">{ticker}
                <span style="font-weight:300;font-size:0.85rem;color:#5C6378;margin-left:0.5rem">${monthly}/mo</span>
              </div>
              <div class="card-name">{card.get('name','')}</div>
            </div>
            <div style="display:flex;gap:0.5rem;align-items:center">
              <span style="font-family:var(--mono);font-size:0.72rem;color:#5C6378">score {score}/10</span>
              <span class="card-badge {badge_cls}">{label}</span>
            </div>
          </div>

          <div class="card-rec">
            <div>
              <div class="rec-label">RECOMMENDED DAY</div>
              <div class="rec-day" style="color:{color}">{rec_day}</div>
            </div>
            <div>
              <div class="rec-label">TOP 3 DAYS</div>
              <div style="font-family:var(--mono);font-size:0.95rem;margin-top:4px">{top3_str}</div>
            </div>
            <div>
              <div class="rec-label">CONFIDENCE</div>
              <div class="rec-conf" style="margin-top:4px">{conf_display}</div>
              <div class="conf-bar-wrap" style="width:80px">
                <div class="conf-bar {conf_cls}" style="width:{conf_pct}%"></div>
              </div>
            </div>
            <div>
              <div class="rec-label">PREDICTED SAVING</div>
              <div style="font-family:var(--mono);font-size:0.88rem;color:{color};margin-top:4px">{pred_save:+.2f}%</div>
            </div>
            {"<div><div class='rec-label'>WINDOW</div><div style='font-family:var(--mono);font-size:0.82rem;color:#9AA0B4;margin-top:4px'>" + window_str + "</div></div>" if window_str else ""}
          </div>

          {"<div class='card-advisory'>" + advisory + ("<div class='key-factor'>↳ " + key_factor + "</div>" if key_factor else "") + "</div>" if advisory else ""}
        </div>
        """, unsafe_allow_html=True)

    # Refresh button
    st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
    if st.button("⟳  Refresh Plan"):
        with st.spinner("Refreshing..."):
            try:
                from scripts.monthly_refresh import run_monthly_refresh
                run_monthly_refresh()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Page 2: Portfolio Simulator ───────────────────────────────────────────────
