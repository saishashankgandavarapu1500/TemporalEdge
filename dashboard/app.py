"""
TemporalEdge — Streamlit Dashboard entry point.
Phases 5-7: Execution Plan / Simulator / Evidence + multi-user sidebar.

All rendering logic lives in dashboard/components/.
This file handles: page config, CSS injection, sidebar, header, tab routing.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TemporalEdge",
    page_icon="⏱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system — load CSS from file ───────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "styles.css"
st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Components ────────────────────────────────────────────────────────────────
from components.utils import (
    get_active_portfolio, _format_confidence,
    load_execution_plan, _DEFAULT_PORTFOLIO, MODEL_PROXY,
)
from components.execution_plan import render_execution_plan
from components.simulator     import render_simulator
from components.evidence      import render_evidence

def render_header(active_tab: str):
    now      = datetime.now().strftime("%d %b %Y  %H:%M")
    active   = get_active_portfolio()
    total_mo = sum(v["monthly_usd"] for v in active.values())
    n_tickers = len(active)
    st.markdown(f"""
    <div class="te-header">
        <div>
            <div class="te-logo">TEMPORAL<span>EDGE</span></div>
            <div class="te-meta">portfolio-aware DCA timing system</div>
        </div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:#5C6378;
                    text-align:right;line-height:2;">
            {now} UTC<br>
            <span style="color:#9AA0B4">{n_tickers} tickers · </span>
            <span style="color:#2DB37A">${total_mo}/mo</span><br>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page 1: Execution Plan ────────────────────────────────────────────────────


def render_sidebar():
    """Sidebar portfolio editor — stateful, rerun-safe."""
    with st.sidebar:
        # ── Sidebar header ─────────────────────────────────────────────────
        st.markdown("""
        <div style="font-family:var(--mono);font-size:0.88rem;font-weight:600;
                    color:#E8EAF0;letter-spacing:0.06em;padding-top:0.3rem;">
            TRY UR OWN TICKER</div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:#9AA0B4;
                    margin-top:0.4rem;line-height:1.7;">
            Enter your tickers and monthly investment amounts.<br>
            TemporalEdge will find the optimal buy day for each.
        </div>
        <div style="font-family:var(--mono);font-size:0.62rem;color:#5C6378;
                    margin-top:0.35rem;margin-bottom:0.9rem;">
            🔒 Session only · no data stored · resets on tab close
        </div>
        """, unsafe_allow_html=True)

        # ── Initialise state ───────────────────────────────────────────────
        if "sidebar_rows" not in st.session_state:
            st.session_state["sidebar_rows"] = [
                {"ticker": t, "amount": v["monthly_usd"]}
                for t, v in _DEFAULT_PORTFOLIO.items()
            ]
        if "user_portfolio" not in st.session_state:
            st.session_state["user_portfolio"] = dict(_DEFAULT_PORTFOLIO)

        rows  = st.session_state["sidebar_rows"]
        n_rows = len(rows)

        # Column header
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.65rem;color:#5C6378;'
            'letter-spacing:0.08em;margin-bottom:0.3rem;">TICKER &nbsp;&nbsp;&nbsp; $/MO</div>',
            unsafe_allow_html=True
        )

        # ── Rows ───────────────────────────────────────────────────────────
        delete_idx = None
        for i in range(n_rows):
            c1, c2, c3 = st.columns([2, 1.4, 0.6])
            with c1:
                st.text_input(f"t_{i}", value=rows[i]["ticker"],
                    label_visibility="collapsed", key=f"ticker_input_{i}",
                    placeholder="e.g. VOO", max_chars=10)
            with c2:
                st.number_input(f"a_{i}", value=float(rows[i]["amount"]),
                    min_value=1.0, max_value=50000.0, step=1.0,
                    label_visibility="collapsed", key=f"amount_input_{i}",
                    format="%.0f")
            with c3:
                if st.button("✕", key=f"del_{i}", help="Remove"):
                    delete_idx = i

        if delete_idx is not None:
            for i in range(n_rows):
                rows[i] = {
                    "ticker": str(st.session_state.get(f"ticker_input_{i}", rows[i]["ticker"])).upper().strip(),
                    "amount": int(float(st.session_state.get(f"amount_input_{i}", rows[i]["amount"])))
                }
            rows.pop(delete_idx)
            for k in [f"ticker_input_{delete_idx}", f"amount_input_{delete_idx}", f"del_{delete_idx}"]:
                st.session_state.pop(k, None)
            st.session_state["sidebar_rows"] = rows
            st.rerun()

        # ── Add ticker ─────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:0.35rem'></div>", unsafe_allow_html=True)
        if n_rows < 15:
            if st.button("＋  Add ticker", key="add_ticker_btn", use_container_width=True):
                for i in range(n_rows):
                    rows[i] = {
                        "ticker": str(st.session_state.get(f"ticker_input_{i}", rows[i]["ticker"])).upper().strip(),
                        "amount": int(float(st.session_state.get(f"amount_input_{i}", rows[i]["amount"])))
                    }
                # Only add if the last row is not already empty (prevents duplicate blank rows)
                if not rows or rows[-1]["ticker"] != "":
                    rows.append({"ticker": "", "amount": 10})
                    st.session_state["sidebar_rows"] = rows
                    st.rerun()
        else:
            st.caption("Max 15 tickers")

        st.markdown("<hr style='border-color:#252A38;margin:0.6rem 0'>", unsafe_allow_html=True)

        # ── Pending changes indicator ──────────────────────────────────────
        applied = st.session_state.get("user_portfolio", {})
        cur_tickers = [str(st.session_state.get(f"ticker_input_{i}", rows[i]["ticker"])).upper().strip() for i in range(n_rows)]
        pending = set(t for t in cur_tickers if t) != set(applied.keys())
        if not pending:
            for i in range(n_rows):
                t = cur_tickers[i]
                a = int(float(st.session_state.get(f"amount_input_{i}", rows[i]["amount"])))
                if t and applied.get(t, {}).get("monthly_usd") != a:
                    pending = True; break
        if pending:
            st.markdown('<div style="font-family:var(--mono);font-size:0.65rem;'
                'color:#E8A020;text-align:center;margin-bottom:0.3rem;">⚠ unsaved — click Apply</div>',
                unsafe_allow_html=True)

        # ── Apply / Reset ──────────────────────────────────────────────────
        ca, cb = st.columns(2)
        with ca:
            if st.button("✓  Apply", key="apply_btn", use_container_width=True, type="primary"):
                new_portfolio = {}
                for i in range(n_rows):
                    t = str(st.session_state.get(f"ticker_input_{i}", rows[i]["ticker"])).upper().strip()
                    a = int(float(st.session_state.get(f"amount_input_{i}", rows[i]["amount"])))
                    if t:
                        rows[i] = {"ticker": t, "amount": a}
                        new_portfolio[t] = {"name": t, "monthly_usd": a}
                if new_portfolio:
                    st.session_state["user_portfolio"] = new_portfolio
                    st.session_state["sidebar_rows"]   = rows
                    st.cache_data.clear()
                    st.toast(f"✓ Saved — {len(new_portfolio)} tickers", icon="✅")
                    st.rerun()
                else:
                    st.warning("Add at least one ticker")
        with cb:
            if st.button("↺  Reset", key="reset_btn", use_container_width=True):
                for i in range(n_rows):
                    st.session_state.pop(f"ticker_input_{i}", None)
                    st.session_state.pop(f"amount_input_{i}", None)
                st.session_state["user_portfolio"] = dict(_DEFAULT_PORTFOLIO)
                st.session_state["sidebar_rows"]   = [
                    {"ticker": t, "amount": v["monthly_usd"]}
                    for t, v in _DEFAULT_PORTFOLIO.items()
                ]
                st.rerun()

        # ── Summary ────────────────────────────────────────────────────────
        active   = get_active_portfolio()
        total_mo = sum(v["monthly_usd"] for v in active.values())
        st.markdown(f"""
        <div style="background:#141720;border:1px solid #252A38;border-radius:6px;
                    padding:0.5rem 0.75rem;margin-top:0.6rem;">
            <div style="font-family:var(--mono);font-size:0.6rem;color:#5C6378;
                        letter-spacing:0.08em;margin-bottom:0.2rem;">ACTIVE PORTFOLIO</div>
            <div style="font-family:var(--mono);font-size:0.82rem;color:#E8EAF0;">
                {len(active)} tickers · <span style="color:#2DB37A">${total_mo}/mo</span>
            </div>
        </div>
        <div style="font-family:var(--mono);font-size:0.6rem;color:#3C4258;
                    margin-top:0.6rem;line-height:1.6;">
            🔒 Session only · resets on tab close
        </div>
        """, unsafe_allow_html=True)


def main():
    render_sidebar()
    render_header("plan")

    tab1, tab2, tab3 = st.tabs([
        "⬡  EXECUTION PLAN",
        "◎  SIMULATOR",
        "▦  EVIDENCE",
    ])

    with tab1:
        render_execution_plan()

    with tab2:
        render_simulator()

    with tab3:
        render_evidence()


main()
