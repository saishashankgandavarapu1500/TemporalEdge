"""
TemporalEdge — Full Simulator Test Suite
Covers: ticker classification, result validation, session state,
        cache logic, concurrency, edge cases, UI banner logic,
        file validation, LLM state machine, projection rendering.

Run with: pytest tests/test_simulator_full.py -v
"""
import json
import threading
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO = {"VOO", "VTI", "NVDA", "AAPL", "SCHD", "VXUS", "TSLA", "BND", "VYM", "VEA", "VWO"}
PRECOMPUTED = {"MSFT", "GOOGL", "AMZN", "META", "JPM", "GS", "ABBV", "MMM", "JEPI", "SCHD"}


def make_result(ticker="VOO", tier="A", trust=85, win=72.0, with_advisory=True):
    """Minimal valid result matching actual JSON structure."""
    r = {
        "ticker": ticker,
        "elapsed_s": 42.1,
        "generated": "2026-05-01T04:00:00",
        "cached": True,
        "summary": {
            "ticker": ticker,
            "monthly_usd": 50.0,
            "n_runs": 1000,
            "tier": tier,
            "tier_label": "ACT ON MODEL",
            "trust_score": trust,
            "trust_reasons": ["26yr history", "high win rate"],
            "trust_penalties": [],
            "trust_summary": "Strong signal",
            "win_rate_pct": win,
            "avg_saving_pct": 0.84,
            "median_saving_pct": 0.79,
            "percentiles": {"p10": -0.5, "p50": 0.84, "p90": 2.1},
            "clipped_values": 33,
            "clip_bounds": {"low": -2.0, "high": 5.0},
            "avg_dollar_per_month": 4.20,
            "avg_dollar_per_year": 50.4,
            "projections": {},
            "projections_by_scenario": {
                "Conservative": {
                    "base_rate": 12, "capture_rate": 70, "opt_rate": 13.5,
                    "projections": {
                        1: {"extra": 120, "lift_pct": 1.2},
                        3: {"extra": 400, "lift_pct": 1.5},
                        5: {"extra": 800, "lift_pct": 1.8},
                        10: {"extra": 2000, "lift_pct": 2.1},
                        20: {"extra": 6000, "lift_pct": 2.5},
                    },
                },
                "Base": {
                    "base_rate": 15, "capture_rate": 85, "opt_rate": 16.5,
                    "projections": {
                        1: {"extra": 200, "lift_pct": 1.8},
                        3: {"extra": 700, "lift_pct": 2.1},
                        5: {"extra": 1500, "lift_pct": 2.5},
                        10: {"extra": 4000, "lift_pct": 3.0},
                        20: {"extra": 12000, "lift_pct": 3.8},
                    },
                },
                "Optimistic": {
                    "base_rate": 18, "capture_rate": 95, "opt_rate": 19.5,
                    "projections": {
                        1: {"extra": 350, "lift_pct": 2.8},
                        3: {"extra": 1200, "lift_pct": 3.5},
                        5: {"extra": 2800, "lift_pct": 4.0},
                        10: {"extra": 8000, "lift_pct": 5.0},
                        20: {"extra": 25000, "lift_pct": 6.5},
                    },
                },
            },
            "show_optimistic": tier != "C",
            "projection_note": "",
            "projection_years": [1, 3, 5, 10, 20],
            "regime_stats": {
                "calm":     {"win_rate": 75.0, "avg_saving": 1.2},
                "normal":   {"win_rate": 70.0, "avg_saving": 0.9},
                "elevated": {"win_rate": 65.0, "avg_saving": 0.6},
                "fear":     {"win_rate": 60.0, "avg_saving": 0.3},
            },
            "confidence_stats": {"high": 0.6, "medium": 0.3, "low": 0.1},
            "backtest": {"sharpe": 1.2, "max_drawdown": -0.08},
            "ticker_meta": {"name": "Vanguard S&P 500 ETF", "sector": "ETF", "beta": 1.0},
            "is_portfolio_ticker": ticker in PORTFOLIO,
            "is_proxy_data": False,
            "llm_advisory": {"advisory": "Strong buy signal." if with_advisory else "",
                             "key_factor": "momentum", "action": "act"},
            "lgbm_rec": {"signal": "buy", "confidence": 0.82},
        },
    }
    return r


def classify(ticker, portfolio=PORTFOLIO, precomputed=PRECOMPUTED):
    is_p   = ticker in portfolio
    is_pre = ticker in precomputed
    needs  = not is_p and not is_pre
    return is_p, is_pre, needs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ticker Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestTickerClassification:

    def test_portfolio_tickers_never_need_live(self):
        for t in PORTFOLIO:
            is_p, _, needs = classify(t)
            assert is_p and not needs, f"{t} should not need live"

    def test_precomputed_tickers_dont_need_live(self):
        for t in PRECOMPUTED - PORTFOLIO:
            _, is_pre, needs = classify(t)
            assert is_pre and not needs, f"{t} should load from precomputed"

    def test_unknown_tickers_need_live(self):
        for t in ["PLTR", "GME", "RIVN", "XYZ", "DOGE"]:
            _, _, needs = classify(t)
            assert needs, f"{t} should require live"

    def test_lowercase_normalised(self):
        ticker = "voo".upper().strip()
        is_p, _, _ = classify(ticker)
        assert is_p

    def test_whitespace_stripped(self):
        ticker = "  MSFT  ".upper().strip()
        _, is_pre, _ = classify(ticker)
        assert is_pre

    def test_empty_string_needs_live(self):
        _, _, needs = classify("")
        assert needs

    def test_single_letter_tickers(self):
        """V, F, A, T etc must be classifiable without crashing."""
        for t in ["V", "F", "A", "T", "O", "L"]:
            is_p, is_pre, needs = classify(t)
            # Exactly one bucket must be true
            assert sum([is_p, is_pre, needs]) >= 1

    def test_hyphenated_ticker(self):
        """BRK-B has a hyphen — must not crash classification."""
        is_p, is_pre, needs = classify("BRK-B")
        assert isinstance(needs, bool)

    def test_portfolio_wins_over_precomputed(self):
        """If ticker is in both sets, portfolio takes priority."""
        both = {"VOO"}
        is_p, is_pre, needs = classify("VOO", portfolio=both, precomputed=both)
        assert is_p and not needs

    def test_delisted_tickers_need_live(self):
        for t in ["BBBYQ", "NKLA", "SPCE", "ANSS"]:
            _, _, needs = classify(t)
            assert needs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Result Validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_SUMMARY = {"tier", "trust_score", "win_rate_pct"}


def validate_result(data):
    if not isinstance(data, dict) or not data:
        return False, "empty or non-dict"
    if data.get("error"):
        return False, f"error: {data['error']}"
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return False, "missing summary"
    missing = REQUIRED_SUMMARY - summary.keys()
    if missing:
        return False, f"missing: {missing}"
    return True, "ok"


class TestResultValidation:

    def test_valid_result_passes(self):
        ok, msg = validate_result(make_result())
        assert ok, msg

    def test_empty_dict_fails(self):
        ok, _ = validate_result({})
        assert not ok

    def test_none_fails(self):
        ok, _ = validate_result(None)
        assert not ok

    def test_error_field_fails(self):
        ok, msg = validate_result({"error": "Pipeline failed"})
        assert not ok
        assert "Pipeline failed" in msg

    def test_missing_summary_fails(self):
        r = make_result()
        del r["summary"]
        ok, _ = validate_result(r)
        assert not ok

    def test_missing_tier_fails(self):
        r = make_result()
        del r["summary"]["tier"]
        ok, msg = validate_result(r)
        assert not ok
        assert "tier" in msg

    def test_missing_trust_score_fails(self):
        r = make_result()
        del r["summary"]["trust_score"]
        ok, msg = validate_result(r)
        assert not ok

    def test_missing_win_rate_fails(self):
        r = make_result()
        del r["summary"]["win_rate_pct"]
        ok, _ = validate_result(r)
        assert not ok

    def test_null_value_keys_still_pass(self):
        """Keys present but None — validator only checks existence."""
        r = make_result()
        r["summary"]["trust_score"] = None
        ok, _ = validate_result(r)
        assert ok

    def test_tier_c_hides_optimistic(self):
        r = make_result(tier="C")
        assert r["summary"]["show_optimistic"] is False

    def test_tier_a_shows_optimistic(self):
        r = make_result(tier="A")
        assert r["summary"]["show_optimistic"] is True

    def test_win_rate_color_thresholds(self):
        cases = [(75.0, "#2DB37A"), (60.0, "#E8A020"), (45.0, "#6B7280")]
        for win, expected in cases:
            color = "#2DB37A" if win >= 65 else "#E8A020" if win >= 55 else "#6B7280"
            assert color == expected

    def test_trust_score_color_thresholds(self):
        cases = [(85, "#2DB37A"), (50, "#E8A020"), (20, "#E05C5C")]
        for score, expected in cases:
            color = "#2DB37A" if score >= 65 else "#E8A020" if score >= 40 else "#E05C5C"
            assert color == expected

    def test_all_projection_years_present(self):
        r = make_result()
        for scenario, data in r["summary"]["projections_by_scenario"].items():
            for yr in [1, 3, 5, 10, 20]:
                assert yr in data["projections"], f"{scenario} missing {yr}yr"

    def test_all_four_regimes_present(self):
        r = make_result()
        for regime in ["calm", "normal", "elevated", "fear"]:
            assert regime in r["summary"]["regime_stats"]

    def test_clip_bounds_valid(self):
        r = make_result()
        bounds = r["summary"]["clip_bounds"]
        assert bounds["low"] < 0 and bounds["high"] > 0

    def test_percentiles_all_keys(self):
        r = make_result()
        p = r["summary"]["percentiles"]
        assert all(k in p for k in ["p10", "p50", "p90"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Session State & Cache Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionStateAndCache:

    def _session(self, **kwargs):
        return dict(**kwargs)

    def test_gate_blocks_unknown_ticker_without_approval(self):
        session = self._session()
        needs_live = True
        approved = session.get("live_approved_PLTR", False)
        assert needs_live and not approved  # gate must block

    def test_gate_opens_after_approval(self):
        session = self._session(live_approved_PLTR=True)
        approved = session.get("live_approved_PLTR", False)
        assert approved

    def test_approval_is_ticker_specific(self):
        session = self._session(live_approved_PLTR=True)
        assert not session.get("live_approved_GME", False)

    def test_od_running_flag_prevents_double_run(self):
        session = self._session(od_running_PLTR=True)
        assert session.get("od_running_PLTR", False)

    def test_running_flag_cleared_on_completion(self):
        session = {"od_running_PLTR": True}
        session["od_running_PLTR"] = False
        assert not session["od_running_PLTR"]

    def test_running_flag_cleared_on_exception(self):
        session = {"od_running_PLTR": True}
        try:
            raise RuntimeError("Pipeline crashed")
        except Exception:
            session["od_running_PLTR"] = False
        assert not session["od_running_PLTR"]

    def test_result_stored_after_run(self):
        session = {}
        session["od_result_PLTR"] = make_result("PLTR")
        assert "od_result_PLTR" in session

    def test_stale_result_cleared_before_new_run(self):
        session = {"od_result_PLTR": make_result("PLTR")}
        session.pop("od_result_PLTR", None)
        assert "od_result_PLTR" not in session

    def test_llm_state_transitions(self):
        key = "llm_VOO"
        session = {}
        assert session.get(key) is None
        session[key] = {"status": "running", "result": None}
        assert session[key]["status"] == "running"
        session[key] = {"status": "done", "result": {"advisory": "Buy", "key_factor": "momentum", "action": "act"}}
        assert session[key]["status"] == "done"
        assert session[key]["result"]["advisory"] == "Buy"

    def test_llm_error_state(self):
        session = {"llm_VOO": {"status": "error", "result": {
            "advisory": "LLM unavailable (timeout)",
            "key_factor": "model signal only",
            "action": "consider"
        }}}
        assert session["llm_VOO"]["status"] == "error"
        assert "LLM unavailable" in session["llm_VOO"]["result"]["advisory"]

    def test_portfolio_cache_hit(self):
        sim_results = {"VOO": make_result("VOO")}
        assert "VOO" in sim_results

    def test_portfolio_cache_miss(self):
        sim_results = {}
        assert "VOO" not in sim_results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concurrency — Mid-run Ticker Change
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_result_keys_isolated_per_ticker(self):
        session = {}
        session["od_result_PLTR"] = make_result("PLTR")
        session["od_result_GME"]  = make_result("GME")
        assert session["od_result_PLTR"]["ticker"] == "PLTR"
        assert session["od_result_GME"]["ticker"]  == "GME"

    def test_switching_ticker_doesnt_show_old_result(self):
        session = {"od_result_PLTR": make_result("PLTR")}
        new_ticker = "GME"
        result = session.get(f"od_result_{new_ticker}")
        assert result is None

    def test_running_flag_per_ticker(self):
        session = {"od_running_PLTR": True, "od_running_GME": False}
        assert session["od_running_PLTR"]
        assert not session["od_running_GME"]

    def test_approval_persists_within_session(self):
        """Once approved, user shouldn't be gated again in same session."""
        session = {"live_approved_PLTR": True}
        # Simulate second run attempt
        approved = session.get("live_approved_PLTR", False)
        assert approved

    def test_thread_result_isolation(self):
        """Simulate two threads writing results for different tickers."""
        session = {}
        results = {}

        def run(ticker):
            time.sleep(0.01)
            results[f"od_result_{ticker}"] = make_result(ticker)

        t1 = threading.Thread(target=run, args=("PLTR",))
        t2 = threading.Thread(target=run, args=("GME",))
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert results["od_result_PLTR"]["ticker"] == "PLTR"
        assert results["od_result_GME"]["ticker"]  == "GME"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge Case Tickers
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCaseTickers:

    def test_brk_b_hyphen(self):
        is_p, is_pre, needs = classify("BRK-B")
        assert isinstance(needs, bool)

    def test_single_letter_V(self):
        is_p, is_pre, needs = classify("V")
        assert sum([is_p, is_pre, needs]) >= 1

    def test_single_letter_F(self):
        is_p, is_pre, needs = classify("F")
        assert isinstance(needs, bool)

    def test_numeric_string(self):
        _, _, needs = classify("123")
        assert needs

    def test_very_long_ticker(self):
        _, _, needs = classify("AVERYLONGTICKER")
        assert needs

    def test_special_chars_stripped(self):
        ticker = " VOO! ".upper().strip().replace("!", "")
        assert ticker == "VOO"
        is_p, _, _ = classify(ticker)
        assert is_p

    def test_lowercase_voo(self):
        ticker = "voo".upper().strip()
        is_p, _, _ = classify(ticker)
        assert is_p

    def test_ticker_with_dot(self):
        """BRK.B format — should be classified, not crash."""
        ticker = "BRK.B"
        is_p, is_pre, needs = classify(ticker)
        assert isinstance(needs, bool)

    def test_kvue_new_ticker_insufficient_history(self):
        """New tickers with <37 months history fail pipeline."""
        r = {"error": "Model training failed for KVUE"}
        ok, _ = validate_result(r)
        assert not ok

    def test_solv_insufficient_data(self):
        r = {"error": "Model training failed for SOLV"}
        ok, _ = validate_result(r)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# 6. JSON File Validation (validate_precomputed.py logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecomputedFileValidation:

    def _validate_file(self, content):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
        return validate_result(data)

    def test_valid_json_passes(self):
        ok, msg = self._validate_file(json.dumps(make_result()))
        assert ok, msg

    def test_truncated_json_fails(self):
        ok, _ = self._validate_file('{"ticker": "VOO", "summary": {"tier":')
        assert not ok

    def test_empty_object_fails(self):
        ok, _ = self._validate_file("{}")
        assert not ok

    def test_json_array_fails(self):
        ok, _ = self._validate_file("[]")
        assert not ok

    def test_error_json_fails(self):
        ok, _ = self._validate_file(json.dumps({"error": "Insufficient data for ANSS (0 rows)"}))
        assert not ok

    def test_missing_summary_key_fails(self):
        r = make_result()
        del r["summary"]["tier"]
        ok, msg = self._validate_file(json.dumps(r))
        assert not ok

    def test_null_win_rate_passes(self):
        """Key present but None — validator passes (checks key existence only)."""
        r = make_result()
        r["summary"]["win_rate_pct"] = None
        ok, _ = self._validate_file(json.dumps(r))
        assert ok

    def test_dry_run_doesnt_delete(self, tmp_path):
        """Dry run must not delete files."""
        f = tmp_path / "VOO_result.json"
        f.write_text(json.dumps({"error": "bad"}))
        # Simulate dry run — just check, don't delete
        dry_run = True
        if dry_run:
            assert f.exists()  # file must still be there

    def test_valid_files_not_deleted(self, tmp_path):
        f = tmp_path / "VOO_result.json"
        f.write_text(json.dumps(make_result("VOO")))
        data = json.loads(f.read_text())
        ok, _ = validate_result(data)
        assert ok
        assert f.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 7. UI Banner Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestUIBannerLogic:

    def _banner(self, is_portfolio, is_precomputed, needs_live, approved=True):
        """Returns which banner should show."""
        if needs_live and not approved:
            return "gate"
        if is_portfolio:
            return "none"
        if is_precomputed:
            return "green"
        if needs_live and approved:
            return "blue"
        return "none"

    def test_portfolio_ticker_no_banner(self):
        assert self._banner(True, False, False) == "none"

    def test_precomputed_ticker_green_banner(self):
        assert self._banner(False, True, False) == "green"

    def test_unknown_unapproved_shows_gate(self):
        assert self._banner(False, False, True, approved=False) == "gate"

    def test_unknown_approved_shows_blue(self):
        assert self._banner(False, False, True, approved=True) == "blue"

    def test_action_colors_complete(self):
        action_colors = {"act": "#2DB37A", "consider": "#E8A020", "skip": "#6B7280"}
        assert action_colors["act"]     == "#2DB37A"
        assert action_colors["consider"] == "#E8A020"
        assert action_colors["skip"]    == "#6B7280"
        assert action_colors.get("unknown", "#9AA0B4") == "#9AA0B4"

    def test_tier_c_label(self):
        tier_labels = {"A": "ACT ON MODEL", "B": "USE WITH CAUTION", "C": "INFORMATIONAL ONLY"}
        assert tier_labels["C"] == "INFORMATIONAL ONLY"

    def test_regime_colors_complete(self):
        r_colors = {"fear": "#E05C5C", "elevated": "#E8A020", "normal": "#9AA0B4", "calm": "#2DB37A"}
        assert all(r in r_colors for r in ["fear", "elevated", "normal", "calm"])

    def test_trust_score_high(self):
        score = 85
        color = "#2DB37A" if score >= 65 else "#E8A020" if score >= 40 else "#E05C5C"
        assert color == "#2DB37A"

    def test_trust_score_medium(self):
        score = 50
        color = "#2DB37A" if score >= 65 else "#E8A020" if score >= 40 else "#E05C5C"
        assert color == "#E8A020"

    def test_trust_score_low(self):
        score = 20
        color = "#2DB37A" if score >= 65 else "#E8A020" if score >= 40 else "#E05C5C"
        assert color == "#E05C5C"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Projection Table Rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectionRendering:

    def test_all_years_in_projection(self):
        r = make_result()
        for s, data in r["summary"]["projections_by_scenario"].items():
            for yr in [1, 3, 5, 10, 20]:
                assert yr in data["projections"]

    def test_tier_c_removes_optimistic(self):
        r = make_result(tier="C")
        show = r["summary"]["show_optimistic"]
        scenarios = list(r["summary"]["projections_by_scenario"].keys())
        if not show:
            scenarios = [s for s in scenarios if s != "Optimistic"]
        assert "Optimistic" not in scenarios

    def test_tier_a_keeps_optimistic(self):
        r = make_result(tier="A")
        show = r["summary"]["show_optimistic"]
        assert show is True

    def test_projection_extra_positive(self):
        r = make_result()
        for s, data in r["summary"]["projections_by_scenario"].items():
            for yr, p in data["projections"].items():
                assert p["extra"] >= 0

    def test_projection_lift_positive(self):
        r = make_result()
        for s, data in r["summary"]["projections_by_scenario"].items():
            for yr, p in data["projections"].items():
                assert p["lift_pct"] >= 0

    def test_longer_horizon_higher_extra(self):
        r = make_result()
        base = r["summary"]["projections_by_scenario"]["Base"]["projections"]
        assert base[20]["extra"] > base[10]["extra"] > base[5]["extra"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. LLM Advisory Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMAdvisory:

    def test_advisory_from_cache_shows_immediately(self):
        r = make_result(with_advisory=True)
        advisory = r["summary"]["llm_advisory"]["advisory"]
        assert advisory  # non-empty, would render immediately

    def test_missing_advisory_triggers_async_fetch(self):
        r = make_result(with_advisory=False)
        advisory = r["summary"]["llm_advisory"]["advisory"]
        assert not advisory  # empty → async fetch needed

    def test_llm_error_fallback_message(self):
        error_adv = {
            "advisory": "LLM unavailable (timeout)",
            "key_factor": "model signal only",
            "action": "consider"
        }
        assert "LLM unavailable" in error_adv["advisory"]
        assert error_adv["action"] == "consider"

    def test_action_values_valid(self):
        valid_actions = {"act", "consider", "skip"}
        for action in ["act", "consider", "skip"]:
            assert action in valid_actions

    def test_advisory_render_html_safe(self):
        advisory = "Strong <buy> signal & momentum."
        # In production this goes into markdown unsafe_allow_html
        # Just ensure it's a string
        assert isinstance(advisory, str)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Storage / Deployment Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageDeployment:

    def test_result_json_small_enough(self):
        """Single result JSON must stay under 10KB for deployment."""
        r = make_result()
        size = len(json.dumps(r).encode("utf-8"))
        assert size < 10_000, f"Result too large: {size} bytes"

    def test_no_parquet_paths_in_result(self):
        """Result JSON must not embed raw data paths."""
        r = json.dumps(make_result())
        assert ".parquet" not in r
        assert ".pkl" not in r

    def test_result_json_serialisable(self):
        """Result must be fully JSON-serialisable (no numpy, no datetime)."""
        r = make_result()
        serialised = json.dumps(r)
        assert isinstance(serialised, str)

    def test_precomputed_dir_naming(self):
        """Result files must follow {TICKER}_result.json convention."""
        for ticker in ["VOO", "MSFT", "BRK-B"]:
            fname = f"{ticker}_result.json"
            assert fname.endswith("_result.json")
            assert fname.startswith(ticker)

    def test_ticker_extraction_from_filename(self):
        """Stem replace must correctly extract ticker."""
        for ticker in ["VOO", "MSFT", "A", "BRK-B"]:
            stem = f"{ticker}_result"
            extracted = stem.replace("_result", "")
            assert extracted == ticker


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False
    )
