"""
TemporalEdge — S&P 500 Precompute Script  (Phase 8)
====================================================
Runs the full on-demand pipeline for every S&P 500 constituent and saves
results to data/precomputed/{TICKER}_result.json.

Usage:
    python scripts/precompute_sp500.py                  # all tickers
    python scripts/precompute_sp500.py --workers 6      # 6 parallel workers
    python scripts/precompute_sp500.py --resume         # skip already done
    python scripts/precompute_sp500.py --tickers AAPL MSFT NVDA

Runtime on M2 MacBook Air:
    ~45s per ticker  |  4 workers  |  ~6-7 hrs for all 503
    Results ~100KB each  →  ~50MB total
    Resume-safe: each ticker saved as it finishes.
"""

import sys
import json
import time
import argparse
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.utils.logger import get_logger

log = get_logger("scripts.precompute_sp500")

PRECOMPUTED_DIR = ROOT / "data" / "precomputed"
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE   = PRECOMPUTED_DIR / "_progress.json"

# ── S&P 500 + high-demand ETFs (April 2026) ───────────────────────────────────
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
    "ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG",
    "AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BK","BBWI","BAX","BDX","BMY","AVGO","BR","BRO","BG","CDNS","CZR","CPT",
    "CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR",
    "CNC","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF",
    "CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA",
    "CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA",
    "CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DTE","DUK",
    "DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH",
    "ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
    "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
    "FITB","FSLR","FE","FIS","FI","FLT","FMC","F","FTNT","FTV","FOXA","FOX",
    "BEN","FCX","GRMN","IT","GEHC","GEN","GNRC","GD","GE","GIS","GM","GPC",
    "GILD","GPN","GL","GS","HAL","HIG","HAS","HCA","HSIC","HSY","HES","HPE",
    "HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII",
    "IBM","IEX","IDXX","ITW","ILMN","INCY","IR","PODD","INTC","ICE","IFF","IP",
    "IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J",
    "JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI",
    "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LNC","LIN",
    "LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR",
    "MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET",
    "MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ",
    "MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM",
    "NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG",
    "NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL",
    "OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE",
    "PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD",
    "PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX","RL","RJF","RTX","O",
    "REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL",
    "SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM",
    "SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF",
    "SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX",
    "TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB",
    "TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI",
    "UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX","VFC","VTRS","VICI",
    "V","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC",
    "WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","WYNN","XEL","XYL",
    "YUM","ZBRA","ZBH","ZION","ZTS",
    # High-demand ETFs
    "QQQ","SPY","IVV","VTI","VOO","VEA","VWO","VXUS","BND","AGG","SCHD",
    "JEPI","JEPQ","ARKK","XLK","XLF","XLE","XLV","XLY","XLP","GLD","SLV",
    "VYM","BRK-B","IWM","DIA","SOXX","SMH",
]

# Deduplicate while preserving order
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))


def _result_path(ticker: str) -> Path:
    return PRECOMPUTED_DIR / f"{ticker}_result.json"


def _is_done(ticker: str) -> bool:
    p = _result_path(ticker)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=35)


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"done": [], "failed": [], "started_at": None, "updated_at": None}


def _save_progress(prog: dict):
    prog["updated_at"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))


def precompute_ticker(ticker: str, monthly_usd: float = 10.0) -> tuple:
    """Run pipeline for one ticker. Returns (ticker, success, message)."""
    try:
        from src.pipeline.on_demand import run_on_demand
        result = run_on_demand(
            ticker=ticker,
            monthly_usd=monthly_usd,
            horizon_years=10,
            n_runs=1000,
            force_retrain=False,
            progress_cb=None,
        )
        if result:
            with open(_result_path(ticker), "w") as f:
                json.dump(result, f, indent=2, default=str)
            win = result.get("summary", {}).get("win_rate_pct", "?")
            return ticker, True, f"win={win}%"
        return ticker, False, "pipeline returned None"
    except Exception as e:
        return ticker, False, str(e)[:80]


def run_precompute(
    tickers=None,
    workers: int = 4,
    resume: bool = True,
    monthly_usd: float = 10.0,
):
    target   = tickers or SP500_TICKERS
    progress = _load_progress()

    if resume:
        todo = [t for t in target if not _is_done(t)]
        skip = len(target) - len(todo)
        if skip:
            log.info(f"  Resuming — {skip} already done, {len(todo)} remaining")
    else:
        todo = target

    if not todo:
        log.info("  All tickers precomputed. Use --force to redo.")
        return

    total    = len(todo)
    done_cnt = 0
    fail_cnt = 0
    t0       = time.time()
    lock     = threading.Lock()

    if not progress["started_at"]:
        progress["started_at"] = datetime.now().isoformat()

    log.info("=" * 60)
    log.info("  TEMPORALEDGE — S&P 500 PRECOMPUTE  (Phase 8)")
    log.info(f"  {total} tickers | {workers} workers | ~{total*45/workers/3600:.1f} hrs est.")
    log.info(f"  Output: {PRECOMPUTED_DIR}")
    log.info("=" * 60)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(precompute_ticker, t, monthly_usd): t for t in todo}

            for future in concurrent.futures.as_completed(futures):
                ticker, success, msg = future.result()

                with lock:
                    done_cnt += 1
                    elapsed   = time.time() - t0
                    rate      = done_cnt / elapsed
                    remaining = (total - done_cnt) / rate if rate > 0 else 0
                    eta       = datetime.now() + timedelta(seconds=remaining)

                    if success:
                        progress["done"].append(ticker)
                        log.info(f"  [{done_cnt:3d}/{total}] ✓ {ticker:<8} {msg} | ETA {eta.strftime('%H:%M')}")
                    else:
                        fail_cnt += 1
                        progress["failed"].append({"ticker": ticker, "error": msg})
                        log.warning(f"  [{done_cnt:3d}/{total}] ✗ {ticker:<8} FAILED: {msg}")

                    if done_cnt % 10 == 0:
                        _save_progress(progress)

    except KeyboardInterrupt:
        log.info("\n  Ctrl+C detected — saving progress and exiting cleanly...")
        log.info(f"  Completed {done_cnt} tickers this session.")
        log.info("  Run the same command again to resume where you left off.")

    _save_progress(progress)
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"  DONE in {elapsed/3600:.1f} hrs | {done_cnt-fail_cnt} ok | {fail_cnt} failed")
    if fail_cnt:
        log.warning(f"  Failed: {[f['ticker'] for f in progress['failed']]}")
        log.warning("  Re-run with --resume to retry failed tickers")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute S&P 500 timing results")
    parser.add_argument("--workers",  type=int,   default=4)
    parser.add_argument("--resume",   action="store_true", default=True)
    parser.add_argument("--force",    action="store_true", default=False)
    parser.add_argument("--tickers",  nargs="+")
    parser.add_argument("--monthly",  type=float, default=10.0)
    parser.add_argument("--dry-run",  action="store_true", default=False,
                        dest="dry_run", help="Show what would run without executing")
    args = parser.parse_args()

    if args.dry_run:
        target = args.tickers or SP500_TICKERS
        todo   = [t for t in target if not _is_done(t)]
        done   = len(target) - len(todo)
        print(f"\nDry run — {len(target)} tickers total")
        print(f"  Already cached: {done}")
        print(f"  Would run:      {len(todo)}")
        print(f"  Est. time:      ~{len(todo)*45/args.workers/3600:.1f} hrs ({args.workers} workers)")
        if todo:
            print(f"\n  First 20 to run:")
            for t in todo[:20]:
                print(f"    {t}")
            if len(todo) > 20:
                print(f"    ... and {len(todo)-20} more")
    else:
        run_precompute(
            tickers=args.tickers,
            workers=args.workers,
            resume=not args.force,
            monthly_usd=args.monthly,
        )
