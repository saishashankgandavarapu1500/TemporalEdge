"""
validate_precomputed.py — check all result JSONs in data/precomputed/
Usage:
  python scripts/validate_precomputed.py            # delete bad files
  python scripts/validate_precomputed.py --dry-run  # preview only
"""
import json
import sys
from pathlib import Path

PRECOMPUTED_DIR = Path(__file__).parent.parent / "data" / "precomputed"
DRY_RUN = "--dry-run" in sys.argv

# Minimum keys required inside summary for a result to be valid
REQUIRED_SUMMARY_KEYS = {"tier", "trust_score", "win_rate_pct"}


def _validate(data: dict) -> tuple[bool, str]:
    # Must be a non-empty dict
    if not isinstance(data, dict) or not data:
        return False, "empty or non-dict"

    # If pipeline wrote an error, it's invalid
    if data.get("error"):
        return False, f"error field set: {data['error']}"

    # Must have a summary dict with required keys
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return False, "missing summary"

    missing = REQUIRED_SUMMARY_KEYS - summary.keys()
    if missing:
        return False, f"summary missing keys: {missing}"

    return True, "ok"


def validate():
    files = sorted(PRECOMPUTED_DIR.glob("*_result.json"))
    if not files:
        print(f"\nNo result files found in {PRECOMPUTED_DIR}")
        return

    ok_list, bad_list = [], []

    for f in files:
        ticker = f.stem.replace("_result", "")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            bad_list.append((ticker, f, f"invalid JSON: {e}"))
            continue
        except Exception as e:
            bad_list.append((ticker, f, f"read error: {e}"))
            continue

        ok, reason = _validate(data)
        if ok:
            ok_list.append(ticker)
        else:
            bad_list.append((ticker, f, reason))

    print(f"\n{'='*55}")
    print(f"  VALIDATION RESULTS  {'[DRY RUN]' if DRY_RUN else ''}")
    print(f"{'='*55}")
    print(f"  ✓ Valid:   {len(ok_list)}")
    print(f"  ✗ Invalid: {len(bad_list)}")
    print(f"  Total:     {len(files)}")
    print(f"{'='*55}")

    if bad_list:
        print(f"\n  CORRUPT / INCOMPLETE FILES:")
        for ticker, path, reason in bad_list:
            print(f"    ✗ {ticker:<10} — {reason}")
            if not DRY_RUN:
                path.unlink()
                print(f"             → deleted")
            else:
                print(f"             → [dry-run] would delete")

        if DRY_RUN:
            print(f"\n  Re-run without --dry-run to delete and reprocess.")
        else:
            print(f"\n  Deleted {len(bad_list)} files.")
    else:
        print("\n  All files are valid! ✓")

    if ok_list:
        sample = ok_list[:5]
        print(f"\n  Sample valid: {', '.join(sample)}{'...' if len(ok_list) > 5 else ''}")


if __name__ == "__main__":
    validate()
