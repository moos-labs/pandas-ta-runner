#!/usr/bin/env python3
"""Tests for --as-of (borsagator #436, walk-forward backtest).

--as-of YYYY-MM-DD: fetch full history, drop every row with date >
as-of, fail loudly if fewer than MIN_ROWS (60) sessions remain. The
truncation must happen inside the runner, before any indicator call,
so rows after the cutoff never reach the indicators. The snapshot's
last_date must equal the as-of date (when it is a session in the
series) and first_date_used must be recorded.

Run: python3 tests/test_asof_436.py   (stdlib + pandas only, no
network, no pandas-ta -- fetch_ohlc and compute_indicators are
stubbed for the main() tests; the truncation logic under test does
not depend on indicator math).

Failing-before probe: point TA_UNDER_TEST at the pre-change script
and every test below fails (the helper does not exist there).
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TA_PATH = os.environ.get("TA_UNDER_TEST", REPO / "scripts" / "ta_snapshot.py")

spec = importlib.util.spec_from_file_location("ta_snapshot_asof", TA_PATH)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


def _frame(n=200, start="2024-01-01"):
    """n business-day sessions; close rises so the as-of close is
    distinguishable from any later close."""
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "Open": [10.0 + 0.1 * i for i in range(n)],
            "High": [10.5 + 0.1 * i for i in range(n)],
            "Low": [9.9 + 0.1 * i for i in range(n)],
            "Close": [10.2 + 0.1 * i for i in range(n)],
            "Volume": [100 + i for i in range(n)],
        },
        index=idx,
    )


AS_OF = "2024-06-28"  # a Friday session in the frame

# 1. apply_asof drops every row after the cutoff, keeps the cutoff row.
df = _frame()
out = ta.apply_asof(df, AS_OF, min_rows=20)
n_expected = int((df.index <= AS_OF).sum())
check("apply_asof drops rows after as-of",
      len(out) == n_expected
      and out.index[-1] == pd.Timestamp(AS_OF),
      f"len={len(out)} expected={n_expected} last={out.index[-1]}")
i_asof = df.index.get_loc(pd.Timestamp(AS_OF))
check("apply_asof keeps the as-of row itself",
      out["Close"].iloc[-1] == 10.2 + 0.1 * i_asof)

# 2. A cutoff on a non-session date keeps the last session on/before it.
out2 = ta.apply_asof(df, "2024-06-29", min_rows=20)  # a Saturday
check("non-session as-of keeps last session on/before it",
      out2.index[-1] == pd.Timestamp(AS_OF))

# 3. Too-short remainder fails loudly instead of snapshotting a sliver.
try:
    ta.apply_asof(_frame(70), "2024-01-15", min_rows=20)
    check("too-short remainder raises RuntimeError", False)
except RuntimeError as e:
    check("too-short remainder raises RuntimeError",
          "only 11 sessions" in str(e) and "2024-01-15" in str(e), str(e))

# 4. The default floor is the design's 60-session warm-up (§1.3).
try:
    ta.apply_asof(_frame(70), "2024-03-15")  # 55 sessions <= cutoff
    check("default MIN_ROWS=60 enforced", False)
except RuntimeError as e:
    check("default MIN_ROWS=60 enforced",
          "< 60 minimum rows" in str(e), str(e))
check("MIN_ROWS constant is 60", ta.MIN_ROWS == 60)

# 5. main() wiring: stubbed fetch + stubbed indicators; the snapshot
# must end at as-of and indicators.csv must contain no later row.
seen = {}


def fake_fetch(ticker):
    return _frame()


def fake_compute(df_used):
    seen["indicator_input"] = df_used.copy()
    cols = {k: None for k in (
        "rsi_14", "macd", "macd_signal", "macd_hist", "ema_20", "ema_50",
        "ema_200", "adx_14", "bb_upper", "bb_mid", "bb_lower", "bb_pctb",
        "atr_14", "stoch_k", "stoch_d")}
    return df_used, cols


tmp = Path(tempfile.mkdtemp(prefix="ta436-main-"))
os.chdir(tmp)
ta.fetch_ohlc = fake_fetch
ta.compute_indicators = fake_compute
old_argv = sys.argv
sys.argv = ["ta_snapshot.py", "--ticker", "TEST", "--as-of", AS_OF]
try:
    rc = ta.main()
    check("main() exits 0 with --as-of", rc == 0, f"rc={rc}")
    snap = json.loads((tmp / "results" / "ta_snapshot.json").read_text())
    check("snapshot last_date equals as-of",
          snap["last_date"] == AS_OF, snap["last_date"])
    check("snapshot records first_date_used",
          snap["first_date_used"] == "2024-01-01", snap["first_date_used"])
    check("snapshot records as_of",
          snap["as_of"] == AS_OF, snap.get("as_of"))
    check("snapshot close is the as-of close",
          snap["close"] == round(10.2 + 0.1 * i_asof, 4),
          snap["close"])
    check("no row after as-of reaches the indicators",
          seen["indicator_input"].index.max() == pd.Timestamp(AS_OF),
          str(seen["indicator_input"].index.max()))
    ind_csv = pd.read_csv(tmp / "results" / "indicators.csv",
                          index_col=0, parse_dates=True)
    check("indicators.csv contains no row after as-of",
          ind_csv.index.max() == pd.Timestamp(AS_OF),
          str(ind_csv.index.max()))
    report = (tmp / "results" / "report.md").read_text()
    check("report records the as-of cutoff",
          f"As-of cutoff: **{AS_OF}**" in report, report[:500])
finally:
    sys.argv = old_argv

# 6. main() fails loudly when the truncated series is too short.
tmp2 = Path(tempfile.mkdtemp(prefix="ta436-short-"))
os.chdir(tmp2)


def fake_fetch_short(ticker):
    return _frame(70)


ta.fetch_ohlc = fake_fetch_short
sys.argv = ["ta_snapshot.py", "--ticker", "TEST", "--as-of", "2024-02-01"]
try:
    ta.main()
    check("main(): too-short remainder raises", False)
except RuntimeError as e:
    check("main(): too-short remainder raises",
          "only 24 sessions" in str(e), str(e))
finally:
    sys.argv = old_argv

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
