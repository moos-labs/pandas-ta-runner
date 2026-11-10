#!/usr/bin/env python3
"""Technical-analysis snapshot — market-agnostic runner script.

Generic design: the ticker, the optional drop-before cutoff and the output
location are INPUTS (CLI args / the COMMANDS workflow variable), never
hard-coded market assumptions. Any Yahoo-listed symbol from any market can
be analysed by changing the inputs only.

Data comes from Yahoo Finance's chart API, called DIRECTLY with requests.
yfinance is deliberately NOT used: it localises session timestamps and
crashes with NonExistentTimeError on exchanges whose historical local
midnights never existed (e.g. Africa/Cairo after Egypt reinstated DST in
2023 — 2023-04-28 00:00 never happened). Here, the epoch instants returned
by the chart API are converted to session *dates* in the exchange timezone
reported by the payload itself (meta.exchangeTimezoneName) via
tz_convert + .date, which never localises a (possibly nonexistent) local
midnight. No exchange-specific handling anywhere.

Indicators (pandas-ta-classic), computed on the daily OHLC series:
  RSI(14), MACD(12,26,9), EMA(20), EMA(50), EMA(200), ADX(14),
  Bollinger Bands(20, 2) including %B, ATR(14), Stochastic(14,3).

Outputs (into results/):
  ta_snapshot.json  latest row's values + last close + its date
  indicators.csv    last 250 rows of OHLC + computed indicators
  report.md         snapshot table + short factual trend read

If the ticker returns no data, the verdict is written into the results and
the script exits 0 — the loop working plus the data verdict IS the result.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

RESULTS_DIR = "results"


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Timezone-safe fetch: Yahoo chart API directly via requests.
# ---------------------------------------------------------------------------

def fetch_ohlc(ticker):
    import requests

    p1 = 0
    p2 = int(time.time()) + 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
    resp = requests.get(
        url,
        params={"period1": p1, "period2": p2, "interval": "1d",
                "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error: {chart['error']}")
    result = chart["result"][0]
    stamps = result.get("timestamp") or []
    if not stamps:
        return None
    quote = result["indicators"]["quote"][0]
    meta = result.get("meta", {})
    tz_name = meta.get("exchangeTimezoneName") or "UTC"
    idx_utc = pd.to_datetime(stamps, unit="s", utc=True)
    try:
        dates = idx_utc.tz_convert(tz_name).date
    except Exception:
        dates = idx_utc.date
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=index,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    return df if len(df) else None


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _find_col(df, prefix, exclude=()):
    for c in df.columns:
        if c.startswith(prefix) and c not in exclude:
            return c
    return None


def compute_indicators(df):
    import pandas_ta_classic as ta  # noqa: F401  (registers the .ta accessor)

    out = df.copy()
    out.columns = [c.capitalize() if c.lower() in
                   ("open", "high", "low", "close", "volume") else c
                   for c in out.columns]

    out.ta.rsi(length=14, append=True)
    out.ta.macd(fast=12, slow=26, signal=9, append=True)
    out.ta.ema(length=20, append=True)
    out.ta.ema(length=50, append=True)
    out.ta.ema(length=200, append=True)
    out.ta.adx(length=14, append=True)
    out.ta.bbands(length=20, std=2, append=True)
    out.ta.atr(length=14, append=True)
    out.ta.stoch(k=14, d=3, smooth_k=3, append=True)

    cols = {
        "rsi_14": _find_col(out, "RSI_"),
        "macd": _find_col(out, "MACD_"),
        "macd_signal": _find_col(out, "MACDs_"),
        "macd_hist": _find_col(out, "MACDh_"),
        "ema_20": "EMA_20" if "EMA_20" in out.columns else None,
        "ema_50": "EMA_50" if "EMA_50" in out.columns else None,
        "ema_200": "EMA_200" if "EMA_200" in out.columns else None,
        "adx_14": _find_col(out, "ADX_"),
        "bb_upper": _find_col(out, "BBU_"),
        "bb_mid": _find_col(out, "BBM_"),
        "bb_lower": _find_col(out, "BBL_"),
        "bb_pctb": _find_col(out, "BBP_"),
        "atr_14": _find_col(out, "ATR"),
        "stoch_k": _find_col(out, "STOCHk_"),
        "stoch_d": _find_col(out, "STOCHd_"),
    }
    return out, cols


def _val(row, col):
    if col is None:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)


def build_snapshot(ticker, df, cols, rows_total, rows_used, drop_before):
    last = df.iloc[-1]
    snap = {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "open": _val(last, "Open"),
        "high": _val(last, "High"),
        "low": _val(last, "Low"),
        "close": _val(last, "Close"),
        "rsi_14": _val(last, cols["rsi_14"]),
        "macd": _val(last, cols["macd"]),
        "macd_signal": _val(last, cols["macd_signal"]),
        "macd_hist": _val(last, cols["macd_hist"]),
        "ema_20": _val(last, cols["ema_20"]),
        "ema_50": _val(last, cols["ema_50"]),
        "ema_200": _val(last, cols["ema_200"]),
        "adx_14": _val(last, cols["adx_14"]),
        "bb_upper": _val(last, cols["bb_upper"]),
        "bb_mid": _val(last, cols["bb_mid"]),
        "bb_lower": _val(last, cols["bb_lower"]),
        "bb_pctb": _val(last, cols["bb_pctb"]),
        "atr_14": _val(last, cols["atr_14"]),
        "stoch_k": _val(last, cols["stoch_k"]),
        "stoch_d": _val(last, cols["stoch_d"]),
        "rows_fetched": rows_total,
        "rows_used": rows_used,
        "first_date_used": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "drop_before": drop_before,
    }
    return snap


def _vs(close, ema):
    if ema is None:
        return "n/a"
    return "above" if close >= ema else "below"


def trend_read(snap):
    """Short factual read — positions and zones only, no advice."""
    lines = []
    close = snap["close"]
    for name in ("ema_20", "ema_50", "ema_200"):
        v = snap[name]
        if v is None:
            lines.append(f"- Close vs {name.upper()}: not available "
                         "(insufficient history).")
        else:
            pct = (close - v) / v * 100.0
            lines.append(f"- Close {close} is {_vs(close, v)} "
                         f"{name.upper()} {v} ({pct:+.2f}%).")
    rsi = snap["rsi_14"]
    if rsi is not None:
        if rsi >= 70:
            zone = "overbought zone (>= 70)"
        elif rsi <= 30:
            zone = "oversold zone (<= 30)"
        else:
            zone = "neutral zone (30-70)"
        lines.append(f"- RSI(14) is {rsi}, in the {zone}.")
    if snap["macd"] is not None and snap["macd_signal"] is not None:
        rel = "above" if snap["macd"] >= snap["macd_signal"] else "below"
        lines.append(f"- MACD {snap['macd']} is {rel} its signal line "
                     f"{snap['macd_signal']} (histogram {snap['macd_hist']}).")
    if snap["bb_pctb"] is not None:
        lines.append(f"- Price sits at %B {snap['bb_pctb']} of the "
                     f"Bollinger range (lower {snap['bb_lower']}, "
                     f"mid {snap['bb_mid']}, upper {snap['bb_upper']}).")
    adx = snap["adx_14"]
    if adx is not None:
        if adx >= 25:
            strength = "a strong trend by the standard ADX scale (>= 25)"
        elif adx >= 20:
            strength = "a developing trend (20-25)"
        else:
            strength = "a weak / no clear trend (< 20)"
        lines.append(f"- ADX(14) is {adx}, indicating {strength}.")
    if snap["stoch_k"] is not None and snap["stoch_d"] is not None:
        rel = "above" if snap["stoch_k"] >= snap["stoch_d"] else "below"
        lines.append(f"- Stochastic %K {snap['stoch_k']} is {rel} "
                     f"%D {snap['stoch_d']}.")
    if snap["atr_14"] is not None:
        lines.append(f"- ATR(14) is {snap['atr_14']} "
                     f"({snap['atr_14'] / close * 100.0:.2f}% of the close).")
    return lines


def write_report(path, snap):
    lines = []
    lines.append(f"# Technical snapshot — {snap['ticker']}")
    lines.append("")
    if snap.get("close") is None and snap.get("verdict"):
        lines.append("## Data verdict")
        lines.append("")
        lines.append(snap["verdict"])
        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    lines.append(f"- Last session: **{snap['date']}**")
    lines.append(f"- Data used: {snap['rows_used']} rows "
                 f"({snap['first_date_used']} → {snap['last_date']})"
                 + (f", after dropping rows before {snap['drop_before']}"
                    if snap["drop_before"] else "")
                 + f"; {snap['rows_fetched']} rows fetched in total.")
    lines.append("")
    lines.append("## Latest values")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    labels = [
        ("Open", "open"), ("High", "high"), ("Low", "low"),
        ("Close", "close"), ("RSI(14)", "rsi_14"),
        ("MACD", "macd"), ("MACD signal", "macd_signal"),
        ("MACD histogram", "macd_hist"),
        ("EMA(20)", "ema_20"), ("EMA(50)", "ema_50"),
        ("EMA(200)", "ema_200"), ("ADX(14)", "adx_14"),
        ("Bollinger upper", "bb_upper"), ("Bollinger mid", "bb_mid"),
        ("Bollinger lower", "bb_lower"), ("Bollinger %B", "bb_pctb"),
        ("ATR(14)", "atr_14"), ("Stochastic %K", "stoch_k"),
        ("Stochastic %D", "stoch_d"),
    ]
    for label, key in labels:
        v = snap.get(key)
        lines.append(f"| {label} | {'—' if v is None else v} |")
    lines.append("")
    lines.append("## Trend read (facts only)")
    lines.append("")
    lines.extend(trend_read(snap))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Source: Yahoo Finance chart API (raw daily OHLC, "
                 "not dividend-adjusted), fetched directly — yfinance is "
                 "not used because it crashes on Africa/Cairo DST "
                 "timestamps.")
    lines.append("- Indicators computed with pandas-ta-classic.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="POUL.CA",
                    help="Yahoo Finance ticker (any market).")
    ap.add_argument("--drop-before", default="2019-01-01",
                    help="Drop sessions before this date (YYYY-MM-DD) "
                         "before computing indicators; pass 'none' to "
                         "keep the full history. The pilot default drops "
                         "POUL.CA's corrupted pre-2019 segment (stale, "
                         "flat prices on a different scale).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    drop_before = None if args.drop_before.lower() == "none" else args.drop_before

    try:
        df = fetch_ohlc(args.ticker)
    except Exception as exc:
        df = None
        log(f"[data] {args.ticker}: ERROR {exc}")

    if df is None:
        snap = {"ticker": args.ticker, "close": None,
                "verdict": f"NO DATA: {args.ticker} returned no rows "
                           "from the Yahoo chart API on this runner."}
        log("[verdict] " + snap["verdict"])
        with open(os.path.join(RESULTS_DIR, "ta_snapshot.json"), "w",
                  encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        write_report(os.path.join(RESULTS_DIR, "report.md"), snap)
        return 0

    rows_total = int(len(df))
    log(f"[data] {args.ticker}: {rows_total} rows, "
        f"{df.index[0].date()} -> {df.index[-1].date()}, "
        f"last close {df['Close'].iloc[-1]}")

    if drop_before:
        cutoff = pd.Timestamp(drop_before)
        df_used = df[df.index >= cutoff]
        if len(df_used) == 0:
            df_used = df
        log(f"[data] after drop-before {drop_before}: {len(df_used)} rows")
    else:
        df_used = df

    ind, cols = compute_indicators(df_used)
    snap = build_snapshot(args.ticker, ind, cols, rows_total,
                          int(len(df_used)), drop_before)

    with open(os.path.join(RESULTS_DIR, "ta_snapshot.json"), "w",
              encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    keep = ["Open", "High", "Low", "Close", "Volume"] + \
           [c for c in cols.values() if c]
    tail = ind.tail(250)
    tail_out = tail[[c for c in keep if c in tail.columns]].copy()
    tail_out.index.name = "Date"
    tail_out.to_csv(os.path.join(RESULTS_DIR, "indicators.csv"))

    write_report(os.path.join(RESULTS_DIR, "report.md"), snap)
    log(f"[done] snapshot for {args.ticker} on {snap['date']}: "
        f"close {snap['close']}, RSI {snap['rsi_14']}, "
        f"MACD {snap['macd']} vs signal {snap['macd_signal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
