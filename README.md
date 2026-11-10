# pandas-ta-runner

Runs **pandas-ta-classic** technical analyses via GitHub Actions — the
commands to execute are supplied through the **COMMANDS** repository
variable, and the workflow is triggered manually (`workflow_dispatch`).

This is one repo in a one-public-repo-per-tool pattern: change the
COMMANDS variable, dispatch the workflow, download the `results`
artifact. The design is market-agnostic — the ticker and every analysis
parameter are script inputs; nothing about any specific market is
hard-coded. The same repo later serves other markets (e.g. US equities
via Alpaca/Nsave) by changing inputs only.

## Current pilot

`scripts/ta_snapshot.py` computes a technical snapshot for a ticker:
RSI(14), MACD(12,26,9), EMA(20/50/200), ADX(14), Bollinger(20,2) with %B,
ATR(14) and Stochastic(14,3), and writes `results/ta_snapshot.json`,
`results/indicators.csv` and `results/report.md`.

COMMANDS currently runs it for `POUL.CA` (Cairo Poultry, EGX):

```text
mkdir -p results
python scripts/ta_snapshot.py --ticker POUL.CA 2>&1 | tee results/run.log
```

## Data note

Price history is fetched from Yahoo Finance's chart API directly.
yfinance is deliberately not used: it crashes with
`NonExistentTimeError` on Africa/Cairo timestamps after Egypt reinstated
DST in 2023. POUL.CA's pre-2019 Yahoo segment is corrupted (stale flat
prices on a different scale), so the pilot drops rows before 2019-01-01
via the script's `--drop-before` argument.
