# PSX portfolio risk & context dashboard

A reusable Python tool for a small Pakistan Stock Exchange basket (default
**OGDC, LUCK, MEBL**). It answers three separate questions and keeps them
separate on purpose:

| | Question | Where it comes from |
|---|---|---|
| **FACT** | What is factually true about this stock today? | measured from real price history |
| **UNCERTAINTY** | How wide is the range of plausible outcomes over ~60 days? | Monte Carlo simulation |
| **EXTERNAL VIEW** | What do other people think? | analyst targets you supply / scrape |

**It never presents a single future price as a prediction.** Every forward-looking
number is a percentile, always quoted with its range. A guard in the code
(`Reporter`) raises an error if overpromising language like "guaranteed" ever
reaches the output.

## Install

```bash
pip install -r requirements.txt
```

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the dashboard in your browser. The sidebar controls the whole run:

- **Tickers** — comma- or space-separated; add, remove or replace names freely.
- **Horizon** — a slider in calendar days (default 60), converted to trading
  days internally.
- **Capital per stock (USD)** and **USD/PKR rate**.
- **Drift scenario** — zero drift (default) or historical drift. Both scenarios
  are always computed and shown; this chooses which one the charts and the
  portfolio model are built on.

Press **Run analysis** and, per stock, you get the FACT table, the fan chart
with the analyst-target markers, and the full P10-P90 uncertainty range,
followed by the portfolio correlation matrix, terminal-value histogram and
diversification numbers, and the same "how to read this" framing the CLI
prints. `app.py` contains no modelling of its own — it calls the same functions
as the command line, and every string it renders passes through the same
honesty guard.

## Run the command line

```bash
python psx_dashboard.py
```

Charts land in `./outputs/`, the written report goes to stdout.

## Use it from your own code

Computation is separate from printing, so you can import the pipeline and get
results back as data:

```python
import psx_dashboard as dash

result = dash.run_analysis(
    tickers=["OGDC", "LUCK", "MEBL"],
    horizon_days=60,              # calendar days; converted to trading days
    capital_per_stock=10.0,
    usd_pkr=280.0,
    drift_scenario="zero",        # or "historical"
    with_figures=True,
)

result.profiles                   # measured StockStats per surviving ticker
result.cones["OGDC"]["zero"].pct  # {10: …, 25: …, 50: …, 75: …, 90: …}
result.portfolio.pct              # basket value percentiles, PKR
result.figures["fan_OGDC"]        # matplotlib Figure, unsaved
result.skipped                    # tickers with no usable data
result.log                        # the data trail, line by line
```

`run_analysis` prints nothing (pass `progress=print` to stream its progress
lines). `dash.save_figures(result, "outputs")` writes the charts, and
`dash.report_run(result)` prints the full written report.

## Configuration

Everything you would want to change is in the **CONFIG BLOCK** at the top of
`psx_dashboard.py`:

- `TICKERS`, `HISTORY_YEARS`, `MIN_ROWS` (data quality floor, default 400 rows)
- `HORIZON_TRADING_DAYS` (42 ≈ 60 calendar days), `N_PATHS` (10,000), `RANDOM_SEED`
- `STAT_WINDOW_DAYS` — the window volatility and drift are measured over (252 = last year)
- `CAPITAL_USD_PER_STOCK` ($10) and `USD_PKR` (280)
- `DRIFT_SCENARIO` — `"zero"` (default) or `"historical"`
- `LOCAL_CSV_PATHS` — your own CSVs, used if the live sources fail
- `FUNDAMENTALS` — per-ticker P/E, EPS, dividend yield, debt/equity, ROE and
  12-month analyst targets (avg/high/low). The valuation fields ship **empty**:
  no fundamental figures are invented for you, and every printed field is
  labelled with whether it came from the web or from your config. Analyst
  targets for OGDC, LUCK and MEBL are pre-filled from consensus figures dated
  ~Aug 2026 — they are **someone else's opinion, unverified by this tool, and
  worth re-checking before you rely on them**. A web scrape that succeeds takes
  precedence over these config values, and the source is printed either way.

## Data sources, in order

1. **`psx-data-reader`** — daily prices direct from the exchange.
2. **yfinance** with a `.KA` suffix (`OGDC.KA`, …).
3. **Your local CSV** per ticker, if you set `LOCAL_CSV_PATHS`. Any CSV with a
   date column and a close column works; column names are auto-detected.

The `psx-data-reader` package hard-codes `set_index("TIME")` when it parses the
exchange page, so the moment those table headers change — or the page comes back
without a table — every ticker dies with
`KeyError: "None of ['TIME'] are in the columns"` thrown inside the library. The
adapter here installs a tolerant parser for the duration of the call: it reads
whatever headers the page actually has, finds the date column by name instead of
assuming one (`TIME`, `DATE`, `Date`, … all match), prints the layout it saw once
for diagnosis, and normalises the result into the `Date`/`Close` shape the rest
of the pipeline expects — whether the date arrives as the index or as a column.
If nothing usable is there, **that one source** fails with a clear message and
the run continues to the yfinance fallback.

Each series must pass validation — at least `MIN_ROWS` rows, positive prices,
no stale/flat feed — or it is **rejected**. If a ticker fails every source it is
skipped with the exact reason from each source printed, and nothing is
fabricated to fill the gap. If no ticker survives, the run exits non-zero
rather than pretending.

## What the model does

Geometric Brownian motion in log space, using the **measured** volatility of the
stock. Two scenarios run side by side:

- **Scenario B — zero drift (the honest default).** No view on direction. This
  is what the fan charts and the portfolio model use.
- **Scenario A — historical drift.** Assumes the last year's average daily move
  continues. The report prints a caution that this over-extrapolates a past run,
  especially for a stock sitting near its highs.

Both are always computed and always reported. `DRIFT_SCENARIO` (or the sidebar
radio) only chooses which one the charts and the portfolio model are built on,
and every chart subtitle names the drift it actually used.

The portfolio model is **multivariate**: shocks are drawn together using the
Cholesky factor of the real covariance matrix, so the three names move together
in the simulation exactly as much as they did in reality. It reports the P10/P50/P90
of total basket value, the probability of being up, the average worst drawdown
along the paths, and how much narrower the combined 80% band is than the average
single-name band — the diversification benefit, quantified.

## Outputs

- `outputs/fan_<TICKER>.png` — the cone of simulated prices: median path, 50% and
  80% bands, today's price, and your analyst target range if supplied — drawn as
  faint horizontal levels plus a marked low/avg/high range past the horizon,
  because a 12-month target is not a claim about the next 60 days.
- `outputs/portfolio_terminal_values.png` — histogram of the 10,000 simulated
  basket values with P10/P50/P90 and starting capital marked.

## Limits you should hold in mind

This tool models **price randomness only**. It knows nothing about earnings,
dividends yet to be declared, oil prices, interest rates, currency moves,
politics, regulation, or company announcements. It assumes each day's move is
drawn independently from a fixed distribution, which real markets violate —
returns have fatter tails than this and volatile periods cluster. Treat the
cones as a rough sense of scale under normal conditions, and assume reality can
land outside them. Nothing here is investment advice.

## Self-test

```bash
python selftest_synthetic.py
```

Runs the whole pipeline on **synthetic, invented** price data with fake tickers
(`FAKE_A/B/C`), so the statistics, simulations, charts and report framing can be
exercised on a machine with no access to PSX or Yahoo. Its numbers describe
nothing real — the banner says so, loudly.
