#!/usr/bin/env python3
"""
SELF-TEST ONLY — runs psx_dashboard.py end to end on SYNTHETIC, MADE-UP prices.

*** THE NUMBERS THIS PRODUCES ARE NOT REAL MARKET DATA. ***
*** Nothing it prints tells you anything about OGDC, LUCK or MEBL. ***

Its only job is to prove the machinery works — the statistics, the Monte Carlo,
the correlated portfolio simulation, the charts and the report framing — on a
machine that cannot reach the Pakistan Stock Exchange or Yahoo Finance.

It writes fake CSVs to a temp directory, points the CSV fallback at them, and
runs the real pipeline. Fake tickers (FAKE_A/B/C) are used deliberately so no
output can ever be mistaken for a real PSX name.

Run:  python selftest_synthetic.py
"""

import os
import tempfile

import numpy as np
import pandas as pd

import psx_dashboard as dash


def make_synthetic_csv(path: str, seed: int, n: int = 520, s0: float = 100.0,
                       ann_vol: float = 0.30, ann_drift: float = 0.10,
                       market: np.ndarray = None, beta: float = 0.6) -> None:
    """Write a fake but structurally realistic daily price CSV.

    A shared `market` factor is mixed in so the three fake series are genuinely
    correlated — otherwise the portfolio/Cholesky path would not be exercised.
    """
    rng = np.random.default_rng(seed)
    dvol = ann_vol / np.sqrt(252)
    ddrift = ann_drift / 252

    idio = rng.standard_normal(n)
    shocks = beta * market + np.sqrt(max(1e-9, 1 - beta**2)) * idio
    prices = s0 * np.exp(np.cumsum(ddrift + dvol * shocks))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(path, index=False)


def main() -> int:
    banner = "*" * 78
    print(banner)
    print("* SELF-TEST WITH SYNTHETIC DATA — THESE PRICES ARE INVENTED.")
    print("* No output below describes any real stock. Do not read it as analysis.")
    print(banner)

    tmp = tempfile.mkdtemp(prefix="psx_selftest_")
    n = 520
    market = np.random.default_rng(0).standard_normal(n)  # shared factor

    specs = [
        ("FAKE_A", 1, 180.0, 0.28, 0.15, 0.7),
        ("FAKE_B", 2, 900.0, 0.35, -0.05, 0.5),
        ("FAKE_C", 3, 260.0, 0.22, 0.08, 0.6),
    ]
    csv_paths = {}
    for ticker, seed, s0, vol, drift, beta in specs:
        p = os.path.join(tmp, f"{ticker}.csv")
        make_synthetic_csv(p, seed=seed, n=n, s0=s0, ann_vol=vol,
                           ann_drift=drift, market=market, beta=beta)
        csv_paths[ticker] = p

    # A ticker with no source at all, to prove the fail-loud skip path works.
    tickers = [t for t, *_ in specs] + ["FAKE_MISSING"]

    dash.CONFIG.update({
        "TICKERS": tickers,
        "USE_PSX_READER": False,      # unreachable here; exercised separately
        "USE_YFINANCE": False,
        "TRY_WEB_FUNDAMENTALS": False,
        "LOCAL_CSV_PATHS": csv_paths,
        "OUTPUT_DIR": os.path.join(tmp, "outputs"),
    })
    dash.SECTORS.update({"FAKE_A": "fake sector A", "FAKE_B": "fake sector B",
                         "FAKE_C": "fake sector C"})
    # Supply fake fundamentals for one name only, so both the "have targets"
    # and the "fundamentals unavailable" branches get exercised.
    dash.FUNDAMENTALS["FAKE_A"] = {
        "pe": 5.5, "eps": 32.0, "dividend_yield_pct": 7.5,
        "debt_to_equity": 0.30, "roe_pct": 18.0,
        "target_avg": 210.0, "target_high": 250.0, "target_low": 170.0,
        "source_note": "INVENTED for the self-test",
    }

    rc = dash.main()

    print()
    print(banner)
    print(f"* SELF-TEST COMPLETE (exit {rc}). Artifacts in {tmp}")
    print("* Reminder: every number above came from synthetic random data.")
    print(banner)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
