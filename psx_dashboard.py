#!/usr/bin/env python3
"""
PSX portfolio risk-and-context dashboard.

What this tool does:
  * Pulls real daily price history for a set of Pakistan Stock Exchange tickers.
  * Measures what is factually true today (price, volatility, trend, momentum).
  * Models how uncertain the next ~60 calendar days are, as a RANGE, never a number.
  * Models the three stocks together, with their real correlations, as a basket.

What this tool explicitly does NOT do:
  * It never outputs a single future price as a prediction.
  * It knows nothing about earnings, news, politics, or company events.
  * It models price randomness only. That is a narrow slice of reality.

Run:  python psx_dashboard.py
"""

# =============================================================================
# ============================  CONFIG BLOCK  =================================
# =====  Everything you would normally want to change lives right here.  ======
# =============================================================================

CONFIG = {
    # --- Universe ------------------------------------------------------------
    "TICKERS": ["OGDC", "LUCK", "MEBL"],

    # --- History -------------------------------------------------------------
    "HISTORY_YEARS": 2,      # how much daily history to pull
    "MIN_ROWS": 400,         # a ticker with fewer rows than this is REJECTED,
                             # not patched. ~400 trading days ≈ 1.6 years.

    # --- Forecast horizon ----------------------------------------------------
    "HORIZON_TRADING_DAYS": 42,   # ~60 calendar days (PSX trades ~5 days/week)
    "N_PATHS": 10_000,            # Monte Carlo paths per scenario
    "RANDOM_SEED": 42,            # fixed so the same run reproduces exactly

    # --- Measurement windows -------------------------------------------------
    # Volatility and drift are measured over the most recent N trading days.
    # 252 = last ~1 year, which reflects the CURRENT regime better than the full
    # 2-year sample. The full-sample number is printed alongside for comparison.
    "STAT_WINDOW_DAYS": 252,

    # --- Portfolio sizing ----------------------------------------------------
    "CAPITAL_USD_PER_STOCK": 10.0,   # equal capital into each surviving ticker
    "USD_PKR": 280.0,                # FX rate used to convert USD -> PKR
    # Fractional share counts are used deliberately: this is a MODEL of a basket,
    # not a broker order. Real PSX lots are whole shares.

    # --- Data sources --------------------------------------------------------
    # Order is: psx-data-reader  ->  yfinance (.KA)  ->  your local CSV.
    "USE_PSX_READER": True,
    "USE_YFINANCE": True,
    "YF_SUFFIX": ".KA",
    # Optional per-ticker CSV fallback. Any CSV with a date column and a close
    # column works; column names are auto-detected.
    #   e.g. "LOCAL_CSV_PATHS": {"OGDC": "data/ogdc.csv"}
    "LOCAL_CSV_PATHS": {},

    # --- Fundamentals scraping ----------------------------------------------
    "TRY_WEB_FUNDAMENTALS": True,
    "WEB_FUNDAMENTALS_URL": "https://stockanalysis.com/quote/psx/{ticker}/",
    "WEB_TIMEOUT_SECONDS": 12,

    # --- Output --------------------------------------------------------------
    "OUTPUT_DIR": "outputs",
    "SAVE_CHARTS": True,
}

# -----------------------------------------------------------------------------
# FUNDAMENTALS + ANALYST TARGETS (you fill these in)
# -----------------------------------------------------------------------------
# PSX fundamentals are hard to scrape reliably, so paste them here yourself.
# Anything left as None is simply reported as "not supplied" — nothing is
# invented, and nothing here is verified by this tool. These are YOUR inputs.
#
# The web scrape (if it succeeds) fills a field first; your value below is used
# whenever the scrape has no number for that field. Every printed field is
# labelled with where it came from.
#
# Format, per ticker:
#   "TICK": {
#       "pe": 4.2,                  # price / earnings, trailing
#       "eps": 51.0,                # earnings per share, PKR
#       "dividend_yield_pct": 8.5,  # annual dividend / price, in PERCENT
#       "debt_to_equity": 0.15,     # total debt / shareholder equity
#       "roe_pct": 22.0,            # return on equity, in PERCENT
#       "target_avg": 260.0,        # 12-month analyst target, PKR
#       "target_high": 300.0,
#       "target_low": 210.0,
#       "source_note": "where you got these + as-of date",
#   },
FUNDAMENTALS = {
    "OGDC": {
        "pe": None,
        "eps": None,
        "dividend_yield_pct": None,
        "debt_to_equity": None,
        "roe_pct": None,
        "target_avg": None,
        "target_high": None,
        "target_low": None,
        "source_note": None,
    },
    "LUCK": {
        "pe": None,
        "eps": None,
        "dividend_yield_pct": None,
        "debt_to_equity": None,
        "roe_pct": None,
        "target_avg": None,
        "target_high": None,
        "target_low": None,
        "source_note": None,
    },
    "MEBL": {
        "pe": None,
        "eps": None,
        "dividend_yield_pct": None,
        "debt_to_equity": None,
        "roe_pct": None,
        "target_avg": None,
        "target_high": None,
        "target_low": None,
        "source_note": None,
    },
}

# Sector labels are only used to explain WHY the basket diversifies. Edit freely.
SECTORS = {
    "OGDC": "Oil & gas exploration",
    "LUCK": "Cement / industrials",
    "MEBL": "Islamic banking",
}

# =============================================================================
# ==========================  END CONFIG BLOCK  ===============================
# =============================================================================

import datetime as dt
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

TRADING_DAYS_PER_YEAR = 252  # PSX convention, used for every annualisation below


# -----------------------------------------------------------------------------
# Palette — one place, so every chart reads as the same system.
# -----------------------------------------------------------------------------
C = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",        # primary text
    "ink_soft": "#52514e",   # secondary text, reference lines
    "grid": "#e6e5e1",
    "model": "#2a78d6",      # entity 1: the model (median path, bars)
    "band_50": "#9ec5f4",    # inner (50%) band  - same hue, lighter step
    "band_80": "#cde2fb",    # outer (80%) band  - same hue, lightest step
    "external": "#eb6834",   # entity 2: external view (analyst targets, capital)
}


# -----------------------------------------------------------------------------
# Honest-output guard.
# -----------------------------------------------------------------------------
# Every line of the report goes through emit(). If a banned word ever reaches
# the page, the run fails loudly rather than quietly overpromising.
BANNED_SUBSTRINGS = ("guarantee", "guaranteed", "risk-free", "risk free",
                     "sure thing", "will reach", "will hit")


class Reporter:
    """Collects report lines and refuses to print overpromising language."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        low = text.lower()
        for bad in BANNED_SUBSTRINGS:
            if bad in low:
                raise AssertionError(
                    f"Output guard tripped: the phrase {bad!r} must never be "
                    f"printed by this tool. Offending line: {text!r}"
                )
        self.lines.append(text)
        print(text)

    def rule(self, char: str = "-", width: int = 78) -> None:
        self(char * width)

    def wrap(self, text: str, width: int = 78, indent: str = "") -> None:
        for line in textwrap.wrap(" ".join(text.split()), width=width,
                                  initial_indent=indent, subsequent_indent=indent):
            self(line)


emit = Reporter()


def fmt(x: Optional[float], nd: int = 2, suffix: str = "") -> str:
    """Format a number, or say plainly that it is missing. Never invents a value."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:,.{nd}f}{suffix}"


# =============================================================================
# 1. DATA LAYER — robust, fail-loud
# =============================================================================

class DataQualityError(Exception):
    """Raised when a price series exists but is not fit to model."""


@dataclass
class PriceData:
    ticker: str
    close: pd.Series          # float close prices, DatetimeIndex, ascending
    source: str               # which source actually produced this data


def _clean_and_validate(raw: pd.Series, ticker: str, source: str,
                        min_rows: int) -> pd.Series:
    """Turn whatever a source returned into a trustworthy close series, or raise.

    Fail-loud on purpose: it is far better to skip a ticker than to model
    fabricated or half-empty data.
    """
    s = pd.Series(raw).copy()

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")

    s = s[~s.index.isna()]
    s.index = s.index.tz_localize(None) if getattr(s.index, "tz", None) else s.index
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]                                    # a price <= 0 is bad data
    s = s[~s.index.duplicated(keep="last")].sort_index()

    if len(s) < min_rows:
        raise DataQualityError(
            f"{source} returned only {len(s)} usable rows for {ticker}; "
            f"at least {min_rows} are required."
        )

    # A completely flat series would make volatility zero and the whole model
    # meaningless — treat it as a data fault, not as a real market observation.
    if s.nunique() < 20:
        raise DataQualityError(
            f"{source} returned {s.nunique()} distinct prices for {ticker}; "
            f"that is a stale or broken feed, not a tradable series."
        )

    return s.astype(float)


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Find a column by name, case/space-insensitively."""
    norm = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for want in candidates:
        key = want.strip().lower().replace("_", " ")
        if key in norm:
            return norm[key]
    return None


def fetch_from_psx(ticker: str, years: int) -> pd.Series:
    """Primary source: psx-data-reader, straight from the exchange."""
    from psx import stocks  # imported lazily so a missing package is non-fatal

    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * years) + 10)
    df = stocks(ticker, start=start, end=end)

    if df is None or len(df) == 0:
        raise DataQualityError(f"psx-data-reader returned no rows for {ticker}.")

    col = _pick_column(df, ("Close", "close", "CLOSE", "Price"))
    if col is None:
        raise DataQualityError(
            f"psx-data-reader gave columns {list(df.columns)} for {ticker}; "
            f"no close column found."
        )
    return df[col]


def fetch_from_yfinance(ticker: str, years: int, suffix: str) -> pd.Series:
    """Fallback source: Yahoo Finance, which lists PSX names with a .KA suffix."""
    import yfinance as yf

    symbol = f"{ticker}{suffix}"
    hist = yf.Ticker(symbol).history(period=f"{years}y", auto_adjust=True)

    if hist is None or len(hist) == 0:
        raise DataQualityError(f"yfinance returned no rows for {symbol}.")

    col = _pick_column(hist, ("Close", "Adj Close"))
    if col is None:
        raise DataQualityError(f"yfinance gave no close column for {symbol}.")
    return hist[col]


def fetch_from_csv(path: str, ticker: str) -> pd.Series:
    """Final fallback: a local CSV you supply. Column names are auto-detected."""
    if not os.path.exists(path):
        raise DataQualityError(f"CSV path does not exist: {path}")

    df = pd.read_csv(path)
    date_col = _pick_column(df, ("Date", "Datetime", "Timestamp", "Time", "Day"))
    close_col = _pick_column(df, ("Close", "Adj Close", "Close Price", "Price",
                                  "Last", "Last Price", "Rate"))

    if date_col is None or close_col is None:
        raise DataQualityError(
            f"CSV {path} for {ticker} needs a date column and a close column; "
            f"found columns {list(df.columns)}."
        )

    out = pd.Series(df[close_col].to_numpy(),
                    index=pd.to_datetime(df[date_col], errors="coerce",
                                         format="mixed"))
    return out


def load_prices(ticker: str, cfg: dict) -> Optional[PriceData]:
    """Try every configured source in order. Return None (loudly) if all fail."""
    attempts: list[tuple[str, callable]] = []

    if cfg["USE_PSX_READER"]:
        attempts.append(("psx-data-reader (PSX official)",
                         lambda: fetch_from_psx(ticker, cfg["HISTORY_YEARS"])))
    if cfg["USE_YFINANCE"]:
        attempts.append((f"yfinance ({ticker}{cfg['YF_SUFFIX']})",
                         lambda: fetch_from_yfinance(ticker, cfg["HISTORY_YEARS"],
                                                     cfg["YF_SUFFIX"])))
    csv_path = cfg["LOCAL_CSV_PATHS"].get(ticker)
    if csv_path:
        attempts.append((f"local CSV ({csv_path})",
                         lambda: fetch_from_csv(csv_path, ticker)))

    failures: list[str] = []
    for source_name, getter in attempts:
        try:
            raw = getter()
            close = _clean_and_validate(raw, ticker, source_name, cfg["MIN_ROWS"])
        except Exception as exc:                       # noqa: BLE001 - report all
            failures.append(f"    - {source_name}: {type(exc).__name__}: {exc}")
            continue

        span = f"{close.index[0].date()} to {close.index[-1].date()}"
        print(f"  [OK]   {ticker}: {len(close)} rows from {source_name} ({span})")
        if failures:
            for line in failures:
                print(f"         (earlier attempt failed) {line.strip()}")
        return PriceData(ticker=ticker, close=close, source=source_name)

    print(f"  [SKIP] {ticker}: no usable data. Reasons:")
    if not attempts:
        print("    - no data sources are enabled for this ticker (all of "
              "USE_PSX_READER, USE_YFINANCE are off and no LOCAL_CSV_PATHS "
              "entry exists).")
    for line in failures:
        print(line)
    print(f"         {ticker} is EXCLUDED from this run. No data was invented "
          f"to fill the gap.")
    return None


# =============================================================================
# 2. PER-STOCK STATISTICAL PROFILE — everything measured from real history
# =============================================================================

@dataclass
class StockStats:
    ticker: str
    source: str
    n_rows: int
    last_date: pd.Timestamp
    last_close: float

    high_52w: float
    low_52w: float
    pct_below_52w_high: float     # heat / cheapness gauge: 0% = sitting at highs
    pct_above_52w_low: float

    ann_vol_window: float         # annualised vol over STAT_WINDOW_DAYS
    ann_vol_full: float           # annualised vol over the full sample
    daily_vol: float              # daily log-return std used by the simulation
    daily_drift: float            # mean daily LOG return (historical drift)

    ma50: Optional[float]
    ma200: Optional[float]
    cross_state: str              # "golden cross" / "death cross" / n/a

    rsi14: Optional[float]
    rsi_flag: str                 # overbought / oversold / neutral

    ret_3m: Optional[float]       # trailing 3-month total return (63 trading days)
    ret_12m: Optional[float]      # trailing 12-month total return (252 days)

    log_returns: pd.Series = field(repr=False, default=None)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    RSI compares the average size of up-days to the average size of down-days
    over the last `period` days, on a 0-100 scale. Above 70 is conventionally
    called "overbought", below 30 "oversold". It is a momentum descriptor of
    what already happened — not a forecast.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == exponential mean with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Edge cases, handled explicitly rather than left as NaN/inf:
    #   no down-days at all -> RSI 100 ; no up-days at all -> RSI 0
    #   perfectly flat window (neither) -> RSI 50, the neutral midpoint
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    # Keep the warm-up period as NaN so we never report an RSI we cannot compute.
    return rsi.mask(avg_gain.isna() | avg_loss.isna(), np.nan)


def profile_stock(pd_obj: PriceData, cfg: dict) -> StockStats:
    close = pd_obj.close
    ticker = pd_obj.ticker

    # Log returns: ln(P_t / P_t-1). Log returns add up over time, which is what
    # makes the geometric Brownian motion simulation below coherent.
    log_ret = np.log(close / close.shift(1)).dropna()

    win = min(cfg["STAT_WINDOW_DAYS"], len(log_ret))
    recent = log_ret.iloc[-win:]

    # Annualised volatility = daily std x sqrt(252). It is the typical SIZE of
    # daily moves, scaled to a year. It says nothing about direction.
    daily_vol = float(recent.std(ddof=1))
    ann_vol_window = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    ann_vol_full = float(log_ret.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)

    daily_drift = float(recent.mean())   # historical average daily log return

    # 52-week window = last 252 trading days (or everything, if shorter).
    look = min(TRADING_DAYS_PER_YEAR, len(close))
    window_52w = close.iloc[-look:]
    high_52w = float(window_52w.max())
    low_52w = float(window_52w.min())
    last_close = float(close.iloc[-1])

    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # Golden cross = 50-day average above the 200-day average (medium-term
    # strength). Death cross = the opposite. A state description, not a signal.
    if ma50 is not None and ma200 is not None:
        cross_state = "golden cross (50d above 200d)" if ma50 > ma200 else \
                      "death cross (50d below 200d)"
    else:
        cross_state = "n/a (not enough history for a 200-day average)"

    rsi_series = compute_rsi(close).dropna()
    rsi14 = float(rsi_series.iloc[-1]) if len(rsi_series) else None
    if rsi14 is None:
        rsi_flag = "n/a"
    elif rsi14 > 70:
        rsi_flag = "OVERBOUGHT (>70)"
    elif rsi14 < 30:
        rsi_flag = "OVERSOLD (<30)"
    else:
        rsi_flag = "neutral (30-70)"

    def trailing_return(days: int) -> Optional[float]:
        if len(close) <= days:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1.0)

    return StockStats(
        ticker=ticker,
        source=pd_obj.source,
        n_rows=len(close),
        last_date=close.index[-1],
        last_close=last_close,
        high_52w=high_52w,
        low_52w=low_52w,
        pct_below_52w_high=(last_close / high_52w - 1.0) * 100.0,
        pct_above_52w_low=(last_close / low_52w - 1.0) * 100.0,
        ann_vol_window=ann_vol_window,
        ann_vol_full=ann_vol_full,
        daily_vol=daily_vol,
        daily_drift=daily_drift,
        ma50=ma50,
        ma200=ma200,
        cross_state=cross_state,
        rsi14=rsi14,
        rsi_flag=rsi_flag,
        ret_3m=trailing_return(63),
        ret_12m=trailing_return(252),
        log_returns=log_ret,
    )


# =============================================================================
# 3. FUNDAMENTALS — best-effort web, then your config, then say "unavailable"
# =============================================================================

FUNDAMENTAL_FIELDS = ("pe", "eps", "dividend_yield_pct", "debt_to_equity",
                      "roe_pct", "target_avg", "target_high", "target_low")

# Labels as they appear on stockanalysis.com quote pages.
_WEB_LABEL_MAP = {
    "pe": ("PE Ratio", "P/E Ratio"),
    "eps": ("EPS (ttm)", "EPS"),
    "dividend_yield_pct": ("Dividend Yield", "Dividend (Yield)"),
    "debt_to_equity": ("Debt / Equity", "Debt/Equity"),
    "roe_pct": ("Return on Equity (ROE)", "ROE"),
    "target_avg": ("Price Target", "Analyst Target"),
}


def _parse_number(text: str) -> Optional[float]:
    """Pull a float out of a scraped cell like '8.42%' or '1,234.5' or 'n/a'."""
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(text).replace(",", ""))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_web_fundamentals(ticker: str, cfg: dict) -> tuple[dict, Optional[str]]:
    """Best-effort scrape. Returns (values, error). NEVER raises."""
    if not cfg["TRY_WEB_FUNDAMENTALS"]:
        return {}, "web lookup disabled in config"

    url = cfg["WEB_FUNDAMENTALS_URL"].format(ticker=ticker)
    try:
        import requests
        resp = requests.get(
            url,
            timeout=cfg["WEB_TIMEOUT_SECONDS"],
            headers={"User-Agent": "Mozilla/5.0 (compatible; psx-dashboard/1.0)"},
        )
        if resp.status_code != 200:
            return {}, f"HTTP {resp.status_code} from {url}"

        # Flatten the page to plain text pairs; the quote tables are simple
        # label/value rows, so a light regex is more robust here than a strict
        # DOM path that breaks whenever the site is redesigned.
        text = re.sub(r"<[^>]+>", "|", resp.text)
        text = re.sub(r"\|+", "|", text)

        found: dict[str, float] = {}
        for field_name, labels in _WEB_LABEL_MAP.items():
            for label in labels:
                m = re.search(re.escape(label) + r"\|([^|]{1,24})\|", text)
                if m:
                    val = _parse_number(m.group(1))
                    if val is not None:
                        found[field_name] = val
                        break
        if not found:
            return {}, f"page fetched but no recognisable fundamentals in {url}"
        return found, None

    except Exception as exc:                            # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"


@dataclass
class Fundamentals:
    ticker: str
    values: dict            # field -> float
    origin: dict            # field -> "web" | "config"
    note: Optional[str]
    web_error: Optional[str]

    @property
    def any_values(self) -> bool:
        return bool(self.values)

    def has_targets(self) -> bool:
        return self.values.get("target_avg") is not None or \
               self.values.get("target_high") is not None or \
               self.values.get("target_low") is not None


def gather_fundamentals(ticker: str, cfg: dict) -> Fundamentals:
    web_vals, web_err = fetch_web_fundamentals(ticker, cfg)
    cfg_vals = FUNDAMENTALS.get(ticker, {}) or {}

    values: dict[str, float] = {}
    origin: dict[str, str] = {}

    for f in FUNDAMENTAL_FIELDS:
        # Web first (per spec); your config fills anything the web didn't give.
        if web_vals.get(f) is not None:
            values[f], origin[f] = float(web_vals[f]), "web"
        elif cfg_vals.get(f) is not None:
            values[f], origin[f] = float(cfg_vals[f]), "your config"

    return Fundamentals(ticker=ticker, values=values, origin=origin,
                        note=cfg_vals.get("source_note"), web_error=web_err)


# =============================================================================
# 4. MONTE CARLO RISK CONE (per stock)
# =============================================================================

@dataclass
class ConeResult:
    label: str
    paths: np.ndarray          # (n_paths, n_days) simulated prices
    terminal: np.ndarray       # (n_paths,) price on the final day
    pct: dict                  # {10: p, 25: p, 50: p, 75: p, 90: p}
    prob_above_start: float
    daily_drift_used: float


def simulate_gbm(s0: float, daily_drift: float, daily_vol: float,
                 n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Geometric Brownian motion, simulated day by day in log space.

    Each day the log price moves by:  drift + vol * (a random normal shock)
    Prices are the exponential of the cumulative sum, so they can never go
    negative and compounding is handled correctly.

    This is a model of RANDOMNESS ONLY. It assumes tomorrow's move is drawn from
    the same distribution as the last year's moves, independently each day. Real
    markets have fat tails, volatility clustering, and events. Treat the cone as
    a rough width, not a law.
    """
    shocks = rng.standard_normal((n_paths, n_days))
    log_steps = daily_drift + daily_vol * shocks
    return s0 * np.exp(np.cumsum(log_steps, axis=1))


def run_cone(s0: float, daily_drift: float, daily_vol: float, label: str,
             cfg: dict, rng: np.random.Generator) -> ConeResult:
    paths = simulate_gbm(s0, daily_drift, daily_vol,
                         cfg["HORIZON_TRADING_DAYS"], cfg["N_PATHS"], rng)
    terminal = paths[:, -1]
    pct = {q: float(np.percentile(terminal, q)) for q in (10, 25, 50, 75, 90)}
    return ConeResult(
        label=label,
        paths=paths,
        terminal=terminal,
        pct=pct,
        prob_above_start=float(np.mean(terminal > s0)),
        daily_drift_used=daily_drift,
    )


def run_both_scenarios(stats: StockStats, cfg: dict,
                       rng: np.random.Generator) -> dict[str, ConeResult]:
    """Scenario A uses history's drift; Scenario B assumes no view at all.

    Zero drift here means zero drift IN LOG SPACE, so the median simulated price
    on the final day equals today's price. That is the honest "I have no opinion
    on direction" baseline, and it is the one used for the portfolio model and
    the fan charts.
    """
    return {
        "historical": run_cone(stats.last_close, stats.daily_drift, stats.daily_vol,
                               "Scenario A: historical drift", cfg, rng),
        "zero": run_cone(stats.last_close, 0.0, stats.daily_vol,
                         "Scenario B: zero drift (random walk)", cfg, rng),
    }


# =============================================================================
# 5. PORTFOLIO VIEW — correlated multivariate Monte Carlo
# =============================================================================

@dataclass
class PortfolioResult:
    tickers: list[str]
    shares: dict               # ticker -> fractional share count
    start_value_pkr: float
    start_value_usd: float
    corr: pd.DataFrame
    avg_pairwise_corr: float
    terminal: np.ndarray       # (n_paths,) total basket value in PKR
    pct: dict
    prob_up: float
    avg_max_drawdown: float    # average worst peak-to-trough dip along a path
    band80_width_pct: float    # portfolio 80% band, as % of starting value
    avg_single_band80_pct: float
    diversification_benefit_pct: float
    values: np.ndarray = field(repr=False, default=None)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor of the covariance matrix, with a PSD repair fallback.

    The Cholesky factor L satisfies L @ L.T = covariance. Multiplying independent
    random shocks by L turns them into shocks that move together exactly as much
    as these stocks historically did — which is the whole point of modelling the
    basket rather than three separate stocks.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Numerical noise can leave the sample covariance a hair non-PSD.
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-14, None)
        repaired = vecs @ np.diag(vals) @ vecs.T
        return np.linalg.cholesky(repaired)


def run_portfolio(profiles: list[StockStats], cfg: dict,
                  rng: np.random.Generator,
                  single_cones: dict[str, ConeResult]) -> Optional[PortfolioResult]:
    if len(profiles) < 2:
        return None

    tickers = [p.ticker for p in profiles]

    # Align on dates all names traded — correlation is only meaningful on
    # matched days.
    ret_df = pd.concat({p.ticker: p.log_returns for p in profiles}, axis=1).dropna()
    win = min(cfg["STAT_WINDOW_DAYS"], len(ret_df))
    ret_df = ret_df.iloc[-win:]

    if len(ret_df) < 60:
        print("  [WARN] fewer than 60 overlapping trading days across tickers; "
              "the correlation estimate is thin.")

    corr = ret_df.corr()
    cov = ret_df.cov().to_numpy()          # daily log-return covariance
    L = _safe_cholesky(cov)

    n_assets = len(tickers)
    off_diag = corr.to_numpy()[np.triu_indices(n_assets, k=1)]
    avg_pairwise_corr = float(np.mean(off_diag))

    # Equal capital per name, converted to PKR. Fractional shares are used so the
    # basket is exactly equal-weighted for modelling purposes.
    capital_pkr = cfg["CAPITAL_USD_PER_STOCK"] * cfg["USD_PKR"]
    shares = {p.ticker: capital_pkr / p.last_close for p in profiles}
    s0 = np.array([p.last_close for p in profiles])
    share_vec = np.array([shares[t] for t in tickers])
    start_value_pkr = float(np.sum(share_vec * s0))

    n_days, n_paths = cfg["HORIZON_TRADING_DAYS"], cfg["N_PATHS"]

    # Independent shocks -> correlated shocks via the Cholesky factor.
    z = rng.standard_normal((n_paths, n_days, n_assets))
    correlated = z @ L.T                      # cov(correlated) == cov
    log_paths = np.cumsum(correlated, axis=1)  # drift = 0 by construction
    prices = s0 * np.exp(log_paths)            # (paths, days, assets)

    values = prices @ share_vec                # (paths, days) total basket value
    # Prepend day 0 so drawdown is measured from the real starting value.
    values = np.concatenate([np.full((n_paths, 1), start_value_pkr), values], axis=1)

    terminal = values[:, -1]
    pct = {q: float(np.percentile(terminal, q)) for q in (10, 25, 50, 75, 90)}

    # Max drawdown along each path: the worst peak-to-trough dip experienced
    # on the way, not just where it ended. Averaged over all paths.
    running_max = np.maximum.accumulate(values, axis=1)
    drawdowns = values / running_max - 1.0
    avg_max_drawdown = float(np.mean(drawdowns.min(axis=1)))

    band80_width_pct = (pct[90] - pct[10]) / start_value_pkr * 100.0
    single_widths = [
        (single_cones[t].pct[90] - single_cones[t].pct[10]) / p.last_close * 100.0
        for t, p in zip(tickers, profiles)
    ]
    avg_single = float(np.mean(single_widths))

    return PortfolioResult(
        tickers=tickers,
        shares=shares,
        start_value_pkr=start_value_pkr,
        start_value_usd=start_value_pkr / cfg["USD_PKR"],
        corr=corr,
        avg_pairwise_corr=avg_pairwise_corr,
        terminal=terminal,
        pct=pct,
        prob_up=float(np.mean(terminal > start_value_pkr)),
        avg_max_drawdown=avg_max_drawdown,
        band80_width_pct=band80_width_pct,
        avg_single_band80_pct=avg_single,
        diversification_benefit_pct=(1.0 - band80_width_pct / avg_single) * 100.0,
        values=values,
    )


# =============================================================================
# 6. VISUALS
# =============================================================================

def _style_axes(ax) -> None:
    """Recessive grid and axes, so the data is the loudest thing on the chart."""
    ax.set_facecolor(C["surface"])
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["grid"])
    ax.tick_params(colors=C["ink_soft"], labelsize=9, length=0)


def plot_fan_chart(stats: StockStats, cone: ConeResult, fund: Fundamentals,
                   cfg: dict, out_dir: str) -> str:
    """Fan chart: the cone of simulated prices, with today's price as anchor."""
    paths = cone.paths
    n_days = paths.shape[1]
    x = np.arange(0, n_days + 1)   # day 0 = today, so the cone opens from a point

    def env(q):
        return np.concatenate([[stats.last_close], np.percentile(paths, q, axis=0)])

    p10, p25, p50, p75, p90 = (env(q) for q in (10, 25, 50, 75, 90))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    _style_axes(ax)

    ax.fill_between(x, p10, p90, color=C["band_80"], linewidth=0,
                    label="80% of outcomes (P10-P90)")
    ax.fill_between(x, p25, p75, color=C["band_50"], linewidth=0,
                    label="50% of outcomes (P25-P75)")
    # Today's price goes down first so the median line stays readable on top of
    # it — under zero drift the two sit almost exactly on each other.
    ax.axhline(stats.last_close, color=C["ink_soft"], linewidth=1.2,
               linestyle="--", zorder=2,
               label=f"Today's price {stats.last_close:,.2f}")
    ax.plot(x, p50, color=C["model"], linewidth=2.0, zorder=3,
            label="Median path (NOT a target)")

    # Analyst targets are an EXTERNAL view, so they get their own entity colour
    # and sit past the horizon — they are 12-month views, not 60-day.
    if fund.has_targets():
        tx = n_days + 5
        lo = fund.values.get("target_low")
        hi = fund.values.get("target_high")
        avg = fund.values.get("target_avg")
        if lo is not None and hi is not None:
            ax.plot([tx, tx], [lo, hi], color=C["external"], linewidth=2.0,
                    solid_capstyle="round")
        pts = [v for v in (lo, avg, hi) if v is not None]
        ax.plot([tx] * len(pts), pts, "o", color=C["external"], markersize=8,
                markeredgecolor=C["surface"], markeredgewidth=1.5,
                label="Analyst 12m target (external view)")
        if avg is not None:
            ax.annotate(f"target avg\n{avg:,.0f}", (tx, avg),
                        xytext=(8, 0), textcoords="offset points",
                        color=C["ink_soft"], fontsize=8, va="center")

    # Selective direct labels at the right edge — the three numbers worth reading.
    # P50 is nudged down because under zero drift it lands on today's price line.
    for val, name, dy in ((p90[-1], "P90", 0), (p50[-1], "P50", -11),
                          (p10[-1], "P10", 0)):
        ax.annotate(f"{name} {val:,.0f}", (n_days, val), xytext=(6, dy),
                    textcoords="offset points", fontsize=8.5,
                    color=C["ink_soft"], va="center",
                    bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.0))

    ax.set_xlim(0, n_days + (13 if fund.has_targets() else 5))
    ax.set_xlabel("Trading days ahead", color=C["ink_soft"], fontsize=9.5)
    ax.set_ylabel("Price (PKR)", color=C["ink_soft"], fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(
        f"{stats.ticker} — range of simulated prices over "
        f"{cfg['HORIZON_TRADING_DAYS']} trading days",
        color=C["ink"], fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.015,
            f"{cfg['N_PATHS']:,} random-walk paths, zero drift, measured "
            f"annualised volatility {stats.ann_vol_window*100:.1f}%. "
            f"Randomness only — no earnings or news modelled.",
            transform=ax.transAxes, color=C["ink_soft"], fontsize=8.5)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(C["ink_soft"])

    path = os.path.join(out_dir, f"fan_{stats.ticker}.png")
    fig.tight_layout()
    fig.savefig(path, facecolor=C["surface"])
    plt.close(fig)
    return path


def plot_portfolio_histogram(port: PortfolioResult, cfg: dict,
                             out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    _style_axes(ax)

    ax.hist(port.terminal, bins=60, color=C["band_50"], edgecolor=C["surface"],
            linewidth=0.6, label=f"{cfg['N_PATHS']:,} simulated outcomes")

    # Headroom so the percentile labels never sit on top of the tallest bars.
    ymax = ax.get_ylim()[1] * 1.20
    ax.set_ylim(0, ymax)

    for q, style, frac in ((10, ":", 0.97), (50, "-", 0.86), (90, ":", 0.97)):
        v = port.pct[q]
        ax.axvline(v, color=C["model"], linestyle=style, linewidth=2.0, zorder=3)
        ax.annotate(f"P{q}\n{v:,.0f}", (v, ymax * frac), fontsize=8.5,
                    color=C["ink_soft"], ha="center", va="top", zorder=4,
                    bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.5))

    ax.axvline(port.start_value_pkr, color=C["external"], linewidth=2.0, zorder=3,
               label=f"Starting capital {port.start_value_pkr:,.0f} PKR")
    ax.annotate("start", (port.start_value_pkr, ymax * 0.55), fontsize=8.5,
                color=C["external"], ha="center", zorder=4,
                bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.5))

    ax.set_xlabel(f"Total basket value in PKR after "
                  f"{cfg['HORIZON_TRADING_DAYS']} trading days",
                  color=C["ink_soft"], fontsize=9.5)
    ax.set_ylabel("Number of simulated paths", color=C["ink_soft"], fontsize=9.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Portfolio — distribution of total basket value",
                 color=C["ink"], fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.015,
            f"Equal capital in {', '.join(port.tickers)}; correlated shocks "
            f"(avg pairwise correlation {port.avg_pairwise_corr:.2f}), zero drift.",
            transform=ax.transAxes, color=C["ink_soft"], fontsize=8.5)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(C["ink_soft"])

    path = os.path.join(out_dir, "portfolio_terminal_values.png")
    fig.tight_layout()
    fig.savefig(path, facecolor=C["surface"])
    plt.close(fig)
    return path


# =============================================================================
# 7. OUTPUT FRAMING — FACT vs UNCERTAINTY vs EXTERNAL VIEW
# =============================================================================

def report_stock(stats: StockStats, fund: Fundamentals,
                 cones: dict[str, ConeResult], cfg: dict) -> None:
    emit()
    emit.rule("=")
    emit(f" {stats.ticker} — {SECTORS.get(stats.ticker, 'sector not labelled')}")
    emit.rule("=")

    # ---- PART 1: FACT --------------------------------------------------------
    emit()
    emit("  [1] FACT — measured from real price history. Not an opinion.")
    emit(f"      Data source          : {stats.source}")
    emit(f"      History              : {stats.n_rows} trading days, "
         f"last close dated {stats.last_date.date()}")
    emit(f"      Latest close         : {fmt(stats.last_close)} PKR")
    emit(f"      52-week high / low   : {fmt(stats.high_52w)} / {fmt(stats.low_52w)}")
    emit(f"      vs 52-week high      : {fmt(stats.pct_below_52w_high, 1, '%')}"
         f"   (0% = sitting at its highs; more negative = further off them)")
    emit(f"      vs 52-week low       : {fmt(stats.pct_above_52w_low, 1, '%')}")
    emit(f"      Annualised volatility: {fmt(stats.ann_vol_window*100, 1, '%')}"
         f"   (last {cfg['STAT_WINDOW_DAYS']} days; full sample "
         f"{fmt(stats.ann_vol_full*100, 1, '%')})")
    emit(f"                             = typical size of daily moves, scaled to "
         f"a year. Size only, no direction.")
    emit(f"      50d / 200d average   : {fmt(stats.ma50)} / {fmt(stats.ma200)}")
    emit(f"      Trend state          : {stats.cross_state}")
    emit(f"      14-day RSI           : {fmt(stats.rsi14, 1)}  [{stats.rsi_flag}]")
    emit(f"                             = momentum of what ALREADY happened, "
         f"0-100. Not a forecast.")
    emit(f"      Trailing 3m return   : {fmt((stats.ret_3m or 0)*100, 1, '%') if stats.ret_3m is not None else 'n/a'}")
    emit(f"      Trailing 12m return  : {fmt((stats.ret_12m or 0)*100, 1, '%') if stats.ret_12m is not None else 'n/a'}")

    emit()
    if fund.any_values:
        emit("      Fundamentals (each line labelled with where the number came from):")
        pretty = {
            "pe": ("P/E ratio", 2, ""),
            "eps": ("EPS", 2, " PKR"),
            "dividend_yield_pct": ("Dividend yield", 2, "%"),
            "debt_to_equity": ("Debt / equity", 2, ""),
            "roe_pct": ("Return on equity", 2, "%"),
        }
        for f, (label, nd, sfx) in pretty.items():
            if f in fund.values:
                emit(f"        {label:<22}: {fmt(fund.values[f], nd, sfx)}"
                     f"   (source: {fund.origin[f]})")
            else:
                emit(f"        {label:<22}: not supplied")
        if fund.note:
            emit(f"        Your note             : {fund.note}")
    else:
        emit("      Fundamentals unavailable — the web lookup returned nothing "
             "usable and")
        emit("      no values were supplied in the FUNDAMENTALS config block. "
             "Continuing without them.")
    if fund.web_error:
        emit(f"        (web lookup: {fund.web_error})")

    # ---- PART 2: UNCERTAINTY -------------------------------------------------
    emit()
    emit(f"  [2] UNCERTAINTY — {cfg['N_PATHS']:,} simulated paths over "
         f"{cfg['HORIZON_TRADING_DAYS']} trading days (~60 calendar days).")
    emit("      These are RANGES. No single number here is a prediction.")

    for key in ("zero", "historical"):
        c = cones[key]
        emit()
        emit(f"      {c.label}"
             f"   [daily log drift used: {c.daily_drift_used:+.5f}]")
        emit(f"        P10 {fmt(c.pct[10])} | P25 {fmt(c.pct[25])} | "
             f"P50 {fmt(c.pct[50])} | P75 {fmt(c.pct[75])} | P90 {fmt(c.pct[90])}")
        emit(f"        Probability of finishing above today's "
             f"{fmt(stats.last_close)}: {c.prob_above_start*100:.1f}%")
        width = (c.pct[90] - c.pct[10]) / stats.last_close * 100.0
        emit(f"        Width of the 80% band: {width:.1f}% of today's price.")

    emit()
    emit.wrap(
        "CAUTION on Scenario A: historical drift assumes the last year's average "
        f"daily move simply continues. For a stock {abs(stats.pct_below_52w_high):.1f}% "
        f"{'below' if stats.pct_below_52w_high < 0 else 'from'} its 52-week high, "
        "that over-extrapolates a past run into the future. Scenario B (zero drift) "
        "is the honest default and is what the charts and the portfolio model use.",
        width=74, indent="        ")

    # ---- PART 3: EXTERNAL VIEW ----------------------------------------------
    emit()
    emit("  [3] EXTERNAL VIEW — other people's opinions. Not measured, not "
         "modelled here.")
    if fund.has_targets():
        lo = fund.values.get("target_low")
        avg = fund.values.get("target_avg")
        hi = fund.values.get("target_high")
        emit(f"      Analyst 12-month target — low {fmt(lo)} | avg {fmt(avg)} | "
             f"high {fmt(hi)} PKR")
        if avg is not None:
            gap = (avg / stats.last_close - 1.0) * 100.0
            emit(f"      Average target sits {gap:+.1f}% from today's price "
                 f"(source: {fund.origin.get('target_avg', 'n/a')}).")
        emit("      Note the mismatch of horizons: targets are 12-month views; "
             "the cone above is ~60 days.")
        emit("      Analyst targets are forecasts made by people, and they are "
             "revised and often missed.")
    else:
        emit("      No analyst targets supplied for this ticker, and none were "
             "scraped.")
        emit("      Add them under FUNDAMENTALS in the config block to see them "
             "on the chart.")


def report_portfolio(port: PortfolioResult, profiles: list[StockStats],
                     cfg: dict) -> None:
    emit()
    emit.rule("=")
    emit(" PORTFOLIO — the three names held together")
    emit.rule("=")

    emit()
    emit("  [1] FACT — the basket as constructed")
    emit(f"      Capital per stock : ${cfg['CAPITAL_USD_PER_STOCK']:,.2f} at "
         f"{cfg['USD_PKR']:,.2f} PKR/USD = "
         f"{cfg['CAPITAL_USD_PER_STOCK']*cfg['USD_PKR']:,.2f} PKR each")
    emit(f"      Total starting    : {port.start_value_pkr:,.2f} PKR "
         f"(${port.start_value_usd:,.2f})")
    emit("      Fractional share counts (a modelling device — real lots are whole "
         "shares):")
    for p in profiles:
        emit(f"        {p.ticker:<6} {port.shares[p.ticker]:>9.4f} shares "
             f"@ {p.last_close:,.2f} PKR   [{SECTORS.get(p.ticker, 'n/a')}]")

    emit()
    emit("      Correlation of daily returns (1.00 = moves in lockstep, "
         "0.00 = unrelated):")
    header = "            " + "".join(f"{t:>9}" for t in port.tickers)
    emit(header)
    for t in port.tickers:
        row = f"        {t:<4}" + "".join(
            f"{port.corr.loc[t, o]:>9.2f}" for o in port.tickers)
        emit(row)
    emit(f"      Average pairwise correlation: {port.avg_pairwise_corr:.2f}")

    emit()
    emit(f"  [2] UNCERTAINTY — {cfg['N_PATHS']:,} correlated paths, zero drift, "
         f"{cfg['HORIZON_TRADING_DAYS']} trading days")
    emit("      Shocks are drawn together using the Cholesky factor of the real "
         "covariance matrix,")
    emit("      so the three names move together in the model exactly as much as "
         "they did in reality.")
    emit()
    emit("      Total basket value at the horizon (PKR):")
    emit(f"        P10 {port.pct[10]:,.2f} | P50 {port.pct[50]:,.2f} | "
         f"P90 {port.pct[90]:,.2f}")
    emit(f"      Same in USD at {cfg['USD_PKR']:,.2f}:")
    emit(f"        P10 ${port.pct[10]/cfg['USD_PKR']:,.2f} | "
         f"P50 ${port.pct[50]/cfg['USD_PKR']:,.2f} | "
         f"P90 ${port.pct[90]/cfg['USD_PKR']:,.2f}")
    emit(f"      Probability the basket finishes above "
         f"{port.start_value_pkr:,.2f} PKR: {port.prob_up*100:.1f}%")
    emit("        (Slightly above 50% is expected even with zero drift: prices "
         "compound, so a")
    emit("         basket of lognormal outcomes leans a little to the upside "
         "around its median.")
    emit("         That is arithmetic, not a view on direction.)")
    emit(f"      Average worst drawdown along a path: "
         f"{port.avg_max_drawdown*100:.1f}%")
    emit("        = on a typical path, the basket dips this far below its own "
         "peak at some point")
    emit("          before the horizon. The ride is bumpier than the endpoint "
         "suggests.")

    emit()
    emit("  [3] DIVERSIFICATION — why the basket's cone is tighter than the "
         "stocks' cones")
    emit(f"      Average 80% band of the individual stocks: "
         f"{port.avg_single_band80_pct:.1f}% of starting value")
    emit(f"      80% band of the combined basket          : "
         f"{port.band80_width_pct:.1f}% of starting value")
    emit(f"      Reduction in band width                  : "
         f"{port.diversification_benefit_pct:.1f}%")
    emit()
    sectors = ", ".join(f"{t} ({SECTORS.get(t, 'sector not labelled')})"
                        for t in port.tickers)
    distinct = len({SECTORS.get(t) for t in port.tickers if SECTORS.get(t)})
    spread = ("across different sectors" if distinct > 1
              else "within the same sector, which limits how much they can offset "
                   "each other")
    emit.wrap(
        f"These are {len(port.tickers)} different businesses — {sectors} — spread "
        f"{spread}. Different sectors answer to different drivers, so they do not "
        f"all move on the same news. Their measured average pairwise correlation is "
        f"{port.avg_pairwise_corr:.2f}, which is below 1.00, so their bad days do not "
        f"all land on the same day. When one name is having a poor week another is "
        f"often flat or up, and the swings partially cancel inside the basket. That "
        f"cancellation is the entire reason the combined cone is "
        f"{port.diversification_benefit_pct:.1f}% narrower than the average single-name "
        f"cone. Diversification narrows the range of outcomes; it does not raise the "
        f"middle of it, and it does not remove the risk of everything falling at once "
        f"in a market-wide selloff.",
        width=74, indent="      ")


def closing_paragraph(profiles: list[StockStats], port: Optional[PortfolioResult],
                      cfg: dict) -> None:
    """Auto-generated plain-English framing, built from this run's real numbers."""
    emit()
    emit.rule("=")
    emit(" HOW TO READ EVERYTHING ABOVE")
    emit.rule("=")
    emit()

    widest = max(profiles, key=lambda p: p.ann_vol_window)
    tightest = min(profiles, key=lambda p: p.ann_vol_window)

    para1 = (
        f"Every price number in the uncertainty sections is a percentile, not a "
        f"forecast. The median (P50) is the CENTRE of a wide range of simulated "
        f"outcomes, not a target and not an expectation of where any of these "
        f"stocks will trade. Half of the simulated paths finished below it. Read "
        f"the P10 and P90 together with it, always: the distance between them IS "
        f"the answer this tool produces. A wider cone means more uncertainty, "
        f"nothing more and nothing less. In this run {widest.ticker} has the widest "
        f"cone (annualised volatility "
        f"{widest.ann_vol_window*100:.1f}%) and {tightest.ticker} the tightest "
        f"({tightest.ann_vol_window*100:.1f}%), which tells you how much these two "
        f"names have historically moved around — it does not tell you which one "
        f"will do better."
    )

    para2 = (
        f"This tool models price randomness and nothing else. It has no knowledge "
        f"of earnings, contracts, dividends yet to be declared, oil prices, "
        f"interest rate decisions, currency moves, politics, regulation, or any "
        f"company announcement. It assumes each day's move is drawn independently "
        f"from a distribution shaped by the last "
        f"{cfg['STAT_WINDOW_DAYS']} trading days, which real markets violate: "
        f"real returns have fatter tails, calm and violent periods cluster "
        f"together, and single events can move a stock further in one session than "
        f"this model expects in a month. Treat the cones as a rough sense of scale "
        f"for normal conditions, and assume reality can land outside them."
    )

    if port is not None:
        para3 = (
            f"The portfolio section is the part worth acting on. Holding "
            f"{', '.join(port.tickers)} together produced an 80% band "
            f"{port.diversification_benefit_pct:.1f}% narrower than the average "
            f"single name, because their measured correlation is "
            f"{port.avg_pairwise_corr:.2f} rather than 1.00. Meanwhile the average "
            f"worst dip along a path was {abs(port.avg_max_drawdown)*100:.1f}%, so "
            f"expect the value to travel meaningfully below its own peak on the way "
            f"to wherever it ends up. Nothing here is advice, nothing here is "
            f"certain, and any single number quoted without its percentile range "
            f"attached has been quoted wrongly."
        )
    else:
        para3 = (
            "Only one ticker survived data validation, so no portfolio view was "
            "produced. Nothing here is advice, and any single number quoted without "
            "its percentile range attached has been quoted wrongly."
        )

    for para in (para1, para2, para3):
        emit.wrap(para, width=78)
        emit()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    cfg = CONFIG
    rng = np.random.default_rng(cfg["RANDOM_SEED"])

    out_dir = cfg["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(" PSX PORTFOLIO RISK & CONTEXT DASHBOARD")
    print(f" Run date: {dt.date.today()}   |   Tickers: {', '.join(cfg['TICKERS'])}")
    print(" This tool reports ranges and measured facts. It never predicts a "
          "single price.")
    print("=" * 78)

    # ---- 1. Data -------------------------------------------------------------
    print("\n[1/5] Loading price history "
          f"(need >= {cfg['MIN_ROWS']} rows per ticker)...")
    loaded = [pdata for t in cfg["TICKERS"]
              if (pdata := load_prices(t, cfg)) is not None]

    if not loaded:
        print("\nFATAL: no ticker produced a usable price series. Nothing was "
              "modelled.")
        print("Check your network, or point LOCAL_CSV_PATHS at CSV files you "
              "already have.")
        return 1

    skipped = [t for t in cfg["TICKERS"] if t not in [p.ticker for p in loaded]]
    if skipped:
        print(f"\n  Skipped (no usable data): {', '.join(skipped)}")

    # ---- 2. Profiles ---------------------------------------------------------
    print("\n[2/5] Measuring per-stock statistics...")
    profiles = [profile_stock(p, cfg) for p in loaded]

    # ---- 3. Fundamentals -----------------------------------------------------
    print("[3/5] Gathering fundamentals (web best-effort, then your config)...")
    funds = {p.ticker: gather_fundamentals(p.ticker, cfg) for p in profiles}

    # ---- 4. Simulations ------------------------------------------------------
    print("[4/5] Running Monte Carlo simulations...")
    all_cones = {p.ticker: run_both_scenarios(p, cfg, rng) for p in profiles}
    zero_cones = {t: c["zero"] for t, c in all_cones.items()}
    port = run_portfolio(profiles, cfg, rng, zero_cones)

    # ---- 5. Charts -----------------------------------------------------------
    chart_paths: list[str] = []
    if cfg["SAVE_CHARTS"]:
        print("[5/5] Writing charts...")
        for p in profiles:
            chart_paths.append(
                plot_fan_chart(p, zero_cones[p.ticker], funds[p.ticker], cfg, out_dir))
        if port is not None:
            chart_paths.append(plot_portfolio_histogram(port, cfg, out_dir))
        for cp in chart_paths:
            print(f"  saved {cp}")

    # ---- Report --------------------------------------------------------------
    for p in profiles:
        report_stock(p, funds[p.ticker], all_cones[p.ticker], cfg)

    if port is not None:
        report_portfolio(port, profiles, cfg)
    else:
        emit()
        emit("  Portfolio view skipped: it needs at least 2 tickers with usable "
             "data.")

    closing_paragraph(profiles, port, cfg)

    if chart_paths:
        emit(" Charts written to:")
        for cp in chart_paths:
            emit(f"   - {cp}")
        emit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
