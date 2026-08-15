#!/usr/bin/env python3
"""
Browser front end for the PSX portfolio risk-and-context dashboard.

This file contains NO modelling. Every number and every chart on the page comes
from psx_dashboard.run_analysis(), the same function the command line uses, so
the two can never disagree with each other.

It also keeps the same promise the CLI makes: no forward-looking number is shown
on its own. Percentiles are always rendered as a full P10-P90 row, and every
string that reaches the page is passed through psx_dashboard.assert_honest(),
which raises if overpromising language ever appears.

Run:  streamlit run app.py
"""

import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import psx_dashboard as dash

st.set_page_config(page_title="PSX risk & context dashboard",
                   page_icon="📊", layout="wide")


# -----------------------------------------------------------------------------
# Honest-output guard, applied to everything this page writes.
# -----------------------------------------------------------------------------

def md(text: str, **kwargs) -> None:
    """Write markdown, but only after the honesty guard has cleared it."""
    st.markdown(dash.assert_honest(text), **kwargs)


def caption(text: str) -> None:
    st.caption(dash.assert_honest(text))


def table(records: list[dict]) -> None:
    """Render rows as a table, guarding every cell that is text."""
    for record in records:
        for value in record.values():
            if isinstance(value, str):
                dash.assert_honest(value)
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Sidebar — the controls
# -----------------------------------------------------------------------------

def parse_tickers(raw: str) -> list[str]:
    """Split a free-text ticker list on commas or whitespace."""
    return [t.strip().upper() for t in re.split(r"[,\s]+", raw) if t.strip()]


SCENARIO_LABELS = {
    "zero": "Zero drift — no view on direction (default)",
    "historical": "Historical drift — assumes the past year's average continues",
}

with st.sidebar:
    st.header("Controls")

    tickers_raw = st.text_input(
        "Tickers", value=", ".join(dash.CONFIG["TICKERS"]),
        help="PSX symbols, separated by commas or spaces. Add, remove or "
             "replace names freely.")

    horizon_days = st.slider(
        "Horizon (calendar days)", min_value=15, max_value=365,
        value=60, step=5,
        help="How far ahead to simulate. Converted to trading days internally.")
    caption(f"= {dash.calendar_days_to_trading_days(horizon_days)} trading days "
            f"at {dash.TRADING_DAYS_PER_WEEK} trading days per week.")

    capital_per_stock = st.number_input(
        "Capital per stock (USD)", min_value=1.0,
        value=float(dash.CONFIG["CAPITAL_USD_PER_STOCK"]), step=5.0,
        help="Equal capital is placed into each ticker that survives data "
             "validation.")

    usd_pkr = st.number_input(
        "USD / PKR rate", min_value=1.0, value=float(dash.CONFIG["USD_PKR"]),
        step=1.0, help="Used only to convert your capital into PKR and back.")

    drift_scenario = st.radio(
        "Drift scenario", options=list(dash.DRIFT_SCENARIOS),
        format_func=lambda s: SCENARIO_LABELS[s], index=0,
        help="Both scenarios are always computed and reported. This chooses "
             "which one the charts and the portfolio model are built on.")

    with st.expander("Advanced"):
        try_web = st.checkbox(
            "Try the web fundamentals lookup",
            value=bool(dash.CONFIG["TRY_WEB_FUNDAMENTALS"]),
            help="Best-effort scrape of P/E, EPS and similar. Turn it off when "
                 "you are offline to skip the wait.")

    run_clicked = st.button("Run analysis", type="primary", width="stretch")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

if run_clicked:
    tickers = parse_tickers(tickers_raw)
    if not tickers:
        st.error("Enter at least one ticker.")
    else:
        # Figures from an earlier run are no longer on screen; close them so
        # matplotlib does not accumulate them across reruns.
        previous = st.session_state.get("result")
        if previous is not None:
            for figure in previous.figures.values():
                plt.close(figure)

        with st.spinner(f"Loading price history for {', '.join(tickers)}, "
                        f"measuring statistics and running "
                        f"{dash.CONFIG['N_PATHS']:,} simulated paths..."):
            try:
                cfg = dash.build_config(
                    tickers=tickers, horizon_days=horizon_days,
                    capital_per_stock=capital_per_stock, usd_pkr=usd_pkr,
                    drift_scenario=drift_scenario,
                    overrides={"TRY_WEB_FUNDAMENTALS": try_web})
                st.session_state["result"] = dash.run_analysis(
                    cfg=cfg, with_figures=True)
            except Exception as exc:                        # noqa: BLE001
                st.session_state["result"] = None
                st.exception(exc)


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------

st.title("PSX portfolio risk & context dashboard")
md("This tool reports **measured facts** and **ranges**. It never presents a "
   "single future price as a prediction, and every forward-looking figure below "
   "is a percentile shown with the rest of its range.")

result = st.session_state.get("result")

if result is None:
    st.info("Set your tickers and horizon in the sidebar, then press "
            "**Run analysis**.")
    st.stop()

cfg = result.cfg
horizon = cfg["HORIZON_TRADING_DAYS"]

if not result.profiles:
    st.error("No ticker produced a usable price series, so nothing was "
             "modelled. No data was invented to fill the gap.")
    st.code("\n".join(result.log) or "(no diagnostics)", language="text")
    st.stop()

if result.skipped:
    st.warning(f"Skipped, no usable data: {', '.join(result.skipped)}. "
               f"These are excluded from everything below rather than being "
               f"filled in with invented numbers.")

with st.expander("Data trail — which source produced each series"):
    st.code("\n".join(line for line in result.log if line.strip()) or
            "(no diagnostics)", language="text")

caption(f"Horizon: {horizon} trading days "
        f"(~{dash.trading_days_to_calendar_days(horizon)} calendar days) · "
        f"{cfg['N_PATHS']:,} paths · charts and portfolio built on "
        f"{dash.DRIFT_SCENARIO_LABELS[result.drift_scenario]} · "
        f"volatility measured over the last {cfg['STAT_WINDOW_DAYS']} "
        f"trading days.")


# ---- Per stock --------------------------------------------------------------

for stats in result.profiles:
    ticker = stats.ticker
    fund = result.fundamentals[ticker]

    st.divider()
    st.subheader(f"{ticker} — {dash.SECTORS.get(ticker, 'sector not labelled')}")

    fact_col, fund_col = st.columns([3, 2])

    with fact_col:
        md("**FACT** — measured from real price history. Not an opinion.")
        table([{"Measure": row.label,
                "Value": f"{row.value}{row.inline}".strip(),
                "What it means": row.note}
               for row in dash.fact_rows(stats, cfg)])

    with fund_col:
        md("**EXTERNAL VIEW** — other people's opinions. Not measured here.")
        if fund.any_values:
            table([{"Field": row.label,
                    "Value": f"{row.value}{row.inline}".strip()}
                   for row in dash.fundamental_rows(fund)])
        else:
            md("Fundamentals unavailable — the web lookup returned nothing "
               "usable and no values were supplied in the `FUNDAMENTALS` "
               "config block.")

        if fund.has_targets():
            gap = dash.target_gap_pct(fund, stats.last_close)
            md(f"Analyst 12-month target — low "
               f"**{dash.fmt(fund.values.get('target_low'))}** | avg "
               f"**{dash.fmt(fund.values.get('target_avg'))}** | high "
               f"**{dash.fmt(fund.values.get('target_high'))}** PKR")
            if gap is not None:
                md(f"The average target sits **{gap:+.1f}%** from today's price "
                   f"(source: {fund.origin.get('target_avg', 'n/a')}). Targets "
                   f"are 12-month views, while the cone below is "
                   f"~{dash.trading_days_to_calendar_days(horizon)} days — the "
                   f"horizons do not match. They are forecasts made by people, "
                   f"revised often and frequently missed.")
            if fund.note:
                caption(f"Your note: {fund.note}")
        else:
            md("No analyst targets supplied for this ticker, and none were "
               "scraped. Add them under `FUNDAMENTALS` to see them on the "
               "chart.")

        if fund.web_error:
            caption(f"Web lookup: {fund.web_error}")

    st.pyplot(result.figures[f"fan_{ticker}"])

    md(f"**UNCERTAINTY** — {cfg['N_PATHS']:,} simulated paths over {horizon} "
       f"trading days. These are RANGES. No single number here is a prediction.")
    table([{
        "Scenario": cone.label + (" ← used for the chart"
                                  if key == result.drift_scenario else ""),
        "Daily log drift": f"{cone.daily_drift_used:+.5f}",
        "P10": dash.fmt(cone.pct[10]),
        "P25": dash.fmt(cone.pct[25]),
        "P50": dash.fmt(cone.pct[50]),
        "P75": dash.fmt(cone.pct[75]),
        "P90": dash.fmt(cone.pct[90]),
        "P(above today)": f"{cone.prob_above_start * 100:.1f}%",
        "80% band width": f"{dash.band80_width_pct(cone, stats.last_close):.1f}% "
                          f"of today's price",
    } for key, cone in ((k, result.cones[ticker][k])
                        for k in ("zero", "historical"))])
    caption(dash.drift_caution(stats, result.drift_scenario))


# ---- Portfolio --------------------------------------------------------------

st.divider()
st.header(f"Portfolio — the {len(result.tickers)} names held together")

port = result.portfolio

if port is None:
    st.info("Portfolio view skipped: it needs at least 2 tickers with usable "
            "data.")
else:
    md("**FACT** — the basket as constructed")
    md(f"Capital per stock **${cfg['CAPITAL_USD_PER_STOCK']:,.2f}** at "
       f"**{cfg['USD_PKR']:,.2f}** PKR/USD, so "
       f"**{port.start_value_pkr:,.2f} PKR** "
       f"(${port.start_value_usd:,.2f}) in total. Fractional share counts are a "
       f"modelling device — real PSX lots are whole shares.")
    table([{"Ticker": p.ticker,
            "Shares": f"{port.shares[p.ticker]:,.4f}",
            "Last close (PKR)": f"{p.last_close:,.2f}",
            "Sector": dash.SECTORS.get(p.ticker, "sector not labelled")}
           for p in result.profiles])

    corr_col, hist_col = st.columns([2, 3])

    with corr_col:
        md("**Correlation of daily returns** — 1.00 = moves in lockstep, "
           "0.00 = unrelated.")
        st.dataframe(port.corr.round(2), width="stretch")
        caption(f"Average pairwise correlation: {port.avg_pairwise_corr:.2f}. "
                f"Shocks are drawn together using the Cholesky factor of the "
                f"real covariance matrix, so the names move together in the "
                f"model exactly as much as they did in reality.")

    with hist_col:
        st.pyplot(result.figures["portfolio"])

    md(f"**UNCERTAINTY** — {cfg['N_PATHS']:,} correlated paths, "
       f"{port.drift_label}, {horizon} trading days. Total basket value at the "
       f"horizon:")
    table([
        {"Measure": "Total basket value (PKR)",
         "P10": f"{port.pct[10]:,.2f}", "P50": f"{port.pct[50]:,.2f}",
         "P90": f"{port.pct[90]:,.2f}"},
        {"Measure": f"Same in USD at {cfg['USD_PKR']:,.2f}",
         "P10": f"${port.pct[10] / cfg['USD_PKR']:,.2f}",
         "P50": f"${port.pct[50] / cfg['USD_PKR']:,.2f}",
         "P90": f"${port.pct[90] / cfg['USD_PKR']:,.2f}"},
    ])
    md(f"Probability the basket finishes above "
       f"**{port.start_value_pkr:,.2f} PKR**: **{port.prob_up * 100:.1f}%**. "
       f"Average worst drawdown along a path: "
       f"**{port.avg_max_drawdown * 100:.1f}%** — on a typical path the basket "
       f"dips this far below its own peak at some point before the horizon, so "
       f"the ride is bumpier than the endpoint suggests.")
    if port.drift_scenario == "zero":
        caption("Slightly above 50% is expected even with zero drift: prices "
                "compound, so a basket of lognormal outcomes leans a little to "
                "the upside around its median. That is arithmetic, not a view "
                "on direction.")
    else:
        caption("This scenario feeds each name's measured past drift into the "
                "model, so the figure above reflects that assumption "
                "continuing, not a view of the future.")

    md("**DIVERSIFICATION** — why the basket's cone is tighter than the "
       "stocks' cones")
    table([{
        "Average 80% band of the individual stocks":
            f"{port.avg_single_band80_pct:.1f}% of starting value",
        "80% band of the combined basket":
            f"{port.band80_width_pct:.1f}% of starting value",
        "Reduction in band width":
            f"{port.diversification_benefit_pct:.1f}%",
    }])
    md(dash.diversification_paragraph(port))


# ---- Closing framing --------------------------------------------------------

st.divider()
st.header(dash.CLOSING_HEADING.title())

for paragraph in dash.closing_paragraphs(result.profiles, port, cfg):
    md(paragraph)

caption("Charts and numbers on this page are produced by the same functions as "
        "`python psx_dashboard.py`. Nothing here is investment advice.")
