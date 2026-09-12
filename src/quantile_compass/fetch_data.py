"""
Fetch daily historical price data for the core portfolio risk factors and a
few supplementary risk-factor series, and align them on their common
overlapping date range.

Run directly to (re)populate data/raw/market_data.csv:

    python -m quantile_compass.fetch_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# Core scenario: 60% NASDAQ100 (beta 1.6) / 40% DAX (beta 1.3) equity
# portfolio, viewed in RUB terms.
CORE_TICKERS = {
    "^NDX": "NASDAQ100",
    "^GDAXI": "DAX",
    "RUB=X": "USD_RUB",
    "EURRUB=X": "EUR_RUB",
}

# Supplementary series for a broader risk-factor comparison. Not part of the
# core weighted portfolio.
EXTRA_TICKERS = {
    "^GSPC": "SP500",
    "^VIX": "VIX",
    "GC=F": "GOLD",
}

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "market_data.csv"


def fetch_series(ticker: str) -> pd.Series:
    """Fetch max-available daily closing prices for one ticker."""
    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker!r}")
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "Date"
    return close


def fetch_all(tickers: dict[str, str]) -> pd.DataFrame:
    """Fetch every ticker in `tickers` and report each one's own availability."""
    series = {}
    for ticker, name in tickers.items():
        s = fetch_series(ticker)
        print(
            f"  {name:10s} ({ticker:9s}): {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} rows)"
        )
        series[name] = s
    return pd.DataFrame(series)


def build_dataset(include_extra: bool = True) -> pd.DataFrame:
    print("Fetching core series...")
    core = fetch_all(CORE_TICKERS)

    if include_extra:
        print("Fetching supplementary series...")
        extra = fetch_all(EXTRA_TICKERS)
        combined = core.join(extra, how="inner")
    else:
        combined = core

    # Inner join across everything requested: only dates where all requested
    # series have data. Sorted, deduplicated, trading-day gaps left as-is.
    combined = combined.dropna(how="any").sort_index()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--core-only", action="store_true", help="Skip the supplementary series.")
    args = parser.parse_args()

    df = build_dataset(include_extra=not args.core_only)

    print(f"\nResolved common date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Rows: {len(df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
