"""Load and clean the fetched market data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "market_data.csv"

# The RUB exchange-rate series are sparse before mid-2006 on the free data
# feed, leaving multi-month holes in the joined index (a 448-day hole at
# 2006-05-16, a 210-day one at 2005-02-22). Returns computed across holes like
# that are not daily returns, so the usable sample starts where the data
# becomes dense. From this date there are ~246 observations per year - i.e.
# genuine daily coverage - with a single 26-day gap in Aug 2008 that
# `returns.compute_log_returns` invalidates rather than treats as one day.
DENSE_DATA_START = "2006-05-16"

# Columns known to occasionally carry single-day bad ticks in the free Yahoo
# Finance feed (thin FX crosses, futures contract-roll artifacts). VIX is
# deliberately excluded: it can legitimately move 50-100%+ in a single real
# session (e.g. the Feb 2018 "Volmageddon" spike), so the same filter would
# wrongly "clean" a genuine event.
CLEANABLE_COLUMNS = ["NASDAQ100", "DAX", "USD_RUB", "EUR_RUB", "SP500", "GOLD"]


def load_market_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw fetched CSV, indexed by date."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


def clean_bad_ticks(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Detect and interpolate single-day "V-shaped" price spikes: a day whose
    return exceeds `threshold` in magnitude and is (mostly) reversed by the
    very next day's return in the opposite direction. Real market moves of
    that size persist; a value that fully round-trips in one day is a bad
    tick, not a crisis.

    Returns a new DataFrame; flagged points are replaced by linear
    interpolation between their (good) neighbors.
    """
    out = df.copy()
    for col in CLEANABLE_COLUMNS:
        if col not in out.columns:
            continue
        s = out[col]
        ret = s.pct_change()
        next_ret = ret.shift(-1)
        # flag point t where |ret[t]| is extreme and next_ret[t] roughly
        # reverses it (opposite sign, similar magnitude)
        is_spike = ret.abs() > threshold
        reverts = (np.sign(ret) != np.sign(next_ret)) & (next_ret.abs() > threshold * 0.5)
        flagged = is_spike & reverts
        if flagged.any():
            s = s.mask(flagged)
            s = s.interpolate(method="linear")
            out[col] = s
    return out


def prepare_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    start: str | None = DENSE_DATA_START,
) -> pd.DataFrame:
    """
    Load, trim to the dense period, and clean the dataset - the standard entry
    point for the rest of the package.

    Pass ``start=None`` to keep the full fetched history including its sparse
    early section.
    """
    df = load_market_data(path)
    if start is not None:
        df = df.loc[start:]
    return clean_bad_ticks(df)
