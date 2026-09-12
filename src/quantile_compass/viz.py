"""Reusable plotting helpers shared by the notebook, the docs and the app."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Crisis windows the sample covers, used to shade charts consistently.
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "Global financial crisis": ("2008-09-01", "2009-03-31"),
    "Ruble crisis": ("2014-11-01", "2015-02-28"),
    "COVID-19": ("2020-02-19", "2020-04-30"),
    "Russia-Ukraine": ("2022-02-24", "2022-04-30"),
}

PALETTE = {
    "equity": "#2563eb",
    "forex": "#db2777",
    "total": "#059669",
    "accent": "#d97706",
    "muted": "#6b7280",
    "crisis": "#9ca3af",
}


def shade_crises(ax: plt.Axes, windows: dict[str, tuple[str, str]] | None = None) -> None:
    """Shade the crisis windows on a time-axis chart."""
    for start, end in (windows or CRISIS_WINDOWS).values():
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end), color=PALETTE["crisis"], alpha=0.18, lw=0
        )


def _format_time_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(alpha=0.25, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_price_levels(
    prices: pd.DataFrame, columns: list[str], ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot price series rebased to 100 at the start of the sample."""
    ax = ax or plt.subplots(figsize=(10, 4.5), constrained_layout=True)[1]
    for i, col in enumerate(columns):
        rebased = prices[col] / prices[col].iloc[0] * 100
        color = list(PALETTE.values())[i % len(PALETTE)]
        ax.plot(rebased.index, rebased, lw=1.3, label=col, color=color)
    shade_crises(ax)
    _format_time_axis(ax, "Index level (start = 100)")
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_volatility(
    vol_equity: pd.Series,
    vol_forex: pd.Series,
    ax: plt.Axes | None = None,
    as_percent: bool = True,
) -> plt.Axes:
    """Plot the two EWMA volatility series with crisis windows shaded."""
    ax = ax or plt.subplots(figsize=(10, 4.5), constrained_layout=True)[1]
    scale = 100 if as_percent else 1
    ax.plot(vol_equity.index, vol_equity * scale, lw=1.2, color=PALETTE["equity"], label="Equity")
    ax.plot(vol_forex.index, vol_forex * scale, lw=1.2, color=PALETTE["forex"], label="Forex")
    shade_crises(ax)
    _format_time_axis(ax, "EWMA annualised volatility (%)" if as_percent else "EWMA volatility")
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_correlation(correlation: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the EWMA equity/forex correlation through time."""
    ax = ax or plt.subplots(figsize=(10, 4.5), constrained_layout=True)[1]
    ax.plot(correlation.index, correlation, lw=1.2, color=PALETTE["accent"])
    ax.axhline(0, color=PALETTE["muted"], lw=0.9, ls="--")
    shade_crises(ax)
    _format_time_axis(ax, "EWMA equity-forex correlation")
    ax.set_ylim(-1, 1)
    return ax


def plot_return_distribution(
    returns: pd.Series, ax: plt.Axes | None = None, bins: int = 100
) -> plt.Axes:
    """Histogram of portfolio returns."""
    ax = ax or plt.subplots(figsize=(10, 4.5), constrained_layout=True)[1]
    ax.hist(returns * 100, bins=bins, color=PALETTE["total"], alpha=0.85)
    ax.set_xlabel("Daily portfolio return (%)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25, lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plot_var_comparison(var_by_method: dict[str, float], ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart comparing VaR estimates across methods."""
    ax = ax or plt.subplots(figsize=(7, 4.2), constrained_layout=True)[1]
    names = list(var_by_method)
    values = [var_by_method[n] * 100 for n in names]
    colors = [PALETTE["equity"], PALETTE["forex"], PALETTE["accent"]][: len(names)]
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("1-day 99% VaR (%)")
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, max(values) * 1.2)
    return ax
