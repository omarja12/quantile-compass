"""Log returns and the equity/forex decomposition of the portfolio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default scenario: a ruble-based investor holding US and German equities.
DEFAULT_MAX_GAP_DAYS = 7


@dataclass(frozen=True)
class PortfolioSpec:
    """
    Weights and betas of the equity portfolio.

    Attributes:
        w_us: Share of the portfolio held in US equity.
        w_de: Share held in German equity.
        beta_us: Market beta of the US holding.
        beta_de: Market beta of the German holding.
    """

    w_us: float = 0.6
    w_de: float = 0.4
    beta_us: float = 1.6
    beta_de: float = 1.3

    def __post_init__(self) -> None:
        total = self.w_us + self.w_de
        if not np.isclose(total, 1.0):
            raise ValueError(f"weights must sum to 1, got {total}")


def compute_log_returns(
    prices: pd.DataFrame | pd.Series,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame | pd.Series:
    """
    Continuously-compounded (log) returns.

    Returns spanning more than `max_gap_days` calendar days are set to NaN:
    when the data source is missing a stretch of days, the change across that
    hole is not a one-day return and must not be treated as one.
    """
    log_ret = np.log(prices / prices.shift(1))
    spacing = prices.index.to_series().diff().dt.days
    too_wide = spacing > max_gap_days
    if isinstance(log_ret, pd.DataFrame):
        log_ret.loc[too_wide, :] = np.nan
    else:
        log_ret.loc[too_wide] = np.nan
    return log_ret


def decompose_portfolio_returns(
    prices: pd.DataFrame,
    spec: PortfolioSpec | None = None,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame:
    """
    Split the portfolio's ruble return into its equity and forex components.

    The ruble value of a foreign holding is the local-currency price times the
    exchange rate, so in logs the return separates cleanly into an equity part
    and a currency part:

        r_equity = w_de * beta_de * r_DAX + w_us * beta_us * r_NASDAQ
        r_forex  = w_de * r_EURRUB      + w_us * r_USDRUB
        r_total  = r_equity + r_forex

    Args:
        prices: Price data containing NASDAQ100, DAX, USD_RUB, EUR_RUB columns.
        spec: Portfolio weights/betas; defaults to 60/40 US/German with betas 1.6/1.3.
        max_gap_days: Returns spanning wider gaps than this are dropped.

    Returns:
        DataFrame with `equity`, `forex` and `total` return columns, NaN rows removed.
    """
    spec = spec or PortfolioSpec()
    required = {"NASDAQ100", "DAX", "USD_RUB", "EUR_RUB"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"missing required price columns: {sorted(missing)}")

    r = compute_log_returns(prices[sorted(required)], max_gap_days=max_gap_days)

    equity = spec.w_de * spec.beta_de * r["DAX"] + spec.w_us * spec.beta_us * r["NASDAQ100"]
    forex = spec.w_de * r["EUR_RUB"] + spec.w_us * r["USD_RUB"]

    out = pd.DataFrame({"equity": equity, "forex": forex})
    out["total"] = out["equity"] + out["forex"]
    return out.dropna(how="any")
