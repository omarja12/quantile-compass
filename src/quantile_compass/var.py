"""Value-at-Risk estimators: parametric, historical, and age-weighted historical."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .volatility import (
    DEFAULT_BURN_IN,
    DEFAULT_LAMBDA,
    TRADING_DAYS,
    ewma_covariance,
    ewma_variance,
)

DEFAULT_ALPHA = 0.01


def parametric_var(
    sensitivities: np.ndarray | list[float],
    cov_matrix: pd.DataFrame | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    horizon_days: int = 1,
    trading_days: int = TRADING_DAYS,
    annualized_cov: bool = True,
) -> float:
    r"""
    Normal parametric VaR for a linear portfolio of risk factors.

    .. math::
        VaR_{h,\alpha} = \Phi^{-1}(1-\alpha)\,\sqrt{\theta' \Omega_h \theta}

    Args:
        sensitivities: Exposure vector :math:`\theta` - e.g. ``[1, 1]`` for the
            combined portfolio, ``[1, 0]`` for equity stand-alone.
        cov_matrix: Risk-factor covariance matrix.
        alpha: Significance level (0.01 = 99% confidence).
        horizon_days: Risk horizon in trading days.
        trading_days: Trading days per year, used to de-annualize.
        annualized_cov: Whether `cov_matrix` is annualized.

    Returns:
        VaR as a positive fraction of portfolio value (0.04 = a 4% loss).
    """
    _validate_alpha(alpha)
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    theta = np.asarray(sensitivities, dtype=float)
    omega = np.asarray(cov_matrix, dtype=float)
    variance = float(theta @ omega @ theta)
    if variance < 0:
        raise ValueError("covariance matrix produced a negative variance")
    scale = np.sqrt(horizon_days / trading_days) if annualized_cov else np.sqrt(horizon_days)
    return float(scale * norm.ppf(1 - alpha) * np.sqrt(variance))


def historical_var(returns: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """
    Historical-simulation VaR: the empirical alpha-quantile of realised
    returns, sign-flipped so a loss is reported positive. Makes no
    distributional assumption.
    """
    _validate_alpha(alpha)
    if returns.empty:
        raise ValueError("cannot compute VaR on an empty series")
    return float(-returns.quantile(alpha))


def age_weighted_historical_var(
    returns: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    r"""
    Age-weighted ("hybrid") historical VaR after Boudoukh, Richardson and
    Whitelaw (1998).

    Each observation gets an exponentially decaying probability weight, most
    recent first (:math:`\omega_T = 1-\lambda`, :math:`\omega_{i-1} = \lambda\omega_i`).
    Returns are then sorted and the quantile is read off the cumulative
    weighted distribution, so recent market conditions dominate the tail
    without discarding older history outright.
    """
    _validate_alpha(alpha)
    if returns.empty:
        raise ValueError("cannot compute VaR on an empty series")

    n = len(returns)
    # weights increase towards the most recent observation
    ages = np.arange(n - 1, -1, -1)
    weights = (1 - lam) * lam**ages
    weights /= weights.sum()

    order = np.argsort(returns.to_numpy())
    sorted_returns = returns.to_numpy()[order]
    cumulative = np.cumsum(weights[order])

    idx = int(np.searchsorted(cumulative, alpha))
    idx = min(idx, n - 1)
    return float(-sorted_returns[idx])


def scale_var_horizon(var_1day: float, horizon_days: int) -> float:
    """Square-root-of-time scaling of a 1-day VaR (assumes i.i.d. returns)."""
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    return var_1day * np.sqrt(horizon_days)


def rolling_parametric_var(
    equity: pd.Series,
    forex: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    horizon_days: int = 1,
    lam: float = DEFAULT_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
) -> pd.Series:
    r"""
    Parametric VaR recomputed at every date from that date's EWMA covariance.

    The combined portfolio's variance at time *t* is

    .. math::
        \sigma^2_t = \sigma^2_{E,t} + 2\sigma_{EF,t} + \sigma^2_{F,t}

    (the quadratic form with exposures :math:`\theta = [1, 1]`), and the VaR is
    :math:`\Phi^{-1}(1-\alpha)\sqrt{h}\,\sigma_t`.

    Because the EWMA estimates carry no look-ahead, the value at *t* is what the
    model would have forecast on the morning of *t* - which is what makes the
    series usable for backtesting.

    Returns:
        VaR as a positive fraction of portfolio value, NaN through the burn-in.
    """
    _validate_alpha(alpha)
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    kw = dict(lam=lam, burn_in=burn_in)
    var_e = ewma_variance(equity, **kw)
    var_f = ewma_variance(forex, **kw)
    cov_ef = ewma_covariance(equity, forex, **kw)

    total_var = var_e + 2 * cov_ef + var_f
    total_var = total_var.clip(lower=0)  # numerical guard
    var = norm.ppf(1 - alpha) * np.sqrt(horizon_days) * np.sqrt(total_var)
    return var.rename("var")


def count_var_breaches(returns: pd.Series, var_series: pd.Series) -> dict[str, float]:
    """
    Backtest a VaR series: how often did the realised loss exceed the forecast?

    A well-calibrated 99% VaR should be breached on about 1% of days. Far fewer
    means the model is too conservative and ties up capital; far more means it
    understates the risk.

    Returns:
        Dict with the number of observations compared, the breach count, the
        realised breach rate and the rate the confidence level implies.
    """
    aligned = pd.concat([returns.rename("r"), var_series.rename("v")], axis=1).dropna()
    if aligned.empty:
        return {"observations": 0, "breaches": 0, "breach_rate": float("nan")}
    breaches = (aligned["r"] < -aligned["v"]).sum()
    return {
        "observations": int(len(aligned)),
        "breaches": int(breaches),
        "breach_rate": float(breaches / len(aligned)),
    }


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
