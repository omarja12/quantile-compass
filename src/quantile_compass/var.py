"""Value-at-Risk estimators: parametric, historical, and age-weighted historical."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .volatility import DEFAULT_LAMBDA, TRADING_DAYS

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


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
