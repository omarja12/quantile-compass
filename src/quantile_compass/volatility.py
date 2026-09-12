"""EWMA volatility, covariance and correlation estimation (RiskMetrics style)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.94
TRADING_DAYS = 250
DEFAULT_BURN_IN = 250


def _ewma_recursion(x: np.ndarray, lam: float, burn_in: int) -> tuple[np.ndarray, int]:
    """
    Run the EWMA recursion over `x` (a series of products r_i * r_j).

    The seed is the average of the first `burn_in` observations, so every
    estimate from index `burn_in` onward depends only on data strictly before
    it. Returns the estimates and the seed window length actually used.
    """
    n = len(x)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out, 0
    k = max(1, min(burn_in, n))
    out[0] = float(np.mean(x[:k]))
    for i in range(1, n):
        out[i] = (1 - lam) * x[i - 1] + lam * out[i - 1]
    return out, k


def ewma_variance(
    returns: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
    mask_burn_in: bool = True,
) -> pd.Series:
    r"""
    EWMA variance estimate.

    .. math::
        \hat\sigma_t^2 = (1-\lambda) r_{t-1}^2 + \lambda \hat\sigma_{t-1}^2

    The recursion is seeded from the first `burn_in` observations rather than
    the whole sample, so that from `burn_in` onward the estimate at time *t*
    depends only on returns before *t* - i.e. it is usable as a genuine
    one-day-ahead forecast. Seeding on the full sample (as textbook treatments
    often do) leaks future information into every earlier estimate.

    Args:
        returns: Return series.
        lam: Decay factor, strictly between 0 and 1.
        burn_in: Number of observations used to seed the recursion.
        mask_burn_in: If True, the seeding window is returned as NaN, since
            those estimates are informed by their own window.
    """
    _validate_lambda(lam)
    r = returns.to_numpy(dtype=float)
    var, k = _ewma_recursion(r**2, lam, burn_in)
    if mask_burn_in and len(var):
        var[:k] = np.nan
    return pd.Series(var, index=returns.index, name=f"{returns.name}_var")


def ewma_volatility(
    returns: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
    mask_burn_in: bool = True,
) -> pd.Series:
    """EWMA standard deviation - the square root of :func:`ewma_variance`."""
    var = ewma_variance(returns, lam=lam, burn_in=burn_in, mask_burn_in=mask_burn_in)
    return np.sqrt(var).rename(f"{returns.name}_vol")


def ewma_covariance(
    a: pd.Series,
    b: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
    mask_burn_in: bool = True,
) -> pd.Series:
    r"""
    EWMA covariance between two contemporaneous return series.

    .. math::
        \hat\sigma_{ab,t} = (1-\lambda) r_{a,t-1} r_{b,t-1} + \lambda \hat\sigma_{ab,t-1}

    Seeded the same way as :func:`ewma_variance`, for the same reason.
    """
    _validate_lambda(lam)
    if not a.index.equals(b.index):
        raise ValueError("series must share an index")
    ra, rb = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    cov, k = _ewma_recursion(ra * rb, lam, burn_in)
    if mask_burn_in and len(cov):
        cov[:k] = np.nan
    return pd.Series(cov, index=a.index, name="covariance")


def ewma_correlation(
    a: pd.Series,
    b: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    burn_in: int = DEFAULT_BURN_IN,
    mask_burn_in: bool = True,
) -> pd.Series:
    """EWMA correlation, i.e. covariance normalised by the two EWMA volatilities."""
    kw = dict(lam=lam, burn_in=burn_in, mask_burn_in=mask_burn_in)
    cov = ewma_covariance(a, b, **kw)
    corr = cov / (ewma_volatility(a, **kw) * ewma_volatility(b, **kw))
    return corr.rename("correlation")


def annualize_volatility(daily_vol: pd.Series | float, trading_days: int = TRADING_DAYS):
    """Scale a daily volatility to an annual one by the square root of time."""
    return daily_vol * np.sqrt(trading_days)


def covariance_matrix(
    equity: pd.Series,
    forex: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    annualize: bool = True,
    trading_days: int = TRADING_DAYS,
    burn_in: int = DEFAULT_BURN_IN,
    asof: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Build the 2x2 equity/forex covariance matrix as of a given date
    (default: the final observation).

    Returns a labelled DataFrame so it reads clearly in a notebook.
    """
    kw = dict(lam=lam, burn_in=burn_in)
    var_e_s = ewma_variance(equity, **kw)
    var_f_s = ewma_variance(forex, **kw)
    cov_s = ewma_covariance(equity, forex, **kw)
    if asof is None:
        var_e, var_f, cov_ef = var_e_s.iloc[-1], var_f_s.iloc[-1], cov_s.iloc[-1]
    else:
        var_e, var_f, cov_ef = var_e_s.loc[asof], var_f_s.loc[asof], cov_s.loc[asof]

    if not np.isfinite([var_e, var_f, cov_ef]).all():
        raise ValueError(
            f"EWMA estimates are undefined at the requested date: the series has "
            f"{len(equity)} observations and burn_in={burn_in}, so no estimate is "
            f"available outside the seeding window. Supply more data or lower burn_in."
        )
    scale = trading_days if annualize else 1.0
    matrix = np.array([[var_e, cov_ef], [cov_ef, var_f]]) * scale
    return pd.DataFrame(matrix, index=["equity", "forex"], columns=["equity", "forex"])


def _validate_lambda(lam: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be strictly between 0 and 1, got {lam}")
