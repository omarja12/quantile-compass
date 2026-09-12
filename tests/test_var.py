import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantile_compass.var import (
    age_weighted_historical_var,
    historical_var,
    parametric_var,
    scale_var_horizon,
)


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


def test_parametric_var_matches_closed_form_single_factor():
    daily_vol = 0.02
    cov = np.array([[daily_vol**2]])
    got = parametric_var([1.0], cov, alpha=0.01, horizon_days=1, annualized_cov=False)
    assert got == pytest.approx(norm.ppf(0.99) * daily_vol)


def test_parametric_var_deannualizes_correctly():
    annual_vol = 0.20
    cov = np.array([[annual_vol**2]])
    got = parametric_var([1.0], cov, alpha=0.01, horizon_days=1, trading_days=250)
    expected = np.sqrt(1 / 250) * norm.ppf(0.99) * annual_vol
    assert got == pytest.approx(expected)


def test_parametric_var_accounts_for_correlation():
    """Negatively correlated factors must give a smaller combined VaR than positively correlated ones."""
    v = 0.02**2
    positive = np.array([[v, 0.8 * v], [0.8 * v, v]])
    negative = np.array([[v, -0.8 * v], [-0.8 * v, v]])
    kw = dict(alpha=0.01, horizon_days=1, annualized_cov=False)
    assert parametric_var([1, 1], negative, **kw) < parametric_var([1, 1], positive, **kw)


def test_stand_alone_var_picks_out_single_factor():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    kw = dict(alpha=0.01, horizon_days=1, annualized_cov=False)
    assert parametric_var([1, 0], cov, **kw) == pytest.approx(norm.ppf(0.99) * np.sqrt(0.04))
    assert parametric_var([0, 1], cov, **kw) == pytest.approx(norm.ppf(0.99) * np.sqrt(0.09))


def test_historical_var_matches_numpy_quantile():
    rng = np.random.default_rng(11)
    r = _series(rng.normal(0, 0.02, 1000))
    assert historical_var(r, alpha=0.05) == pytest.approx(-np.quantile(r.to_numpy(), 0.05))


def test_historical_var_is_positive_for_a_loss_making_tail():
    r = _series([-0.10, -0.05, 0.0, 0.01, 0.02, 0.03])
    assert historical_var(r, alpha=0.10) > 0


def test_age_weighted_var_reacts_to_recent_shocks_more_than_plain_historical():
    """Recent large losses should push age-weighted VaR above the equally weighted figure."""
    calm = list(np.full(400, -0.001))
    recent_crash = [-0.08, -0.09, -0.10, -0.07, -0.11]
    r = _series(calm + recent_crash)
    assert age_weighted_historical_var(r, alpha=0.05) > historical_var(r, alpha=0.05)


def test_age_weighted_var_converges_to_historical_as_lambda_approaches_one():
    rng = np.random.default_rng(5)
    r = _series(rng.normal(0, 0.02, 2000))
    aged = age_weighted_historical_var(r, alpha=0.05, lam=0.9999)
    plain = historical_var(r, alpha=0.05)
    assert aged == pytest.approx(plain, abs=0.005)


def test_horizon_scaling_is_sqrt_of_time():
    assert scale_var_horizon(0.04, 10) == pytest.approx(0.04 * np.sqrt(10))
    assert scale_var_horizon(0.04, 1) == pytest.approx(0.04)


def test_parametric_var_horizon_scaling_matches_explicit_scaling():
    cov = np.array([[0.02**2]])
    one_day = parametric_var([1.0], cov, horizon_days=1, annualized_cov=False)
    ten_day = parametric_var([1.0], cov, horizon_days=10, annualized_cov=False)
    assert ten_day == pytest.approx(scale_var_horizon(one_day, 10))


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 2.0])
def test_invalid_alpha_rejected(bad_alpha):
    r = _series([0.01, -0.01])
    with pytest.raises(ValueError, match="alpha must be"):
        historical_var(r, alpha=bad_alpha)


def test_empty_series_rejected():
    with pytest.raises(ValueError, match="empty series"):
        historical_var(pd.Series([], dtype=float))


def test_negative_horizon_rejected():
    with pytest.raises(ValueError, match="horizon_days must be positive"):
        scale_var_horizon(0.04, 0)
