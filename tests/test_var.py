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


def _corr_series(n, seed, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n)
    return (
        pd.Series(rng.normal(0, vol, n), index=idx, name="equity"),
        pd.Series(rng.normal(0, vol, n), index=idx, name="forex"),
    )


def test_rolling_var_matches_pointwise_parametric_var_at_the_same_date():
    """The rolling series must agree with the single-date calculation it generalises."""
    from quantile_compass.var import rolling_parametric_var
    from quantile_compass.volatility import covariance_matrix

    eq, fx = _corr_series(800, seed=21)
    rolling = rolling_parametric_var(eq, fx, alpha=0.01, horizon_days=1)
    cov = covariance_matrix(eq, fx, annualize=True)
    pointwise = parametric_var([1, 1], cov, alpha=0.01, horizon_days=1)
    assert rolling.iloc[-1] == pytest.approx(pointwise, rel=1e-9)


def test_rolling_var_is_masked_through_the_burn_in():
    from quantile_compass.var import rolling_parametric_var

    eq, fx = _corr_series(600, seed=3)
    v = rolling_parametric_var(eq, fx, burn_in=250)
    assert v.iloc[:250].isna().all()
    assert v.iloc[250:].notna().all()


def test_rolling_var_widens_with_confidence_level():
    from quantile_compass.var import rolling_parametric_var

    eq, fx = _corr_series(700, seed=8)
    low = rolling_parametric_var(eq, fx, alpha=0.05).dropna()
    high = rolling_parametric_var(eq, fx, alpha=0.001).dropna()
    assert (high > low).all()


def test_rolling_var_scales_with_sqrt_horizon():
    from quantile_compass.var import rolling_parametric_var

    eq, fx = _corr_series(600, seed=9)
    one = rolling_parametric_var(eq, fx, horizon_days=1).dropna()
    ten = rolling_parametric_var(eq, fx, horizon_days=10).dropna()
    np.testing.assert_allclose(ten.to_numpy(), one.to_numpy() * np.sqrt(10), rtol=1e-10)


def test_breach_counting_on_a_known_series():
    from quantile_compass.var import count_var_breaches

    idx = pd.date_range("2020-01-01", periods=5)
    returns = pd.Series([-0.10, 0.01, -0.02, -0.30, 0.05], index=idx)
    var = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05], index=idx)
    out = count_var_breaches(returns, var)
    assert out["observations"] == 5
    assert out["breaches"] == 2  # -0.10 and -0.30 exceed the 0.05 threshold
    assert out["breach_rate"] == pytest.approx(0.4)


def test_breach_rate_is_near_alpha_for_well_specified_normal_data():
    """A normal VaR on genuinely normal data should breach at roughly alpha."""
    from quantile_compass.var import count_var_breaches, rolling_parametric_var

    eq, fx = _corr_series(6000, seed=101)
    total = eq + fx
    var = rolling_parametric_var(eq, fx, alpha=0.05)
    out = count_var_breaches(total, var)
    assert out["breach_rate"] == pytest.approx(0.05, abs=0.02)


def test_breach_counting_ignores_unaligned_and_missing_values():
    from quantile_compass.var import count_var_breaches

    idx = pd.date_range("2020-01-01", periods=4)
    returns = pd.Series([-0.10, np.nan, -0.02, -0.30], index=idx)
    var = pd.Series([np.nan, 0.05, 0.05, 0.05], index=idx)
    out = count_var_breaches(returns, var)
    assert out["observations"] == 2  # only the two dates where both are present
    assert out["breaches"] == 1
