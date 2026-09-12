import numpy as np
import pandas as pd
import pytest

from quantile_compass.volatility import (
    annualize_volatility,
    covariance_matrix,
    ewma_correlation,
    ewma_covariance,
    ewma_variance,
    ewma_volatility,
)


def _series(values, name="r"):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)), name=name)


UNMASKED = dict(mask_burn_in=False)


def test_ewma_variance_matches_hand_computed_recursion():
    r = _series([0.01, -0.02, 0.03, -0.01])
    lam = 0.94
    out = ewma_variance(r, lam=lam, burn_in=4, **UNMASKED)

    seed = float(np.mean(np.array([0.01, -0.02, 0.03, -0.01]) ** 2))
    expected = [seed]
    for prev in [0.01, -0.02, 0.03]:
        expected.append((1 - lam) * prev**2 + lam * expected[-1])

    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-12)


def test_ewma_variance_uses_only_past_information():
    """A return after the seeding window must not affect estimates on its own day."""
    rng = np.random.default_rng(1)
    base = list(rng.normal(0, 0.01, 60))
    calm = ewma_variance(_series(base + [0.005]), lam=0.94, burn_in=50, **UNMASKED)
    shocked = ewma_variance(_series(base + [0.5]), lam=0.94, burn_in=50, **UNMASKED)
    assert calm.iloc[-1] == pytest.approx(shocked.iloc[-1]), "no look-ahead allowed"
    # ...and the shock must show up the *next* day if one exists
    calm2 = ewma_variance(_series(base + [0.005, 0.0]), lam=0.94, burn_in=50, **UNMASKED)
    shocked2 = ewma_variance(_series(base + [0.5, 0.0]), lam=0.94, burn_in=50, **UNMASKED)
    assert shocked2.iloc[-1] > calm2.iloc[-1]


def test_burn_in_window_is_masked_by_default():
    rng = np.random.default_rng(2)
    r = _series(rng.normal(0, 0.01, 300))
    out = ewma_variance(r, burn_in=250)
    assert out.iloc[:250].isna().all()
    assert out.iloc[250:].notna().all()


def test_ewma_volatility_is_sqrt_of_variance():
    r = _series([0.01, -0.02, 0.03, -0.015])
    np.testing.assert_allclose(
        ewma_volatility(r, burn_in=2, **UNMASKED).to_numpy(),
        np.sqrt(ewma_variance(r, burn_in=2, **UNMASKED).to_numpy()),
        rtol=1e-12,
    )


def test_ewma_covariance_of_series_with_itself_equals_variance():
    r = _series([0.01, -0.02, 0.03, -0.015])
    np.testing.assert_allclose(
        ewma_covariance(r, r, burn_in=2, **UNMASKED).to_numpy(),
        ewma_variance(r, burn_in=2, **UNMASKED).to_numpy(),
        rtol=1e-12,
    )


def test_correlation_of_identical_series_is_one():
    r = _series([0.01, -0.02, 0.03, -0.015])
    corr = ewma_correlation(r, r, burn_in=2, **UNMASKED)
    np.testing.assert_allclose(corr.to_numpy(), np.ones(len(r)), rtol=1e-10)


def test_perfectly_opposed_series_have_correlation_minus_one():
    r = _series([0.01, -0.02, 0.03, -0.015])
    corr = ewma_correlation(r, -r, burn_in=2, **UNMASKED)
    np.testing.assert_allclose(corr.to_numpy(), -np.ones(len(r)), rtol=1e-10)


def test_correlation_stays_within_bounds_on_random_data():
    rng = np.random.default_rng(7)
    a = _series(rng.normal(0, 0.01, 300))
    b = _series(rng.normal(0, 0.02, 300))
    corr = ewma_correlation(a, b).dropna().to_numpy()
    assert np.all(corr >= -1.0000001) and np.all(corr <= 1.0000001)


def test_annualization_scales_by_sqrt_trading_days():
    assert annualize_volatility(0.01, trading_days=250) == pytest.approx(0.01 * np.sqrt(250))


def test_covariance_matrix_is_symmetric_and_labelled():
    rng = np.random.default_rng(3)
    eq = _series(rng.normal(0, 0.01, 600), name="equity")
    fx = _series(rng.normal(0, 0.005, 600), name="forex")
    m = covariance_matrix(eq, fx)
    assert list(m.columns) == ["equity", "forex"]
    assert m.loc["equity", "forex"] == pytest.approx(m.loc["forex", "equity"])
    assert m.loc["equity", "equity"] > 0


def test_covariance_matrix_fails_loudly_without_enough_data():
    rng = np.random.default_rng(4)
    eq = _series(rng.normal(0, 0.01, 100), name="equity")
    fx = _series(rng.normal(0, 0.005, 100), name="forex")
    with pytest.raises(ValueError, match="no estimate is available"):
        covariance_matrix(eq, fx, burn_in=250)


def test_annualized_matrix_is_scaled_version_of_daily():
    rng = np.random.default_rng(9)
    eq = _series(rng.normal(0, 0.01, 600), name="equity")
    fx = _series(rng.normal(0, 0.005, 600), name="forex")
    daily = covariance_matrix(eq, fx, annualize=False)
    annual = covariance_matrix(eq, fx, annualize=True, trading_days=250)
    np.testing.assert_allclose(annual.to_numpy(), daily.to_numpy() * 250, rtol=1e-12)


@pytest.mark.parametrize("bad_lambda", [0.0, 1.0, -0.5, 1.5])
def test_invalid_lambda_rejected(bad_lambda):
    with pytest.raises(ValueError, match="lambda must be"):
        ewma_variance(_series([0.01, 0.02]), lam=bad_lambda)
