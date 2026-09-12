import numpy as np
import pandas as pd
import pytest

from quantile_compass.returns import (
    PortfolioSpec,
    compute_log_returns,
    decompose_portfolio_returns,
)


def _dates(n, start="2020-01-01", freq="D"):
    return pd.date_range(start, periods=n, freq=freq)


def test_log_returns_match_hand_calculation():
    prices = pd.Series([100.0, 110.0, 99.0], index=_dates(3))
    result = compute_log_returns(prices)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(1.10))
    assert result.iloc[2] == pytest.approx(np.log(99 / 110))


def test_log_returns_are_additive_across_time():
    """log returns sum to the total log return - the property the decomposition relies on."""
    prices = pd.Series([100.0, 105.0, 120.0, 90.0], index=_dates(4))
    r = compute_log_returns(prices).dropna()
    assert r.sum() == pytest.approx(np.log(90 / 100))


def test_returns_spanning_a_data_gap_are_invalidated():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-06-01", "2020-06-02"])
    prices = pd.Series([100.0, 101.0, 300.0, 303.0], index=idx)
    r = compute_log_returns(prices, max_gap_days=7)
    assert r.iloc[1] == pytest.approx(np.log(101 / 100))
    assert np.isnan(r.iloc[2]), "the 150-day jump must not count as a daily return"
    assert r.iloc[3] == pytest.approx(np.log(303 / 300))


def test_decomposition_components_sum_to_total():
    n = 50
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "NASDAQ100": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "DAX": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "USD_RUB": 60 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "EUR_RUB": 70 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        },
        index=_dates(n),
    )
    out = decompose_portfolio_returns(prices)
    pd.testing.assert_series_equal(out["equity"] + out["forex"], out["total"], check_names=False)


def test_decomposition_applies_weights_and_betas():
    idx = _dates(2)
    prices = pd.DataFrame(
        {
            "NASDAQ100": [100.0, 110.0],
            "DAX": [100.0, 105.0],
            "USD_RUB": [60.0, 66.0],
            "EUR_RUB": [70.0, 70.0],
        },
        index=idx,
    )
    spec = PortfolioSpec(w_us=0.6, w_de=0.4, beta_us=1.6, beta_de=1.3)
    out = decompose_portfolio_returns(prices, spec=spec)

    expected_equity = 0.4 * 1.3 * np.log(1.05) + 0.6 * 1.6 * np.log(1.10)
    expected_forex = 0.4 * np.log(1.0) + 0.6 * np.log(1.10)
    assert out["equity"].iloc[0] == pytest.approx(expected_equity)
    assert out["forex"].iloc[0] == pytest.approx(expected_forex)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="weights must sum to 1"):
        PortfolioSpec(w_us=0.7, w_de=0.4)


def test_missing_columns_raise():
    prices = pd.DataFrame({"NASDAQ100": [1.0, 2.0]}, index=_dates(2))
    with pytest.raises(KeyError, match="missing required price columns"):
        decompose_portfolio_returns(prices)
