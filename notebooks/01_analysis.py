# %% [markdown]
# # Portfolio Value-at-Risk: equity and currency risk decomposed
#
# A ruble-based investor holds a 60/40 portfolio of US (NASDAQ 100) and German
# (DAX) equities. Because the holdings are denominated in dollars and euros,
# the investor carries two distinct risks: the equity markets themselves, and
# the currency in which those markets are priced.
#
# This notebook decomposes the portfolio's ruble return into those two pieces,
# estimates their volatility and correlation through time with EWMA, and
# compares three ways of computing Value-at-Risk over a sample covering four
# separate crises.
#
# All of the calculations come from the `quantile_compass` package, so the same
# code backs this notebook, the docs and the Streamlit app.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from quantile_compass.data import prepare_dataset
from quantile_compass.returns import PortfolioSpec, decompose_portfolio_returns
from quantile_compass.var import (
    age_weighted_historical_var,
    historical_var,
    parametric_var,
    scale_var_horizon,
)
from quantile_compass.viz import (
    plot_correlation,
    plot_price_levels,
    plot_return_distribution,
    plot_var_comparison,
    plot_volatility,
)
from quantile_compass.volatility import (
    annualize_volatility,
    covariance_matrix,
    ewma_correlation,
    ewma_volatility,
)

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# %% [markdown]
# ## 1. Data
#
# Prices come from the `fetch_data` module (run `python -m quantile_compass.fetch_data`
# to refresh). `prepare_dataset` trims the sparse early period and repairs the
# handful of bad ticks the free data feed carries.

# %%
prices = prepare_dataset()
print(f"{prices.index.min().date()} -> {prices.index.max().date()}  ({len(prices):,} rows)")
prices.tail()

# %%
fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
plot_price_levels(prices, ["NASDAQ100", "DAX"], ax=ax)
ax.set_title("Equity indices, rebased to 100")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
plot_price_levels(prices, ["USD_RUB", "EUR_RUB"], ax=ax)
ax.set_title("Exchange rates, rebased to 100 (higher = weaker ruble)")
plt.show()

# %% [markdown]
# ## 2. Decomposing the portfolio return
#
# In logs the ruble return separates cleanly into an equity part and a currency part:
#
# $$r_t = \underbrace{w_{de}\beta_{de} r^{DAX}_t + w_{us}\beta_{us} r^{NDX}_t}_{\text{equity}}
#        + \underbrace{w_{de} r^{EURRUB}_t + w_{us} r^{USDRUB}_t}_{\text{forex}}$$

# %%
spec = PortfolioSpec(w_us=0.6, w_de=0.4, beta_us=1.6, beta_de=1.3)
returns = decompose_portfolio_returns(prices, spec=spec)
returns.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
plot_return_distribution(returns["total"], ax=ax)
ax.set_title("Distribution of daily portfolio returns")
plt.show()

# %% [markdown]
# ## 3. EWMA volatility and correlation
#
# Volatility is estimated with the RiskMetrics EWMA recursion at $\lambda = 0.94$.
# The recursion is seeded from a burn-in window rather than the full sample, so
# each estimate depends only on returns that precede it.

# %%
vol_equity = annualize_volatility(ewma_volatility(returns["equity"]))
vol_forex = annualize_volatility(ewma_volatility(returns["forex"]))
correlation = ewma_correlation(returns["equity"], returns["forex"])

fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
plot_volatility(vol_equity, vol_forex, ax=ax)
ax.set_title("EWMA annualised volatility")
plt.show()

# %%
print(f"Equity vol: mean {vol_equity.mean():.1%}, peak {vol_equity.max():.1%} on {vol_equity.idxmax().date()}")
print(f"Forex  vol: mean {vol_forex.mean():.1%}, peak {vol_forex.max():.1%} on {vol_forex.idxmax().date()}")

# %% [markdown]
# The two components peak in different crises. Equity volatility peaks in the
# 2008 crisis and again in the COVID crash; currency volatility is comparatively
# quiet through both, then dominates in 2014 and 2022 — shocks specific to the
# ruble rather than to global equity markets.

# %%
fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
plot_correlation(correlation, ax=ax)
ax.set_title("EWMA correlation between equity and forex returns")
plt.show()

# %%
print(f"Correlation: mean {correlation.mean():.3f}, range {correlation.min():.3f} to {correlation.max():.3f}")

# %% [markdown]
# ## 4. Value-at-Risk
#
# Three estimators, all at 99% confidence over one day:
#
# - **Parametric** — assumes normality, driven by the current EWMA covariance, so it
#   reacts quickly to recent shocks.
# - **Historical** — the empirical quantile of realised returns; no distributional
#   assumption, but every observation in the sample counts equally.
# - **Age-weighted** — the hybrid of Boudoukh, Richardson and Whitelaw (1998):
#   an empirical quantile over exponentially decaying probability weights.

# %%
cov = covariance_matrix(returns["equity"], returns["forex"])
cov

# %%
var_methods = {
    "Parametric": parametric_var([1, 1], cov),
    "Historical": historical_var(returns["total"]),
    "Age-weighted": age_weighted_historical_var(returns["total"]),
}
for name, value in var_methods.items():
    print(f"{name:14s} 1-day 99% VaR: {value:.2%}   10-day: {scale_var_horizon(value, 10):.2%}")

# %%
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
plot_var_comparison(var_methods, ax=ax)
ax.set_title("1-day 99% VaR by method")
plt.show()

# %% [markdown]
# The spread between the three is the interesting part. Historical VaR is the
# highest because the 20-year sample contains crises far more severe than
# current conditions, and it weights them as heavily as last week. The
# age-weighted estimate is the lowest, because it discounts those distant
# crises in favour of the recent, calmer regime. Parametric sits between them.
#
# None of the three is "correct" on its own — they answer slightly different
# questions, and the gap between them is itself a measure of how much the
# answer depends on that choice.

# %% [markdown]
# ## 5. Stand-alone component risk
#
# Setting the exposure vector to $[1, 0]$ or $[0, 1]$ isolates each risk factor.

# %%
components = {
    "Equity only": parametric_var([1, 0], cov),
    "Forex only": parametric_var([0, 1], cov),
    "Combined": parametric_var([1, 1], cov),
}
for name, value in components.items():
    print(f"{name:14s}: {value:.2%}")

undiversified = components["Equity only"] + components["Forex only"]
benefit = undiversified - components["Combined"]
print(f"\nSum of stand-alone: {undiversified:.2%}")
print(f"Diversification benefit: {benefit:.2%}")
