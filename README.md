<img src="docs/assets/logo-dark.png" alt="" width="84" align="left">

# Quantile Compass

[![CI](https://github.com/omarja12/quantile-compass/actions/workflows/ci.yml/badge.svg)](https://github.com/omarja12/quantile-compass/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

Value-at-Risk for a portfolio that carries two different kinds of risk at once.

A ruble-based investor holds a 60/40 portfolio of US (NASDAQ 100) and German (DAX)
equities. Because those holdings are priced in dollars and euros, the investor is
exposed both to the equity markets and to the currency they're quoted in — and in a
crisis those two exposures behave very differently.

This project decomposes the portfolio's ruble return into an equity component and a
currency component, tracks the volatility and correlation of each over twenty years
spanning four crises, and compares three ways of answering the same question: *how
much could this lose tomorrow?*

🎛️ **[Try the live app →](https://quantile-compass-n5y28ojjmzrnriql57pvvd.streamlit.app/)** &nbsp;·&nbsp; 📊 **[Documentation and results →](https://omarja12.github.io/quantile-compass/)**

---

## The headline result

**The two risks peak in different crises.**

![EWMA annualised volatility](docs/assets/ewma_volatility.png)

Equity volatility peaks in the 2008 crisis and again in the COVID crash (107.8% on
19 March 2020). Currency volatility barely reacts to either — then dominates in the
2014 ruble crisis and in 2022, hitting 124.2% on 31 March 2022, a level equity
volatility never reaches anywhere in the sample.

Looking only at total portfolio volatility, you'd see two spikes and conclude there's
one risk driver flaring up occasionally. Split the series and there are clearly two,
firing on different events.

**The three VaR methods disagree by more than three percentage points.**

![VaR by method](docs/assets/var_comparison.png)

| Method | 1-day 99% VaR | 10-day |
|---|---|---|
| Parametric | 4.11% | 13.00% |
| Historical | 5.96% | 18.85% |
| Age-weighted | 2.92% | 9.24% |

Same portfolio, same data, same confidence level. Historical is highest because a
twenty-year sample contains crises far worse than today and weights them equally;
age-weighted is lowest because it discounts them in favour of the recent calm regime.
The gap between them measures how much the answer depends on that choice rather than
on the data.

## Quick start

```bash
git clone https://github.com/omarja12/quantile-compass.git
cd quantile-compass
pip install -e ".[app,dev]"

python -m quantile_compass.fetch_data    # fetch the dataset
pytest                                    # run the test suite
streamlit run app/streamlit_app.py        # launch the interactive explorer
```

```python
from quantile_compass.data import prepare_dataset
from quantile_compass.returns import decompose_portfolio_returns
from quantile_compass.volatility import covariance_matrix
from quantile_compass.var import parametric_var, historical_var

prices = prepare_dataset()
returns = decompose_portfolio_returns(prices)

cov = covariance_matrix(returns["equity"], returns["forex"])
print(f"parametric: {parametric_var([1, 1], cov):.2%}")
print(f"historical: {historical_var(returns['total']):.2%}")
```

## What's in here

```
src/quantile_compass/
├── fetch_data.py    # pull daily history from Yahoo Finance, align across tickers
├── data.py          # trim the sparse early period, repair bad ticks
├── returns.py       # log returns, equity/forex decomposition
├── volatility.py    # EWMA variance, covariance, correlation
├── var.py           # parametric, historical and age-weighted VaR
└── viz.py           # shared plotting helpers

app/streamlit_app.py       # interactive explorer
notebooks/01_analysis.py   # the full walkthrough (jupytext-paired)
scripts/build_figures.py   # regenerate every figure in the docs
tests/                     # 39 tests covering the maths
docs/                      # MkDocs Material site
```

## Design notes

A few decisions worth calling out, because they're the difference between a
plausible-looking result and a correct one:

**The EWMA recursion is seeded without look-ahead.** Seeding it with the mean squared
return over the whole sample — the usual textbook shortcut — leaks future information
into every estimate in the series. This implementation seeds from a burn-in window
and masks that window in the output, so each estimate depends only on returns that
precede it. There's a test that fails if that property breaks.

**The data is cleaned, and the cleaning is defensible.** The free data feed carries
real bad ticks (USD/RUB printing 0.72 on 2016-01-06 against a true level near 73) and
multi-month holes in the sparse pre-2006 history. Both are handled explicitly: gaps
invalidate the returns that span them, and bad ticks are detected by their signature —
a move beyond 50% that reverses the very next day. Genuine extremes that *persist*,
like the 2022 ruble collapse, are deliberately left alone, and the VIX is exempt from
the filter entirely because it really does move 50–100% in a day sometimes.

**Figures and numbers are generated, not typed.** Everything in the docs comes out of
`scripts/build_figures.py` running against the actual dataset.

## Data

Daily data from Yahoo Finance via `yfinance`, fetched on demand rather than committed:

- **Core** — NASDAQ 100 (`^NDX`), DAX (`^GDAXI`), USD/RUB (`RUB=X`), EUR/RUB (`EURRUB=X`)
- **Supplementary** — S&P 500 (`^GSPC`), VIX (`^VIX`), gold futures (`GC=F`)

The usable sample runs from May 2006 (where the FX series become dense) to the present:
**4,988 daily returns over ~20 years**, covering the 2008 financial crisis, the 2014
ruble crisis, COVID-19, and the 2022 Russia-Ukraine crisis.

## References

- Basel Committee on Banking Supervision (2019), [Minimum Capital Requirements for Market Risk](https://www.bis.org/bcbs/publ/d457.htm)
- J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th ed.
- Boudoukh, J., Richardson, M. and Whitelaw, R. (1998), ["The Best of Both Worlds: A Hybrid Approach to Calculating Value at Risk"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=51420), *Risk*, 11(5), 64–67
- Jorion, P. (2007), *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed., McGraw-Hill

## License

[MIT](LICENSE)
