# Methodology

## Risk decomposition

The ruble value of a foreign holding is its local-currency price multiplied by the
exchange rate:

$$P^{RUB}_t = P^{USD}_t \times X^{USD/RUB}_t$$

Taking logs turns that product into a sum, which is what lets the portfolio's return
separate cleanly into an equity part and a currency part:

$$r_t = \underbrace{w_{de}\,\beta_{de}\, r^{DAX}_t + w_{us}\,\beta_{us}\, r^{NDX}_t}_{\text{equity}}
      + \underbrace{w_{de}\, r^{EUR/RUB}_t + w_{us}\, r^{USD/RUB}_t}_{\text{forex}}$$

| Symbol | Meaning | Default |
|---|---|---|
| \(w_{us}, w_{de}\) | Portfolio weights | 0.6 / 0.4 |
| \(\beta_{us}, \beta_{de}\) | Market betas of the holdings | 1.6 / 1.3 |
| \(r^{DAX}, r^{NDX}\) | Equity log returns | — |
| \(r^{EUR/RUB}, r^{USD/RUB}\) | Currency log returns | — |

Betas apply to the equity leg only. Currency exposure follows the size of each
holding, not its market sensitivity, so the forex leg is weighted but not levered.

!!! note "Why log returns"
    Log returns add across time and across the decomposition above. Simple returns do
    neither, which would leave the equity and forex components failing to sum to the
    total.

## EWMA volatility

Volatility is estimated with the RiskMetrics exponentially weighted moving average at
\(\lambda = 0.94\):

$$\hat\sigma^2_t = (1-\lambda)\, r^2_{t-1} + \lambda\, \hat\sigma^2_{t-1}$$

and covariance with the same recursion applied to the cross-product:

$$\hat\sigma_{ab,t} = (1-\lambda)\, r_{a,t-1} r_{b,t-1} + \lambda\, \hat\sigma_{ab,t-1}$$

Correlation is the covariance normalised by the two volatilities.

!!! warning "Seeding matters more than it looks"
    The recursion needs a starting value. Seeding it with the mean squared return over
    the *whole sample* — the usual textbook shortcut — leaks future information into
    every estimate in the series, including the earliest ones. This implementation
    seeds from a burn-in window (250 observations by default) and masks that window in
    the output, so every reported estimate depends only on returns that precede it.
    The test suite pins this behaviour explicitly.

Daily estimates are annualised by \(\sqrt{250}\).

## Value-at-Risk

All three estimators report VaR as a positive fraction of portfolio value, at a 1%
significance level (99% confidence) unless stated otherwise.

### Parametric

Assumes returns are normally distributed, and reads risk off the current EWMA
covariance matrix:

$$VaR_{h,\alpha} = \Phi^{-1}(1-\alpha)\,\sqrt{\theta' \Omega_h \theta}$$

The exposure vector \(\theta\) selects which risk factors are in play: \([1,1]\) for
the combined portfolio, \([1,0]\) for equity stand-alone, \([0,1]\) for currency
stand-alone. Because the covariance matrix is EWMA-based, this estimator responds
quickly to recent shocks.

### Historical

The empirical \(\alpha\)-quantile of realised returns, sign-flipped so a loss reads
positive. No distributional assumption at all — but every observation counts equally,
so a crisis twenty years ago weighs as much as last week.

### Age-weighted (hybrid)

After Boudoukh, Richardson and Whitelaw (1998). Each observation gets an
exponentially decaying probability weight, heaviest on the most recent:

$$\omega_T = 1-\lambda, \qquad \omega_{i-1} = \lambda\,\omega_i$$

Returns are sorted, the weights accumulated, and the quantile read off the weighted
cumulative distribution. This keeps the distribution-free character of historical
simulation while letting recent conditions dominate the tail.

### Backtesting

Any VaR estimate can be checked against what actually happened. Recomputing the
parametric VaR at every date from that date's EWMA covariance gives a forecast series;
a **breach** is a day whose realised loss exceeded the forecast made before it.

Because the EWMA recursion is seeded without look-ahead (above), those forecasts use
only prior information, which is what makes the comparison meaningful rather than
circular. At a confidence level of \(1-\alpha\), a well-specified model should breach
on about \(\alpha\) of days. See [Results](results.md#does-the-model-actually-work) for
what this sample shows.

### Horizon scaling

$$VaR_{h} = VaR_{1} \times \sqrt{h}$$

Square-root-of-time scaling assumes returns are i.i.d. — an assumption that
volatility clustering visibly violates, so multi-day figures should be read as a
convention rather than a forecast.

## Data handling

The free data feed needs two corrections before any of the above is meaningful.

**Sparse early history.** The RUB exchange-rate series are thin before mid-2006,
leaving multi-month holes in the joined index — a 448-day gap at May 2006, a 210-day
gap at February 2005. A "return" computed across a hole like that is not a daily
return. The usable sample therefore starts at 2006-05-16, from which coverage runs at
roughly 246 observations per year. The single remaining 26-day gap (August 2008) is
handled by invalidating the return that spans it rather than treating it as one day.

**Bad ticks.** Thin FX crosses occasionally carry erroneous single-day values — USD/RUB
prints 0.72 on 2016-01-06 and 5.00 on 2023-03-16, against a true level near 73 and 76,
and the identical value 110.674103 appears on two unrelated dates in 2022. These are
detected by their signature: a move beyond 50% that is substantially reversed the very
next day. Real market moves of that size persist; a value that round-trips in one day
is a data error. Flagged points are replaced by linear interpolation between their
neighbours.

!!! note "What is deliberately *not* cleaned"
    The VIX is excluded from the filter, because it genuinely can move 50–100% in a
    single session — the February 2018 "Volmageddon" spike is real, not a bad tick.
    The genuine 2022 ruble collapse (78 → 139 over two weeks) is likewise untouched,
    because it persisted rather than reverting.

## References

- Basel Committee on Banking Supervision (2019), [Minimum Capital Requirements for Market Risk](https://www.bis.org/bcbs/publ/d457.htm).
- J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th ed. — source of the EWMA recursion and the \(\lambda = 0.94\) convention.
- Boudoukh, J., Richardson, M. and Whitelaw, R. (1998), ["The Best of Both Worlds: A Hybrid Approach to Calculating Value at Risk"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=51420), *Risk*, 11(5), 64–67.
- Jorion, P. (2007), *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed., McGraw-Hill.
