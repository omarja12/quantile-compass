# Interactive app

The Streamlit app lets you change the parameters and watch the estimates move,
rather than reading fixed numbers off a page.

```bash
pip install -e ".[app]"
python -m quantile_compass.fetch_data   # if you haven't already
streamlit run app/streamlit_app.py
```

## What you can change

| Control | Effect |
|---|---|
| **Confidence level** | The VaR quantile, 90% to 99.9%. |
| **EWMA decay (λ)** | How fast the volatility estimate forgets. Lower λ reacts faster to shocks and is noisier; higher λ is smoother and slower. |
| **Horizon** | Scales the 1-day figure by √h. |
| **Portfolio weights and betas** | Re-runs the whole decomposition — the split between equity and currency risk shifts with them. |

## What it shows

- **Volatility** — the two EWMA series with crisis windows shaded.
- **Correlation** — the equity/forex correlation through time.
- **Return distribution** — the histogram with the parametric and historical VaR
  thresholds drawn on it, so you can see where each method puts the cut-off.
- **Risk factors** — the supplementary series (S&P 500, VIX, gold) for context.

Things worth trying: drop λ to 0.80 and watch the volatility series get much
choppier; push the confidence level to 99.9% and see the parametric and historical
estimates diverge further as the normal assumption strains in the deep tail.
