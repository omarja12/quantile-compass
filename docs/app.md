# Interactive app

The Streamlit app lets you change the parameters and watch the estimates move,
rather than reading fixed numbers off a page.

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

The app fetches the dataset itself on first run if there isn't one on disk, so
there's no separate setup step. (Running `python -m quantile_compass.fetch_data`
beforehand just gets it out of the way sooner.)

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

## Deploying it

The app is ready for [Streamlit Community Cloud](https://share.streamlit.io) as-is:

1. Sign in with GitHub and pick **New app** → this repository.
2. Set the main file path to `app/streamlit_app.py`, branch `main`.
3. Deploy.

Two things make that work without further setup. The root `requirements.txt` holds
only the app's runtime dependencies, so the deploy doesn't drag in the test, notebook
and docs toolchains. And the app puts `src/` on `sys.path` itself, so it imports
`quantile_compass` correctly even though nothing pip-installs the package there.

!!! note "First load is slower"
    A fresh deploy has no dataset, so the first visit spends a moment fetching ~20
    years of daily history before anything renders. It is cached from then on, until
    the container restarts.
