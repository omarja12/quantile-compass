"""
Quantile Compass - interactive VaR explorer.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the package importable when the app runs straight from a checkout with
# no install - which is what Streamlit Community Cloud does.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quantile_compass.data import DEFAULT_DATA_PATH, prepare_dataset  # noqa: E402
from quantile_compass.fetch_data import build_dataset  # noqa: E402
from quantile_compass.returns import (  # noqa: E402
    PortfolioSpec,
    compute_log_returns,
    decompose_portfolio_returns,
)
from quantile_compass.var import (  # noqa: E402
    age_weighted_historical_var,
    historical_var,
    parametric_var,
    scale_var_horizon,
)
from quantile_compass.volatility import (  # noqa: E402
    annualize_volatility,
    covariance_matrix,
    ewma_correlation,
    ewma_volatility,
)

st.set_page_config(page_title="Quantile Compass", page_icon="📉", layout="wide")

CRISES = {
    "Global financial crisis": ("2008-09-01", "2009-03-31"),
    "Ruble crisis": ("2014-11-01", "2015-02-28"),
    "COVID-19": ("2020-02-19", "2020-04-30"),
    "Russia-Ukraine": ("2022-02-24", "2022-04-30"),
}


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    """
    Load the dataset, fetching it first if this environment doesn't have one.

    The repo ships the fetch script rather than the data, so a fresh deploy
    (Streamlit Community Cloud included) starts with nothing on disk.
    """
    if not Path(DEFAULT_DATA_PATH).exists():
        with st.spinner("Fetching market data - first run only, this takes a moment..."):
            frame = build_dataset()
            DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(DEFAULT_DATA_PATH)
    return prepare_dataset()


@st.cache_data(show_spinner=False)
def compute(
    lam: float, w_us: float, beta_us: float, beta_de: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_prices()
    spec = PortfolioSpec(w_us=w_us, w_de=round(1 - w_us, 10), beta_us=beta_us, beta_de=beta_de)
    returns = decompose_portfolio_returns(prices, spec=spec)
    frame = pd.DataFrame(
        {
            "equity_vol": annualize_volatility(ewma_volatility(returns["equity"], lam=lam)),
            "forex_vol": annualize_volatility(ewma_volatility(returns["forex"], lam=lam)),
            "correlation": ewma_correlation(returns["equity"], returns["forex"], lam=lam),
        }
    )
    return returns, frame


def shade(fig: go.Figure) -> None:
    for start, end in CRISES.values():
        fig.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.15, line_width=0)


st.title("Quantile Compass")
st.caption("Value-at-Risk for a multi-currency equity portfolio, held by a ruble-based investor.")

with st.sidebar:
    st.header("VaR settings")
    st.caption("These change the risk numbers above, not the volatility and correlation charts.")
    confidence = st.slider(
        "Confidence level (%)",
        90.0,
        99.9,
        99.0,
        step=0.1,
        help="How far into the loss tail to look. Moves the VaR figures and the "
        "threshold lines on the return distribution - it cannot change how "
        "volatile the market actually was.",
    )
    alpha = 1 - confidence / 100
    horizon = st.slider(
        "Horizon (trading days)",
        1,
        20,
        1,
        help="Scales the 1-day VaR by the square root of time. Affects the VaR figures only.",
    )

    st.divider()
    st.header("Model")
    st.caption("Changes the volatility and correlation estimates, and so the VaR too.")
    lam = st.slider(
        "EWMA decay (λ)",
        0.80,
        0.99,
        0.94,
        step=0.01,
        help="How fast the volatility estimate forgets. Lower reacts to shocks faster "
        "and is spikier; higher is smoother and slower. Try 0.80 vs 0.99 and watch "
        "the volatility chart change shape.",
    )

    st.divider()
    st.header("Portfolio")
    st.caption("Re-runs the whole decomposition, so everything on the page responds.")
    w_us = st.slider("US equity weight", 0.0, 1.0, 0.6, step=0.05)
    beta_us = st.slider("US beta", 0.5, 2.5, 1.6, step=0.1)
    beta_de = st.slider("German beta", 0.5, 2.5, 1.3, step=0.1)
    st.caption(f"German equity weight: {1 - w_us:.2f}")

try:
    returns, frame = compute(lam, w_us, beta_us, beta_de)
except Exception as exc:  # noqa: BLE001 - surface any data problem to the user, not a traceback
    st.error(
        "Could not load the market data. The data source may be rate-limiting or "
        "temporarily unavailable - try reloading in a minute.\n\n"
        "Running locally? Fetch the dataset directly with "
        "`python -m quantile_compass.fetch_data`."
    )
    st.caption(f"Details: {type(exc).__name__}: {exc}")
    st.stop()

cov = covariance_matrix(returns["equity"], returns["forex"], lam=lam)
var_param = scale_var_horizon(parametric_var([1, 1], cov, alpha=alpha), horizon)
var_hist = scale_var_horizon(historical_var(returns["total"], alpha=alpha), horizon)
var_aged = scale_var_horizon(
    age_weighted_historical_var(returns["total"], alpha=alpha, lam=lam), horizon
)

st.subheader(f"{horizon}-day VaR at {confidence:.1f}% confidence")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Parametric", f"{var_param * 100:.2f}%")
c2.metric("Historical", f"{var_hist * 100:.2f}%")
c3.metric("Age-weighted", f"{var_aged * 100:.2f}%")
c4.metric("Observations", f"{len(returns):,}")

st.caption(
    "Parametric assumes normally distributed returns and reacts fast to recent shocks through "
    "the EWMA covariance. Historical makes no distributional assumption but weights all history "
    "equally. Age-weighted sits between the two."
)

tab_vol, tab_corr, tab_dist, tab_factors = st.tabs(
    ["Volatility", "Correlation", "Return distribution", "Risk factors"]
)

with tab_vol:
    fig = go.Figure()
    fig.add_scatter(
        x=frame.index,
        y=frame["equity_vol"] * 100,
        name="Equity",
        line=dict(color="#2563eb", width=1.3),
    )
    fig.add_scatter(
        x=frame.index,
        y=frame["forex_vol"] * 100,
        name="Forex",
        line=dict(color="#db2777", width=1.3),
    )
    shade(fig)
    fig.update_layout(
        yaxis_title="EWMA annualised volatility (%)", height=460, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Estimated with λ = {lam:.2f}. Responds to the decay and portfolio controls; "
        "the confidence level and horizon do not affect it."
    )

with tab_corr:
    fig = go.Figure()
    fig.add_scatter(
        x=frame.index,
        y=frame["correlation"],
        name="Correlation",
        line=dict(color="#d97706", width=1.3),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    shade(fig)
    fig.update_layout(
        yaxis_title="EWMA equity-forex correlation",
        yaxis_range=[-1, 1],
        height=460,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Estimated with λ = {lam:.2f}. Negative correlation means currency depreciation offsets "
        "equity losses - a natural hedge for a ruble-based investor holding foreign equities."
    )

with tab_dist:
    fig = go.Figure()
    fig.add_histogram(x=returns["total"] * 100, nbinsx=120, marker_color="#059669")
    for value, name, color in [
        (-var_param / (horizon**0.5) * 100, "Parametric", "#2563eb"),
        (-var_hist / (horizon**0.5) * 100, "Historical", "#db2777"),
    ]:
        fig.add_vline(x=value, line_color=color, line_dash="dash", annotation_text=name)
    fig.update_layout(xaxis_title="Daily portfolio return (%)", yaxis_title="Frequency", height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Dashed lines mark the 1-day VaR thresholds at {confidence:.1f}% confidence - raise the "
        "confidence level and watch them move further into the tail."
    )

with tab_factors:
    prices = load_prices()
    extras = [c for c in ["SP500", "VIX", "GOLD"] if c in prices.columns]
    if extras:
        choice = st.selectbox(
            "Compare against",
            extras,
            help="A market series outside the portfolio, to put its risk in context.",
        )
        factor_vol = annualize_volatility(
            ewma_volatility(compute_log_returns(prices[choice]).dropna(), lam=lam)
        )

        fig = go.Figure()
        fig.add_scatter(
            x=frame.index,
            y=frame["equity_vol"] * 100,
            name="Portfolio equity",
            line=dict(color="#2563eb", width=1.3),
        )
        fig.add_scatter(
            x=factor_vol.index,
            y=factor_vol * 100,
            name=choice,
            line=dict(color="#0891b2", width=1.3),
        )
        shade(fig)
        fig.update_layout(
            yaxis_title="EWMA annualised volatility (%)", height=430, hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{choice} volatility against the portfolio's equity leg, both estimated with "
            f"λ = {lam:.2f} - so this chart responds to the decay slider too."
        )
        if choice == "VIX":
            st.caption(
                "Note that the VIX is itself a volatility index, so this line is the volatility "
                "*of* implied volatility - expect it to sit much higher than the others."
            )

        with st.expander(f"{choice} price level"):
            fig2 = go.Figure()
            fig2.add_scatter(
                x=prices.index, y=prices[choice], name=choice, line=dict(color="#0891b2", width=1.2)
            )
            shade(fig2)
            fig2.update_layout(yaxis_title=choice, height=380, hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("The raw series, for reference - price levels don't depend on the model.")
    else:
        st.info("No supplementary series in the current dataset.")
