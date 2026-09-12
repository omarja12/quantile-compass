"""
Regenerate every figure used by the docs site and the README.

    python scripts/build_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantile_compass.data import prepare_dataset  # noqa: E402
from quantile_compass.returns import decompose_portfolio_returns  # noqa: E402
from quantile_compass.var import (  # noqa: E402
    age_weighted_historical_var,
    historical_var,
    parametric_var,
    scale_var_horizon,
)
from quantile_compass.viz import (  # noqa: E402
    plot_correlation,
    plot_price_levels,
    plot_return_distribution,
    plot_var_comparison,
    plot_volatility,
)
from quantile_compass.volatility import (  # noqa: E402
    annualize_volatility,
    covariance_matrix,
    ewma_correlation,
    ewma_volatility,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "assets"
DPI = 160

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    prices = prepare_dataset()
    returns = decompose_portfolio_returns(prices)
    print(
        f"sample: {returns.index.min().date()} -> {returns.index.max().date()} ({len(returns)} returns)"
    )

    vol_e = annualize_volatility(ewma_volatility(returns["equity"]))
    vol_f = annualize_volatility(ewma_volatility(returns["forex"]))
    corr = ewma_correlation(returns["equity"], returns["forex"])

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_price_levels(prices, ["NASDAQ100", "DAX"], ax=ax)
    save(fig, "price_levels.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_price_levels(prices, ["USD_RUB", "EUR_RUB"], ax=ax)
    save(fig, "fx_levels.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_volatility(vol_e, vol_f, ax=ax)
    save(fig, "ewma_volatility.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_correlation(corr, ax=ax)
    save(fig, "ewma_correlation.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_return_distribution(returns["total"], ax=ax)
    save(fig, "return_distribution.png")

    cov = covariance_matrix(returns["equity"], returns["forex"])
    var_methods = {
        "Parametric": parametric_var([1, 1], cov),
        "Historical": historical_var(returns["total"]),
        "Age-weighted": age_weighted_historical_var(returns["total"]),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    plot_var_comparison(var_methods, ax=ax)
    save(fig, "var_comparison.png")

    summary = {
        "sample_start": str(returns.index.min().date()),
        "sample_end": str(returns.index.max().date()),
        "n_returns": int(len(returns)),
        "equity_vol_mean_pct": round(float(vol_e.mean() * 100), 2),
        "equity_vol_max_pct": round(float(vol_e.max() * 100), 2),
        "equity_vol_max_date": str(vol_e.idxmax().date()),
        "forex_vol_mean_pct": round(float(vol_f.mean() * 100), 2),
        "forex_vol_max_pct": round(float(vol_f.max() * 100), 2),
        "forex_vol_max_date": str(vol_f.idxmax().date()),
        "corr_mean": round(float(corr.mean()), 3),
        "corr_min": round(float(corr.min()), 3),
        "corr_max": round(float(corr.max()), 3),
        "var_parametric_1d_pct": round(var_methods["Parametric"] * 100, 2),
        "var_parametric_10d_pct": round(scale_var_horizon(var_methods["Parametric"], 10) * 100, 2),
        "var_historical_1d_pct": round(var_methods["Historical"] * 100, 2),
        "var_age_weighted_1d_pct": round(var_methods["Age-weighted"] * 100, 2),
    }
    (FIG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
