"""Publication-grade visualization engine for Value at Risk backtest diagnostics
and 4D macroeconomic stress tensors.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stress_engine.monte_carlo import build_parametric_grid_arrays

if TYPE_CHECKING:
    from stress_engine.portfolio import PortfolioVaR

# Apply professional publication styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    }
)

DEFAULT_RESULTS_PATH = "data/raw/synthetic_4d_monte_carlo_results.parquet"


def plot_var_backtest_diagnostics(
    portfolio: "PortfolioVaR",
    lookback_window: int = 252,
    output_filename: str | Path | None = None,
    output_dir: str | Path = "data/output",
) -> list[Path]:
    """Generates one diagnostic figure per risk model (five in total) for a portfolio.

    Plotted returns are the portfolio return series, aligned to the same out-of-sample
    dates as each model's VaR threshold.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_portfolio_name = portfolio.name.split(" - ")[-1].lower().replace(" ", "_")
    exported_paths: list[Path] = []

    models_config = {
        "rolling_historical": {
            "title": f"Rolling Historical Simulation ({lookback_window}-Day)",
            "color": "#e67e22",
            "runner": lambda: portfolio.run_rolling_out_of_sample_backtest(
                lookback_window=lookback_window, method="historical"
            ),
        },
        "rolling_parametric": {
            "title": "Rolling Parametric Normal",
            "color": "#d35400",
            "runner": lambda: portfolio.run_rolling_out_of_sample_backtest(
                lookback_window=lookback_window, method="parametric"
            ),
        },
        "dynamic_ewma": {
            "title": "Dynamic EWMA (lambda=0.94)",
            "color": "#2980b9",
            "runner": lambda: portfolio.run_ewma_out_of_sample_backtest(
                lookback_window=lookback_window, decay_factor=0.94
            ),
        },
        "gjr_garch": {
            "title": "Dynamic GJR-GARCH(1,1) Asymmetric Leverage",
            "color": "#8e44ad",
            "runner": lambda: portfolio.run_gjr_garch_out_of_sample_backtest(
                lookback_window=lookback_window
            ),
        },
        "filtered_historical": {
            "title": "Filtered Historical Simulation (FHS)",
            "color": "#27ae60",
            "runner": lambda: portfolio.run_fhs_out_of_sample_backtest(
                lookback_window=lookback_window
            ),
        },
    }

    for model_key, config in models_config.items():
        bt_res, realized_returns, var_series = config["runner"]()

        # Runners set portfolio_returns; the out-of-sample tail has the same length as var_series
        assert portfolio.portfolio_returns is not None
        n_oos = len(var_series)
        out_of_sample_dates = portfolio.returns.index[-n_oos:]
        returns = np.asarray(realized_returns, dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        ax.plot(
            out_of_sample_dates,
            returns * 100,
            color="#7f8c8d",
            alpha=0.6,
            lw=1.0,
            label="Daily Returns (%)",
        )
        ax.plot(
            out_of_sample_dates,
            var_series * 100,
            color=config["color"],
            lw=1.5,
            label=f"{config['title']} VaR (95%)",
        )

        breach_mask = returns < var_series
        if np.any(breach_mask):
            ax.scatter(
                out_of_sample_dates[breach_mask],
                returns[breach_mask] * 100,
                color="#c0392b",
                s=30,
                zorder=5,
                label=f"Exceptions ({bt_res.total_exceptions} / {bt_res.total_observations})",
            )

        ax.set_title(
            f"{portfolio.name} - {config['title']}\n"
            f"Kupiec POF p-value: {bt_res.kupiec_p_value:.4f} | "
            f"Christoffersen p-value: {bt_res.christoffersen_p_value:.4f}",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.set_xlabel("Trading Date", fontsize=10)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

        plt.tight_layout()
        filename = out_dir / f"var_diagnostics_{clean_portfolio_name}_{model_key}.png"
        fig.savefig(filename, format="png", bbox_inches="tight")
        plt.close(fig)

        exported_paths.append(filename)
        print(f"[+] Saved individual model diagnostic: {filename}")

    return exported_paths


def plot_monte_carlo_drawdown_surface(
    results: pd.DataFrame | None = None,
    output_filename: str | Path = "data/plots/stress_matrix_4d_surface.png",
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    """Plots distress probability over the volatility x shock-severity plane.

    The saved Monte Carlo results store tier labels (for example 'Vol-High'), not the
    numeric values. The numeric volatility and shock for each node come from
    build_parametric_grid_arrays(), which is the same grid the simulation used.
    Tail tier and asymmetry are averaged out, and the title says so.
    """
    if results is None:
        results = pd.read_parquet(results_path)

    required = {"Institution_Key", "Distress_Probability"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Monte Carlo results are missing columns: {sorted(missing)}")

    grid_meta, _, _, _, _ = build_parametric_grid_arrays()
    numeric = grid_meta[["Institution_Key", "annual_vol", "base_shock"]]

    merged = results.merge(numeric, on="Institution_Key", how="inner", validate="one_to_one")
    if len(merged) != len(results):
        raise ValueError(
            f"Only {len(merged)} of {len(results)} result rows matched the parameter grid"
        )

    surface = (
        merged.groupby(["base_shock", "annual_vol"])["Distress_Probability"]
        .mean()
        .unstack("annual_vol")
        .sort_index()
    )
    vol_vals = surface.columns.to_numpy(dtype=np.float64)
    shock_vals = surface.index.to_numpy(dtype=np.float64)
    vols, shocks = np.meshgrid(vol_vals, shock_vals)
    distress = surface.to_numpy(dtype=np.float64)

    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(projection="3d")

    surf = ax.plot_surface(
        vols, shocks, distress, cmap="viridis", edgecolor="none", alpha=0.85
    )
    ax.set_title(
        "4D Monte Carlo Macroeconomic Stress Matrix\n"
        "(averaged over tail tier and asymmetry)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Annualized Volatility (sigma)", fontsize=10)
    ax.set_ylabel("Exogenous Shock Severity", fontsize=10)
    ax.set_zlabel("Distress Probability (DD <= -40%)", fontsize=10)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Probability")

    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved Monte Carlo Drawdown Surface: {path}")
    return path
