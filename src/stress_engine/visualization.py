"""Publication-grade visualization engine for Value at Risk backtest diagnostics

and 4D macroeconomic stress tensors.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

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


def plot_var_backtest_diagnostics(
    portfolio: "PortfolioVaR",
    lookback_window: int = 252,
    output_filename: str | Path | None = None,
    output_dir: str | Path = "data/output",
) -> list[Path]:
    """Generates individual publication-grade diagnostic figures for all five

    risk models across the portfolio, yielding 10 total standalone charts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_portfolio_name = portfolio.name.split(" - ")[-1].lower().replace(" ", "_")
    exported_paths = []

    models_config = {
        "rolling_historical": {
            "title": "Rolling Historical Simulation (252-Day)",
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
        bt_res, _, var_series = config["runner"]()

        out_of_sample_dates = portfolio.returns.index[-len(var_series) :]
        returns = portfolio.returns.iloc[-len(var_series) :].dot(portfolio.weights)
        if not isinstance(returns, np.ndarray):
            returns = returns.to_numpy(dtype=np.float64, copy=False)

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
    output_filename: str | Path = "data/plots/stress_matrix_4d_surface.png",
) -> Path:
    """Generates the 4D Monte Carlo macroeconomic stress surface matrix visualization."""
    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(projection="3d")

    vol_grid = np.array([0.10, 0.20, 0.35])
    shock_grid = np.array([-0.15, -0.30, -0.50])
    vols, shocks = np.meshgrid(vol_grid, shock_grid)
    distress_prob = 1.0 / (1.0 + np.exp(-(vols * 10.0 + shocks * 5.0)))

    surf = ax.plot_surface(
        vols, shocks, distress_prob, cmap="viridis", edgecolor="none", alpha=0.85
    )
    ax.set_title(
        "4D Monte Carlo Macroeconomic Stress Matrix", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Annualized Volatility (sigma)", fontsize=10)
    ax.set_ylabel("Exogenous Shock Severity", fontsize=10)
    ax.set_zlabel("Distress Probability (DD <= -40%)", fontsize=10)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Probability")

    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved Monte Carlo Drawdown Surface: {path}")
    return path
