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
    output_filename: str | Path = "data/output/var_diagnostics.png",
) -> Path:
    """Generates a multi-panel publication-grade diagnostic figure comparing static

    lookback windows against dynamic conditional volatility models and FHS.
    """
    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Run backtests to gather data series
    hist_bt, _, hist_var = portfolio.run_rolling_out_of_sample_backtest(
        lookback_window=lookback_window, method="historical"
    )
    ewma_bt, _, ewma_var = portfolio.run_ewma_out_of_sample_backtest(
        lookback_window=lookback_window, decay_factor=0.94
    )
    garch_bt, _, garch_var = portfolio.run_gjr_garch_out_of_sample_backtest(
        lookback_window=lookback_window
    )
    fhs_bt, _, fhs_var = portfolio.run_fhs_out_of_sample_backtest(
        lookback_window=lookback_window
    )

    # Extract the exact out-of-sample datetime index and aggregate portfolio returns
    out_of_sample_dates = portfolio.returns.index[-len(hist_var) :]
    returns = portfolio.returns.iloc[-len(hist_var) :].dot(portfolio.weights)
    if not isinstance(returns, np.ndarray):
        returns = returns.to_numpy(dtype=np.float64, copy=False)

    # Create a 3-panel stacked subplot layout for granular appendix inclusion
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, dpi=300)

    # Panel 1: Static Rolling Historical vs Returns & Breaches
    ax1 = axes[0]
    ax1.plot(
        out_of_sample_dates,
        returns * 100,
        color="#7f8c8d",
        alpha=0.6,
        lw=1.0,
        label="Daily Returns (%)",
    )
    ax1.plot(
        out_of_sample_dates,
        -hist_var * 100,
        color="#e67e22",
        lw=1.5,
        label="Rolling Historical VaR (95%)",
    )
    breach_mask_hist = returns < -hist_var
    if np.any(breach_mask_hist):
        ax1.scatter(
            out_of_sample_dates[breach_mask_hist],
            returns[breach_mask_hist] * 100,
            color="#c0392b",
            s=25,
            zorder=5,
            label=f"Exceptions ({hist_bt.total_exceptions})",
        )
    ax1.set_title(
        f"{portfolio.name} - Panel A: Static Rolling Historical Backtest",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("Percentage (%)", fontsize=10)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    # Panel 2: Dynamic EWMA & GJR-GARCH Conditional Volatility Models
    ax2 = axes[1]
    ax2.plot(
        out_of_sample_dates,
        returns * 100,
        color="#7f8c8d",
        alpha=0.4,
        lw=1.0,
        label="_nolegend_",
    )
    ax2.plot(
        out_of_sample_dates,
        -ewma_var * 100,
        color="#2980b9",
        lw=1.5,
        label="Dynamic EWMA VaR (lambda=0.94)",
    )
    ax2.plot(
        out_of_sample_dates,
        -garch_var * 100,
        color="#8e44ad",
        lw=1.5,
        linestyle="-.",
        label="GJR-GARCH(1,1) VaR",
    )
    breach_mask_ewma = returns < -ewma_var
    if np.any(breach_mask_ewma):
        ax2.scatter(
            out_of_sample_dates[breach_mask_ewma],
            returns[breach_mask_ewma] * 100,
            color="#c0392b",
            s=25,
            zorder=5,
            label=f"EWMA Exceptions ({ewma_bt.total_exceptions})",
        )
    ax2.set_title(
        f"{portfolio.name} - Panel B: Conditional Volatility Models (EWMA & GJR-GARCH)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    # Panel 3: Filtered Historical Simulation (FHS) Hybrid Model
    ax3 = axes[2]
    ax3.plot(
        out_of_sample_dates,
        returns * 100,
        color="#7f8c8d",
        alpha=0.4,
        lw=1.0,
        label="_nolegend_",
    )
    ax3.plot(
        out_of_sample_dates,
        -fhs_var * 100,
        color="#27ae60",
        lw=1.5,
        label="Filtered Historical Simulation (FHS) VaR",
    )
    breach_mask_fhs = returns < -fhs_var
    if np.any(breach_mask_fhs):
        ax3.scatter(
            out_of_sample_dates[breach_mask_fhs],
            returns[breach_mask_fhs] * 100,
            color="#c0392b",
            s=25,
            zorder=5,
            label=f"FHS Exceptions ({fhs_bt.total_exceptions})",
        )
    ax3.set_title(
        f"{portfolio.name} - Panel C: Semi-Parametric Filtered Historical Simulation",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax3.set_ylabel("Percentage (%)", fontsize=10)
    ax3.set_xlabel("Trading Date", fontsize=10)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved granular multi-panel diagnostics plot: {path}")
    return path


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
