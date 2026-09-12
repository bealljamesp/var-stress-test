"""Publication-grade visualization suite for VaR backtesting and Monte Carlo stress testing."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from stress_engine.portfolio import PortfolioVaR


def get_project_root() -> Path:
    """Returns absolute path to the repository root directory."""
    return Path(__file__).resolve().parents[2]


def apply_institutional_style() -> None:
    """Configures global aesthetic formatting for academic and institutional figures."""
    sns.set_theme(style="ticks", palette="deep")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_var_backtest_diagnostics(
    portfolio: PortfolioVaR,
    lookback_window: int = 252,
    output_filename: str = "var_backtest_diagnostics.png",
) -> None:
    """
    Plots realized portfolio returns against dynamic VaR threshold bands,
    highlighting exception breaches to illustrate volatility clustering.
    """
    path_obj = Path(output_filename)
    if "data" in path_obj.parts:
        save_path = path_obj
    else:
        save_path = Path("data/output") / output_filename

    save_path.parent.mkdir(parents=True, exist_ok=True)
    apply_institutional_style()

    # Execute dynamic backtests
    hist_bt, realized_rets, hist_thresholds = (
        portfolio.run_rolling_out_of_sample_backtest(
            lookback_window=lookback_window, method="historical"
        )
    )
    ewma_bt, _, ewma_thresholds = portfolio.run_ewma_out_of_sample_backtest(
        lookback_window=lookback_window, decay_factor=0.94
    )
    fhs_bt, _, fhs_thresholds = portfolio.run_fhs_out_of_sample_backtest(
        lookback_window=lookback_window
    )

    # Align dates with out-of-sample slices
    assert portfolio.returns is not None
    dates = portfolio.returns.index[lookback_window:]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle(
        f"Dynamic Out-of-Sample VaR (95%) Backtesting: {portfolio.name}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    models = [
        ("Rolling Historical (252-Day Window)", hist_thresholds, hist_bt, "darkorange"),
        ("Dynamic EWMA (RiskMetrics lambda=0.94)", ewma_thresholds, ewma_bt, "navy"),
        (
            "Filtered Historical Simulation (GJR-GARCH Tail)",
            fhs_thresholds,
            fhs_bt,
            "forestgreen",
        ),
    ]

    for ax, (title, thresholds, bt, color) in zip(axes, models):
        # 1. Realized returns
        ax.plot(
            dates,
            realized_rets,
            color="gray",
            alpha=0.5,
            linewidth=0.8,
            label="Realized Returns",
        )

        # 2. VaR Cutoff Boundary Band
        ax.plot(
            dates,
            thresholds,
            color=color,
            linewidth=1.5,
            label=f"95% VaR Threshold ({title.split()[0]})",
        )

        # 3. Highlight Breaches
        breach_mask = realized_rets < thresholds
        breach_dates = dates[breach_mask]
        breach_values = realized_rets[breach_mask]

        ax.scatter(
            breach_dates,
            breach_values,
            color="crimson",
            s=22,
            zorder=5,
            label=f"Breaches ({bt.total_exceptions} / {bt.total_observations} = {bt.empirical_rate * 100:.1f}%)",
        )

        ax.set_title(
            f"{title} | Kupiec POF p={bt.kupiec_p_value:.4f} | Christoffersen Indep p={bt.christoffersen_p_value:.4f}",
            fontsize=10,
            fontweight="semibold",
            loc="left",
        )
        ax.set_ylabel("Daily Return / VaR")
        ax.legend(loc="lower left", frameon=True, fontsize=8)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Backtest Diagnostics plot: {save_path}")
    return save_path


def plot_monte_carlo_drawdown_surface(
    parquet_path: Path | None = None,
    output_filename: str = "stress_matrix_4d_surface.png",
) -> Path:
    """Builds a faceted matrix contrasting tail thickness, volatility regimes,

    and asymmetric leverage shock responses.
    """
    apply_institutional_style()
    root = get_project_root()
    source_file = (
        parquet_path
        if parquet_path is not None
        else root / "data" / "raw" / "synthetic_4d_monte_carlo_results.parquet"
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Parquet source not found: {source_file}. Run monte_carlo.py first."
        )

    df_results = pd.read_parquet(source_file)

    shock_order = ["Shock-Mild", "Shock-Mod", "Shock-Sev"]
    tail_order = ["Tail-Fat", "Tail-Norm", "Tail-Thin"]
    vol_order = ["Vol-Low", "Vol-Norm", "Vol-High"]

    df_results["Shock_Severity"] = pd.Categorical(
        df_results["Shock_Severity"], categories=shock_order, ordered=True
    )
    df_results["Tail_Tier"] = pd.Categorical(
        df_results["Tail_Tier"], categories=tail_order, ordered=True
    )
    df_results["Volatility"] = pd.Categorical(
        df_results["Volatility"], categories=vol_order, ordered=True
    )

    g = sns.catplot(
        data=df_results,
        x="Shock_Severity",
        y="Distress_Probability",
        hue="Asymmetry",
        col="Tail_Tier",
        row="Volatility",
        kind="bar",
        height=2.8,
        aspect=1.2,
        palette="crest",
        sharey=True,
    )

    g.set_axis_labels("Shock Severity Tier", "Probability of Distress (DD <= -40%)")
    g.set_titles(col_template="Tail: {col_name}", row_template="Vol: {row_name}")
    g.add_legend(title="GJR Asymmetry")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(
        "4D Stress Matrix: Macroeconomic Shock Sensitivity Surface",
        fontsize=13,
        fontweight="bold",
    )

    output_dir = root / "data" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / output_filename

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Monte Carlo Drawdown Surface: {save_path}")
    return save_path


if __name__ == "__main__":
    # Generate Monte Carlo Surface directly from stored parquet artifact
    plot_monte_carlo_drawdown_surface()
