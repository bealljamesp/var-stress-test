"""Main entrypoint for historical VaR analysis and stress testing pipelines."""

from stress_engine.portfolio import PortfolioVaR, export_descriptive_statistics


def main() -> None:
    portfolios = [
        PortfolioVaR(
            name="2008 Crisis - Concentrated Financials",
            tickers=["C", "BAC", "XLF", "GS"],
            weights=[0.25, 0.25, 0.25, 0.25],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
        PortfolioVaR(
            name="2008 Crisis - Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.40, 0.30, 0.20, 0.10],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
    ]

    # Generate Week 10 Descriptive Analysis Table CSV for Excel
    export_descriptive_statistics(portfolios)

    for port in portfolios:
        print("\n" + "=" * 65)
        print(f"PORTFOLIO: {port.name}")
        print(f"HORIZON:   {port.start_date} to {port.end_date}")
        print("=" * 65)

        metrics = port.run_analysis()
        print(f"Confidence Level:            {port.confidence_level * 100:.1f}%")
        print(
            f"  In-Sample Hist VaR:        {metrics['hist_pct'] * 100:.2f}%  (${metrics['hist_dollar']:,.2f})"
        )
        print(
            f"  In-Sample Param VaR:       {metrics['param_pct'] * 100:.2f}%  (${metrics['param_dollar']:,.2f})"
        )

        # 252-day Rolling Out-of-Sample Backtests
        hist_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=252, method="historical"
        )
        param_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=252, method="parametric"
        )

        print("-" * 65)
        print(
            "ROLLING OUT-OF-SAMPLE BACKTEST (252-Day Window, 502 Out-of-Sample Days):"
        )
        print("  1. Rolling Historical Simulation:")
        print(
            f"     - Breaches:             {hist_bt.total_exceptions} / {hist_bt.total_observations} ({hist_bt.empirical_rate * 100:.2f}%)"
        )
        print(
            f"     - Kupiec POF:           LR={hist_bt.kupiec_stat:.3f} (p={hist_bt.kupiec_p_value:.4f}) -> {'REJECT H0' if hist_bt.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"     - Christoffersen Indep: LR={hist_bt.christoffersen_stat:.3f} (p={hist_bt.christoffersen_p_value:.4f}) -> {'REJECT H0' if hist_bt.christoffersen_reject else 'ACCEPT H0'}"
        )

        print("  2. Rolling Parametric (Normal):")
        print(
            f"     - Breaches:             {param_bt.total_exceptions} / {param_bt.total_observations} ({param_bt.empirical_rate * 100:.2f}%)"
        )
        print(
            f"     - Kupiec POF:           LR={param_bt.kupiec_stat:.3f} (p={param_bt.kupiec_p_value:.4f}) -> {'REJECT H0' if param_bt.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"     - Christoffersen Indep: LR={param_bt.christoffersen_stat:.3f} (p={param_bt.christoffersen_p_value:.4f}) -> {'REJECT H0' if param_bt.christoffersen_reject else 'ACCEPT H0'}"
        )

        # 3. EWMA Dynamic Volatility Out-of-Sample Backtest
        ewma_bt, _, _ = port.run_ewma_out_of_sample_backtest(
            lookback_window=252, decay_factor=0.94
        )

        print("  3. Dynamic EWMA (RiskMetrics lambda=0.94):")
        print(
            f"     - Breaches:             {ewma_bt.total_exceptions} / {ewma_bt.total_observations} ({ewma_bt.empirical_rate * 100:.2f}%)"
        )
        print(
            f"     - Kupiec POF:           LR={ewma_bt.kupiec_stat:.3f} (p={ewma_bt.kupiec_p_value:.4f}) -> {'REJECT H0' if ewma_bt.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"     - Christoffersen Indep: LR={ewma_bt.christoffersen_stat:.3f} (p={ewma_bt.christoffersen_p_value:.4f}) -> {'REJECT H0' if ewma_bt.christoffersen_reject else 'ACCEPT H0'}"
        )

        # 4. GJR-GARCH(1,1) Asymmetric Dynamic Out-of-Sample Backtest
        garch_bt, _, _ = port.run_gjr_garch_out_of_sample_backtest(lookback_window=252)

        print("  4. Dynamic GJR-GARCH(1,1) (Asymmetric Leverage):")
        print(
            f"     - Breaches:             {garch_bt.total_exceptions} / {garch_bt.total_observations} ({garch_bt.empirical_rate * 100:.2f}%)"
        )
        print(
            f"     - Kupiec POF:           LR={garch_bt.kupiec_stat:.3f} (p={garch_bt.kupiec_p_value:.4f}) -> {'REJECT H0' if garch_bt.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"     - Christoffersen Indep: LR={garch_bt.christoffersen_stat:.3f} (p={garch_bt.christoffersen_p_value:.4f}) -> {'REJECT H0' if garch_bt.christoffersen_reject else 'ACCEPT H0'}"
        )

        # 5. Filtered Historical Simulation (FHS - GJR-GARCH + Empirical Tail)
        fhs_bt, _, _ = port.run_fhs_out_of_sample_backtest(lookback_window=252)

        print("  5. Filtered Historical Simulation (FHS / GJR-GARCH Tail):")
        print(
            f"     - Breaches:             {fhs_bt.total_exceptions} / {fhs_bt.total_observations} ({fhs_bt.empirical_rate * 100:.2f}%)"
        )
        print(
            f"     - Kupiec POF:           LR={fhs_bt.kupiec_stat:.3f} (p={fhs_bt.kupiec_p_value:.4f}) -> {'REJECT H0' if fhs_bt.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"     - Christoffersen Indep: LR={fhs_bt.christoffersen_stat:.3f} (p={fhs_bt.christoffersen_p_value:.4f}) -> {'REJECT H0' if fhs_bt.christoffersen_reject else 'ACCEPT H0'}"
        )
        # Generate Publication-Grade Diagnostic Charts
        from stress_engine.visualization import plot_var_backtest_diagnostics

        clean_name = port.name.split(" - ")[-1].lower().replace(" ", "_")
        plot_var_backtest_diagnostics(
            portfolio=port,
            lookback_window=252,
            output_filename=f"data/output/var_diagnostics_{clean_name}.png",
        )

    # Generate 4D Monte Carlo Drawdown Surface Chart
    from stress_engine.visualization import plot_monte_carlo_drawdown_surface

    plot_monte_carlo_drawdown_surface()


if __name__ == "__main__":
    main()
