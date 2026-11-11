"""Business-impact metrics for the presentation.

For each portfolio and risk model, reports:
  - breach count versus the number expected at 95% confidence
  - the dollar amount by which breach-day losses exceeded the model's warning line,
    on a $1,000,000 book (total and average per breach)

Returns are simple returns, so a loss is the negative of the return. The warning line
is the model's VaR threshold, converted the same way.

Run from the project root (needs network access to download prices):

    python scripts/impact_metrics.py
"""

import numpy as np

from stress_engine.portfolio import PortfolioVaR

CAPITAL = 1_000_000.0
LOOKBACK = 252
CONFIDENCE = 0.95


def portfolios() -> list[PortfolioVaR]:
    return [
        PortfolioVaR(
            name="Concentrated Financials",
            tickers=["C", "BAC", "XLF", "GS"],
            weights=[0.25, 0.25, 0.25, 0.25],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=CAPITAL,
            confidence_level=CONFIDENCE,
        ),
        PortfolioVaR(
            name="Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.40, 0.30, 0.20, 0.10],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=CAPITAL,
            confidence_level=CONFIDENCE,
        ),
    ]


def excess_loss_summary(
    realized_log: np.ndarray, threshold_log: np.ndarray
) -> tuple[int, float, float]:
    """Returns (breach count, total excess $ beyond the line, average excess $ per breach).

    Inputs are simple returns (the engine's default). Loss = -return on a $1 book.
    """
    loss = -realized_log
    line = -threshold_log
    breach = realized_log < threshold_log
    excess = (loss - line)[breach] * CAPITAL
    if excess.size == 0:
        return 0, 0.0, 0.0
    return int(breach.sum()), float(excess.sum()), float(excess.mean())


def main() -> None:
    expected = (1.0 - CONFIDENCE) * 502
    print(f"Expected breaches over 502 out-of-sample days: {expected:.1f}")
    print(f"Book size: ${CAPITAL:,.0f}\n")

    for port in portfolios():
        port.run_analysis()
        print(f"=== {port.name} ===")

        runs = {
            "Rolling Historical": port.run_rolling_out_of_sample_backtest(
                lookback_window=LOOKBACK, method="historical"
            ),
            "Dynamic EWMA": port.run_ewma_out_of_sample_backtest(
                lookback_window=LOOKBACK, decay_factor=0.94
            ),
            "Dynamic GJR-GARCH": port.run_gjr_garch_out_of_sample_backtest(
                lookback_window=LOOKBACK
            ),
        }

        for model, (_, realized, thresholds) in runs.items():
            count, total, avg = excess_loss_summary(realized, thresholds)
            print(
                f"  {model:20s} breaches={count:3d} (expected {expected:.1f})  "
                f"losses beyond line: total ${total:,.0f}, avg ${avg:,.0f} per breach"
            )
        print()


if __name__ == "__main__":
    main()
