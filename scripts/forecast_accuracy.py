"""Forecast accuracy against actual returns, for the presentation.

Part 1, worst day: for each model, the predicted loss line on the day with the
largest actual loss, compared with that loss. The line that sat closest to the
actual loss, without being exceeded by much, predicted best.

Part 2, pinball loss: the standard score for a VaR line. For a 95% line, each
day is scored: if the loss is worse than the line, the score is 0.95 times the
gap; if it is better, the score is 0.05 times the gap. Lower is better. The score
rewards lines that are not breached too often and that are not far from actual
losses. Scores are averaged over all 502 out-of-sample days and over the
crisis window (September 2008 to March 2009).

Returns are simple returns, shown in percentage terms.

Run from the project root (needs network access to download prices):

    python scripts/forecast_accuracy.py
"""

import numpy as np
import pandas as pd

from stress_engine.portfolio import PortfolioVaR

LOOKBACK = 252
CONFIDENCE = 0.95
TAU = 1.0 - CONFIDENCE  # 0.05 for the lower tail
CRISIS_START = "2008-09-01"
CRISIS_END = "2009-03-31"


def portfolios() -> list[PortfolioVaR]:
    return [
        PortfolioVaR(
            name="Concentrated Financials",
            tickers=["C", "BAC", "XLF", "GS"],
            weights=[0.25, 0.25, 0.25, 0.25],
            start_date="2007-01-01",
            end_date="2009-12-31",
        ),
        PortfolioVaR(
            name="Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.40, 0.30, 0.20, 0.10],
            start_date="2007-01-01",
            end_date="2009-12-31",
        ),
    ]


def pinball_loss(realized: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Quantile score for a lower-tail VaR line at level TAU, one value per day."""
    gap = realized - line
    return np.maximum(TAU * gap, (TAU - 1.0) * gap)


def to_pct(simple_return: float) -> float:
    return float(simple_return * 100.0)


def main() -> None:
    for port in portfolios():
        port.run_analysis()
        assert port.returns is not None and port.portfolio_returns is not None

        dates = port.returns.index[LOOKBACK:]
        realized = port.portfolio_returns[LOOKBACK:]

        _, _, hist_line = port.run_rolling_out_of_sample_backtest(
            lookback_window=LOOKBACK, method="historical"
        )
        _, _, ewma_line = port.run_ewma_out_of_sample_backtest(
            lookback_window=LOOKBACK, decay_factor=0.94
        )
        _, _, gjr_line = port.run_gjr_garch_out_of_sample_backtest(
            lookback_window=LOOKBACK
        )
        _, _, fhs_line = port.run_fhs_out_of_sample_backtest(lookback_window=LOOKBACK)

        models = {
            "Rolling historical": hist_line,
            "Dynamic EWMA": ewma_line,
            "Dynamic GJR-GARCH": gjr_line,
            "FHS": fhs_line,
        }

        print(f"=== {port.name} ===")

        # Part 1: worst day
        worst = int(np.argmin(realized))
        worst_date = pd.Timestamp(dates[worst]).date()
        print(f"Worst day in the test period: {worst_date}")
        print(f"  Actual loss that day: {to_pct(realized[worst]):.2f}%")
        for name, line in models.items():
            predicted = to_pct(line[worst])
            gap = abs(to_pct(realized[worst]) - predicted)
            print(
                f"  {name:20s} predicted worst loss {predicted:.2f}%  "
                f"(off by {gap:.2f} points)"
            )

        # Part 2: pinball loss, all days and crisis window
        in_crisis = (dates >= pd.Timestamp(CRISIS_START)) & (
            dates <= pd.Timestamp(CRISIS_END)
        )
        print("\nPinball loss (lower is better), in percentage points:")
        print(f"  {'Model':20s} {'All 502 days':>14s} {'Crisis window':>14s}")
        for name, line in models.items():
            losses = pinball_loss(realized, line) * 100.0
            print(
                f"  {name:20s} {losses.mean():14.4f} {losses[in_crisis].mean():14.4f}"
            )
        print()


if __name__ == "__main__":
    main()
