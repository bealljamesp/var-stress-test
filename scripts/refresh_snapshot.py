"""Re-download the committed price snapshots from Yahoo Finance.

Yahoo revises adjusted prices over time, so a refresh can change results. Run this only
when you want new inputs. Then:

  1. Run `python main.py` and check the printed input fingerprints.
  2. Commit the new snapshot files and the regenerated outputs together, in one commit.

Usage (needs network access):

    python scripts/refresh_snapshot.py
"""

from stress_engine.portfolio import PortfolioVaR

PORTFOLIO_SPECS = [
    ("2008 Crisis - Concentrated Financials", ["C", "BAC", "XLF", "GS"], [0.25, 0.25, 0.25, 0.25]),
    ("2008 Crisis - Multi-Asset Diverse", ["SPY", "TLT", "GLD", "DBC"], [0.40, 0.30, 0.20, 0.10]),
]


def main() -> None:
    for name, tickers, weights in PORTFOLIO_SPECS:
        port = PortfolioVaR(
            name=name,
            tickers=tickers,
            weights=weights,
            start_date="2007-01-01",
            end_date="2009-12-31",
        )
        port.download_data(refresh=True)
        print(f"{name}: input fingerprint {port.input_fingerprint()} -> {port.snapshot_path()}")


if __name__ == "__main__":
    main()
