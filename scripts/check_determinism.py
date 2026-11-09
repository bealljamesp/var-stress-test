"""Determinism check for the GJR-GARCH backtest.

Step 1: downloads the multi-asset portfolio's returns once and saves them to
data/raw/multi_asset_returns.npy. Later runs load that file instead of
downloading again.

Step 2: runs the GJR-GARCH backtest three times on the same saved returns and
prints the breach count and a fingerprint of the threshold series for each run.

Reading the output:
  - Counts and fingerprints identical across the three runs: the fit is
    deterministic. If the count still differs from main.py or impact_metrics.py,
    the difference comes from the price data those scripts downloaded.
  - Counts or fingerprints differ between runs on the same saved data: the fit
    itself is not deterministic. That needs fixing before any GJR number is reported.

Run from the project root:

    python scripts/check_determinism.py

To re-download, delete data/raw/multi_asset_returns.npy first.
"""

import hashlib
from pathlib import Path

import numpy as np

from stress_engine.portfolio import PortfolioVaR

CACHE = Path("data/raw/multi_asset_returns.npy")
LOOKBACK = 252


def make_portfolio() -> PortfolioVaR:
    return PortfolioVaR(
        name="Multi-Asset Diverse",
        tickers=["SPY", "TLT", "GLD", "DBC"],
        weights=[0.40, 0.30, 0.20, 0.10],
        start_date="2007-01-01",
        end_date="2009-12-31",
    )


def load_returns() -> np.ndarray:
    if CACHE.exists():
        returns = np.load(CACHE)
        print(f"Loaded cached returns from {CACHE} ({returns.size} days)")
    else:
        port = make_portfolio()
        port.run_analysis()
        assert port.portfolio_returns is not None
        returns = np.ascontiguousarray(port.portfolio_returns, dtype=np.float64)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, returns)
        print(f"Downloaded returns and saved to {CACHE} ({returns.size} days)")
    fingerprint = hashlib.sha256(returns.tobytes()).hexdigest()[:12]
    print(f"Returns fingerprint: {fingerprint}\n")
    return returns


def main() -> None:
    returns = load_returns()

    for run in range(1, 4):
        port = make_portfolio()
        port.portfolio_returns = returns
        bt, _, thresholds = port.run_gjr_garch_out_of_sample_backtest(
            lookback_window=LOOKBACK
        )
        thr_fp = hashlib.sha256(np.ascontiguousarray(thresholds).tobytes()).hexdigest()[
            :12
        ]
        print(
            f"run {run}: breaches={bt.total_exceptions}  threshold fingerprint={thr_fp}"
        )


if __name__ == "__main__":
    main()
