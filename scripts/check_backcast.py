"""Check whether arch's starting variance matches the backcast formula it documents.

arch's backcast is a weighted average of the first min(75, n) squared returns,
with weights 0.94**k normalized to sum to one. This script compares that formula
with the starting variance arch actually uses (conditional_volatility[0] squared)
for the two GJR windows used in the diagnostic.

Run from the project root (needs network access to download prices):

    python scripts/check_backcast.py
"""

import numpy as np
from arch import arch_model

from stress_engine.portfolio import PortfolioVaR

WINDOW = 252
STARTS = (0, 252, 504)


def weighted_backcast(returns_pct: np.ndarray) -> float:
    tau = min(75, returns_pct.size)
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(returns_pct[:tau] ** 2 * w))


def main() -> None:
    port = PortfolioVaR(
        name="backcast-check",
        tickers=["SPY", "TLT", "GLD", "DBC"],
        weights=[0.40, 0.30, 0.20, 0.10],
        start_date="2007-01-01",
        end_date="2009-12-31",
    )
    port.run_analysis()
    returns = port.portfolio_returns
    assert returns is not None

    for start in STARTS:
        window_pct = returns[start : start + WINDOW] * 100.0
        res = arch_model(
            window_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal"
        ).fit(disp="off")

        arch_init_pct = float(np.asarray(res.conditional_volatility)[0]) ** 2
        formula_pct = weighted_backcast(window_pct)
        sample_pct = float(np.var(window_pct, ddof=1))

        print(f"--- window starting at day {start} ---")
        print(f"  arch starting variance (pct^2):      {arch_init_pct:.6f}")
        print(f"  weighted backcast formula (pct^2):   {formula_pct:.6f}")
        print(f"  sample variance (pct^2):             {sample_pct:.6f}")
        print(
            "  formula matches arch:               "
            f"{np.isclose(arch_init_pct, formula_pct, rtol=1e-6)}"
        )
        print()


if __name__ == "__main__":
    main()
