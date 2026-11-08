"""Check arch's first conditional variance against the presample recursion.

arch's conditional variance for the first observation is computed from the
presample backcast b, not from b itself:

    sigma2_0 = omega + (alpha + 0.5 * gamma) * b + beta * b

with b the weighted average of the first min(75, n) squared returns (weights
0.94**k, normalized). This script evaluates that expression with arch's fitted
parameters and compares it with arch's conditional_volatility[0] squared.

Run from the project root (needs network access to download prices):

    python scripts/check_presample.py
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
        name="presample-check",
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

        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        gamma = float(res.params["gamma[1]"])
        beta = float(res.params["beta[1]"])
        b = weighted_backcast(window_pct)

        arch_sigma2_0 = float(np.asarray(res.conditional_volatility)[0]) ** 2
        presample = omega + (alpha + 0.5 * gamma) * b + beta * b
        plain_backcast = b

        print(f"--- window starting at day {start} ---")
        print(f"  arch sigma2_0 (pct^2):                       {arch_sigma2_0:.6f}")
        print(f"  omega + (alpha + gamma/2) b + beta b:        {presample:.6f}")
        print(f"  plain backcast b (pct^2):                    {plain_backcast:.6f}")
        print(
            "  presample expression matches arch:           "
            f"{np.isclose(arch_sigma2_0, presample, rtol=1e-6)}"
        )
        print()


if __name__ == "__main__":
    main()
