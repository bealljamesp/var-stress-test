"""Diagnose the GJR-GARCH log-likelihood gap against the `arch` package.

For each window, evaluates the in-house likelihood at both our parameters and
arch's parameters, using both our initial variance (sample variance) and arch's
initial variance (backcast). This separates three possible causes:

1. Likelihood formula: if our likelihood at arch's parameters and arch's initial
   variance matches arch's reported log-likelihood, the formulas agree.
2. Initial variance: if the gap closes when both models use the same initial
   variance, the difference is the starting convention.
3. Optimizer: if our likelihood at arch's parameters is higher than at our own
   parameters (with the same initial variance), our optimizer missed the optimum.

Run from the project root (needs network access to download prices):

    python scripts/diagnose_gjr_likelihood.py
"""

import numpy as np
from arch import arch_model

from stress_engine.portfolio import PortfolioVaR
from stress_engine.volatility import _gjr_garch_likelihood_recursion, fit_gjr_garch

WINDOW = 252
STARTS = (252, 504)
LOG_2PI = np.log(2.0 * np.pi)


def full_loglik(
    params: np.ndarray,
    r_sq: np.ndarray,
    is_neg: np.ndarray,
    initial_var: float,
    n: int,
) -> float:
    """Gaussian log-likelihood in decimal units, including the -0.5 n log(2 pi) constant."""
    nll, _ = _gjr_garch_likelihood_recursion(params, r_sq, is_neg, initial_var)
    return -nll - 0.5 * n * LOG_2PI


def main() -> None:
    port = PortfolioVaR(
        name="diagnostic",
        tickers=["SPY", "TLT", "GLD", "DBC"],
        weights=[0.40, 0.30, 0.20, 0.10],
        start_date="2007-01-01",
        end_date="2009-12-31",
    )
    port.run_analysis()
    returns = port.portfolio_returns
    assert returns is not None

    for start in STARTS:
        window = returns[start : start + WINDOW]
        n = window.size
        r_sq = window**2
        is_neg = (window < 0.0).astype(np.float64)

        ours, _ = fit_gjr_garch(window)
        our_params = np.array([ours.omega, ours.alpha, ours.gamma, ours.beta])
        our_init = float(np.var(window, ddof=1))

        res = arch_model(
            window * 100.0, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal"
        ).fit(disp="off")
        arch_params = np.array(
            [
                float(res.params["omega"]) / 100.0**2,
                float(res.params["alpha[1]"]),
                float(res.params["gamma[1]"]),
                float(res.params["beta[1]"]),
            ]
        )
        arch_init = float(np.asarray(res.conditional_volatility)[0]) ** 2 / 100.0**2
        arch_reported = float(res.loglikelihood) + n * np.log(100.0)

        print(f"--- window starting at day {start} ({n} obs) ---")
        print(f"  arch reported log-likelihood (decimal units): {arch_reported:.3f}")
        print(
            "  arch params, arch init var  -> ours formula: "
            f"{full_loglik(arch_params, r_sq, is_neg, arch_init, n):.3f}"
            "  (should match arch reported if formulas agree)"
        )
        print(
            "  our params,  our init var   -> ours formula: "
            f"{full_loglik(our_params, r_sq, is_neg, our_init, n):.3f}"
        )
        print(
            "  arch params, our init var   -> ours formula: "
            f"{full_loglik(arch_params, r_sq, is_neg, our_init, n):.3f}"
        )
        print(
            "  our params,  arch init var  -> ours formula: "
            f"{full_loglik(our_params, r_sq, is_neg, arch_init, n):.3f}"
        )
        print()

    print(
        "Reading: if the first line matches arch's reported value, the formulas agree.\n"
        "If 'arch params, our init var' is higher than 'our params, our init var',\n"
        "the optimizer missed the optimum. If the init-variance choice moves the\n"
        "log-likelihood by about the same amount as the gap, the gap is convention."
    )


if __name__ == "__main__":
    main()
