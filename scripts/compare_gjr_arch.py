"""Cross-check the in-house GJR-GARCH(1,1) fit against the `arch` package.

Fits both models on the same trailing windows of the 4-asset portfolio used in
main.py and prints parameters, log-likelihoods, and conditional volatility
agreement. Both use a zero mean and Gaussian innovations.

Run from the project root (needs network access to download prices):

    python compare_gjr_arch.py
"""

import numpy as np
from arch import arch_model

from stress_engine.portfolio import PortfolioVaR
from stress_engine.volatility import fit_gjr_garch

WINDOW = 252
STARTS = (0, 252, 504)


def main() -> None:
    port = PortfolioVaR(
        name="cross-check",
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

        ours, ours_sigma = fit_gjr_garch(window)

        # arch works in percent returns; scale back to decimal units for comparison.
        am = arch_model(
            window * 100.0, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal"
        )
        res = am.fit(disp="off")
        arch_omega = res.params["omega"] / 100.0**2
        arch_alpha = res.params["alpha[1]"]
        arch_gamma = res.params["gamma[1]"]
        arch_beta = res.params["beta[1]"]
        arch_ll_dec = res.loglikelihood + n * np.log(100.0)
        arch_sigma = np.asarray(res.conditional_volatility) / 100.0

        corr = float(np.corrcoef(ours_sigma, arch_sigma)[0, 1])
        mad = float(np.mean(np.abs(ours_sigma - arch_sigma)))
        rel = mad / float(np.mean(arch_sigma))

        print(f"--- window starting at day {start} ({n} obs) ---")
        print(
            f"  omega  ours={ours.omega:.4e}  arch={arch_omega:.4e}\n"
            f"  alpha  ours={ours.alpha:.4f}  arch={arch_alpha:.4f}\n"
            f"  gamma  ours={ours.gamma:.4f}  arch={arch_gamma:.4f}\n"
            f"  beta   ours={ours.beta:.4f}  arch={arch_beta:.4f}"
        )
        print(f"  loglik ours={ours.log_likelihood:.3f}  arch={arch_ll_dec:.3f}")
        print(
            f"  sigma  corr={corr:.5f}  mean|diff|={mad:.3e}  relative diff={rel:.2%}"
        )
        print()

    print(
        "Expected: sigma correlation near 1 and small relative differences. "
        "Parameters can differ more than the sigma paths do, because the "
        "likelihood is flat near the optimum. The initial variance also "
        "differs (ours: sample variance; arch: backcast), which affects the "
        "first few observations of each window."
    )


if __name__ == "__main__":
    main()
