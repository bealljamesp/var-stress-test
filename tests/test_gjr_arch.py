"""Regression test: the in-house GJR-GARCH(1,1) fit matches the arch package.

Uses a fixed synthetic series generated from a known GJR process, so the test is
deterministic and runs offline. Skips if arch is not installed.
"""

import numpy as np
import pytest

from stress_engine.volatility import fit_gjr_garch

arch_model = pytest.importorskip("arch").arch_model

LOG_2PI = np.log(2.0 * np.pi)
N_OBS = 300
SEED = 2024


def _synthetic_gjr_returns(n: int = N_OBS, seed: int = SEED) -> np.ndarray:
    """Simulate decimal returns from a GJR-GARCH(1,1) process with fat-tailed innovations."""
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta = 2e-6, 0.03, 0.12, 0.85
    # Standardize Student-t(6) innovations to unit variance
    z = rng.standard_t(df=6, size=n) / np.sqrt(6.0 / 4.0)

    r = np.empty(n, dtype=np.float64)
    variance = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(n):
        r[t] = np.sqrt(variance) * z[t]
        variance = (
            omega + (alpha + gamma * float(r[t] < 0.0)) * r[t] ** 2 + beta * variance
        )
    return r


def test_gjr_fit_matches_arch_on_synthetic_series() -> None:
    r = _synthetic_gjr_returns()
    n = r.size

    ours, sigma_ours = fit_gjr_garch(r)

    res = arch_model(
        r * 100.0, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal"
    ).fit(disp="off")

    # Parameters: arch works in percent units, so convert omega back to decimal.
    arch_omega = float(res.params["omega"]) / 100.0**2
    arch_alpha = float(res.params["alpha[1]"])
    arch_gamma = float(res.params["gamma[1]"])
    arch_beta = float(res.params["beta[1]"])

    assert ours.omega == pytest.approx(arch_omega, rel=1e-2)
    assert ours.alpha == pytest.approx(arch_alpha, abs=1e-3)
    assert ours.gamma == pytest.approx(arch_gamma, abs=1e-3)
    assert ours.beta == pytest.approx(arch_beta, abs=1e-3)

    # Conditional volatility: the paths are what the VaR thresholds use.
    sigma_arch = np.asarray(res.conditional_volatility) / 100.0
    corr = float(np.corrcoef(sigma_ours, sigma_arch)[0, 1])
    max_rel_diff = float(np.max(np.abs(sigma_ours - sigma_arch) / sigma_arch))
    assert corr > 0.9999
    assert max_rel_diff < 1e-3

    # Log-likelihood on a common convention: arch includes the -0.5 n log(2 pi)
    # constant and the scaling term, so adjust both sides before comparing.
    arch_ll_dec = float(res.loglikelihood) + n * np.log(100.0)
    ours_ll_full = ours.log_likelihood - 0.5 * n * LOG_2PI
    assert ours_ll_full == pytest.approx(arch_ll_dec, abs=1e-2)
