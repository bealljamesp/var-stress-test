"""Conditional volatility engines: EWMA (RiskMetrics) and GJR-GARCH(1,1)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

# The optimizer works on returns in percent. Daily variances in decimal units are
# around 1e-4, which puts omega far from alpha, gamma, and beta in parameter space and
# makes SLSQP fail. In percent units all four parameters are O(0.01)-O(1).
_PCT = 100.0


@dataclass(frozen=True, slots=True)
class GJRGARCHParams:
    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    persistence: float


def compute_ewma_volatility(
    returns: npt.NDArray[np.float64],
    decay_factor: float = 0.94,
    initial_variance: float | None = None,
) -> npt.NDArray[np.float64]:
    """Calculates dynamic conditional volatility using the RiskMetrics EWMA filter.

    Output index t is the forecast for returns[t], built only from returns through t-1.
    """
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)
    n = returns_arr.size
    variance = np.empty(n, dtype=np.float64)

    init_var = (
        float(np.var(returns_arr[:30], ddof=1))
        if initial_variance is None
        else initial_variance
    )
    variance[0] = init_var

    # 1D recurrence filter: isolated vector operation
    r_sq = returns_arr**2
    one_minus_lambda = 1.0 - decay_factor

    for t in range(1, n):
        variance[t] = decay_factor * variance[t - 1] + one_minus_lambda * r_sq[t - 1]

    return np.sqrt(variance)


def _gjr_garch_likelihood_recursion(
    params: npt.NDArray[np.float64],
    returns_sq: npt.NDArray[np.float64],
    is_negative: npt.NDArray[np.float64],
    initial_var: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Recursive negative log-likelihood. variance[t] is the forecast for returns[t]."""
    omega, alpha, gamma, beta = params
    n = returns_sq.size
    variance = np.empty(n, dtype=np.float64)
    variance[0] = initial_var

    for t in range(1, n):
        variance[t] = (
            omega
            + (alpha + gamma * is_negative[t - 1]) * returns_sq[t - 1]
            + beta * variance[t - 1]
        )
        variance[t] = max(1e-8, variance[t])

    # Gaussian log-likelihood sum: 0.5 * sum(ln(sigma^2) + r^2 / sigma^2)
    nll = 0.5 * np.sum(np.log(variance) + returns_sq / variance)
    return float(nll), variance


def fit_gjr_garch(
    returns: npt.NDArray[np.float64],
) -> tuple[GJRGARCHParams, npt.NDArray[np.float64]]:
    """Fits GJR-GARCH(1,1) via Maximum Likelihood Estimation (MLE).

    The optimizer runs on percent returns from two starting points. The better
    converged fit is kept. Returned parameters, log-likelihood, and conditional
    volatility are in decimal units.

    Raises:
        RuntimeError: if neither starting point converges. A failed fit is never
            returned as if it were a model.
    """
    r = np.ascontiguousarray(returns, dtype=np.float64)
    is_neg = (r < 0.0).astype(np.float64)

    r_pct = r * _PCT
    r_sq_pct = r_pct**2
    sample_var_pct = float(np.var(r_pct, ddof=1))

    # Parameters in percent units: [omega, alpha, gamma, beta]
    bounds = [(1e-9, None), (1e-6, 1.0), (0.0, 1.0), (1e-6, 1.0)]

    # Stationarity constraint: alpha + beta + 0.5 * gamma < 1.0
    constraints = {
        "type": "ineq",
        "fun": lambda p: 0.9999 - (p[1] + p[3] + 0.5 * p[2]),
    }

    starts = (
        np.array([sample_var_pct * 0.05, 0.05, 0.10, 0.80], dtype=np.float64),
        np.array([sample_var_pct * 0.10, 0.10, 0.05, 0.85], dtype=np.float64),
    )

    def loss(p: npt.NDArray[np.float64]) -> float:
        nll, _ = _gjr_garch_likelihood_recursion(p, r_sq_pct, is_neg, sample_var_pct)
        return nll

    best = None
    messages: list[str] = []
    for x0 in starts:
        res = minimize(
            loss,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        messages.append(str(res.message))
        if res.success and np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res

    if best is None:
        raise RuntimeError(
            f"GJR-GARCH MLE failed to converge on {r.size} observations from both "
            f"starting points: {' | '.join(messages)}"
        )

    # Convert the fitted parameters back to decimal units and evaluate the likelihood
    # on the original returns, so the reported values are in decimal units.
    final_dec = np.array(
        [best.x[0] / _PCT**2, best.x[1], best.x[2], best.x[3]], dtype=np.float64
    )
    sample_var = float(np.var(r, ddof=1))
    nll_dec, cond_var = _gjr_garch_likelihood_recursion(
        final_dec, r**2, is_neg, sample_var
    )
    persistence = float(final_dec[1] + final_dec[3] + 0.5 * final_dec[2])

    result_params = GJRGARCHParams(
        omega=float(final_dec[0]),
        alpha=float(final_dec[1]),
        gamma=float(final_dec[2]),
        beta=float(final_dec[3]),
        log_likelihood=float(-nll_dec),
        persistence=persistence,
    )
    return result_params, np.sqrt(cond_var)
