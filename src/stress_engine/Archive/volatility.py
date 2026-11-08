"""Conditional volatility engines: EWMA (RiskMetrics) and GJR-GARCH(1,1)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize


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
    """Calculates dynamic conditional volatility using the RiskMetrics EWMA filter."""
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
    """Internal JIT/C-compatible recursive negative log-likelihood calculator."""
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
    """Fits GJR-GARCH(1,1) via Maximum Likelihood Estimation (MLE) and returns parameters & conditional volatility."""
    r = np.ascontiguousarray(returns, dtype=np.float64)
    r_sq = r**2
    is_neg = (r < 0.0).astype(np.float64)
    sample_var = float(np.var(r, ddof=1))

    # Initial parameter vector: [omega, alpha, gamma, beta]
    init_params = np.array([sample_var * 0.05, 0.05, 0.10, 0.80], dtype=np.float64)

    # Parameter constraints: omega > 0, alpha >= 0, gamma >= 0, beta >= 0
    bounds = [(1e-7, None), (1e-6, 1.0), (0.0, 1.0), (1e-6, 1.0)]

    # Stationarity constraint: alpha + beta + 0.5 * gamma < 1.0
    def stationarity_constraint(p: npt.NDArray[np.float64]) -> float:
        return 0.9999 - (p[1] + p[3] + 0.5 * p[2])

    constraints = {"type": "ineq", "fun": stationarity_constraint}

    def loss(p: npt.NDArray[np.float64]) -> float:
        nll, _ = _gjr_garch_likelihood_recursion(p, r_sq, is_neg, sample_var)
        return nll

    opt_res = minimize(
        loss,
        init_params,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if not opt_res.success:
        # Fallback to initial guess if optimization encounters numerical failure
        final_params = init_params
    else:
        final_params = opt_res.x

    _, cond_var = _gjr_garch_likelihood_recursion(
        final_params, r_sq, is_neg, sample_var
    )
    persistence = float(final_params[1] + final_params[3] + 0.5 * final_params[2])

    result_params = GJRGARCHParams(
        omega=float(final_params[0]),
        alpha=float(final_params[1]),
        gamma=float(final_params[2]),
        beta=float(final_params[3]),
        log_likelihood=float(-opt_res.fun),
        persistence=persistence,
    )
    return result_params, np.sqrt(cond_var)
