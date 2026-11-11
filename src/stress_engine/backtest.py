"""Quantitative risk backtesting engine: Kupiec POF and Christoffersen independence tests."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats


@dataclass(frozen=True, slots=True)
class BacktestResult:
    total_observations: int
    total_exceptions: int
    empirical_rate: float
    target_rate: float
    kupiec_stat: float
    kupiec_p_value: float
    kupiec_reject: bool
    christoffersen_stat: float
    christoffersen_p_value: float
    christoffersen_reject: bool
    combined_stat: float
    combined_p_value: float
    combined_reject: bool


def calculate_var_exceptions(
    returns: npt.NDArray[np.float64],
    var_thresholds: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.bool_]:
    """Evaluates VaR exceptions where loss breaches the modeled VaR cutoff.

    Assumes returns and VaR are expressed in the same sign convention (losses are negative).
    """
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)
    var_arr = np.ascontiguousarray(var_thresholds, dtype=np.float64)
    return returns_arr < var_arr


def kupiec_pof_test(
    exceptions: npt.NDArray[np.bool_],
    confidence_level: float = 0.95,
    critical_p_value: float = 0.05,
) -> tuple[float, float, bool]:
    """Kupiec Unconditional Coverage (POF) Likelihood Ratio Test.

    Tests H0: empirical exception rate == target nominal rate (1 - confidence_level).
    """
    n: int = exceptions.size
    x: int = int(np.sum(exceptions))

    if n == 0:
        raise ValueError("Exceptions array cannot be empty.")

    p: float = 1.0 - confidence_level
    p_hat: float = x / n

    if x == 0:
        # Edge case: 0 breaches in n trials
        lr_stat = -2.0 * (n * np.log(1.0 - p))
    elif x == n:
        # Edge case: all trials breached
        lr_stat = -2.0 * (n * np.log(p))
    else:
        # Standard Kupiec LR formula
        num = (1.0 - p) ** (n - x) * (p**x)
        denom = (1.0 - p_hat) ** (n - x) * (p_hat**x)
        lr_stat = -2.0 * np.log(num / denom)

    p_val = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    reject_null = p_val < critical_p_value
    return float(lr_stat), p_val, reject_null


def christoffersen_independence_test(
    exceptions: npt.NDArray[np.bool_],
    critical_p_value: float = 0.05,
) -> tuple[float, float, bool]:
    """Christoffersen Independence Likelihood Ratio Test.

    Tests H0: Exceptions are independent over time (no exception clustering).
    """
    exc_arr = np.ascontiguousarray(exceptions, dtype=np.int8)
    if exc_arr.size < 2:
        return 0.0, 1.0, False

    # Shift arrays to construct first-order Markov transition pairs: (t-1, t)
    prior_state = exc_arr[:-1]
    curr_state = exc_arr[1:]

    # Count transitions without python loops via vectorized masks
    t00 = int(np.sum((prior_state == 0) & (curr_state == 0)))
    t01 = int(np.sum((prior_state == 0) & (curr_state == 1)))
    t10 = int(np.sum((prior_state == 1) & (curr_state == 0)))
    t11 = int(np.sum((prior_state == 1) & (curr_state == 1)))

    total_prior_0 = t00 + t01
    total_prior_1 = t10 + t11

    if total_prior_0 == 0 or total_prior_1 == 0:
        # If no transitions occur from one state, independence cannot be evaluated
        return 0.0, 1.0, False

    pi_01 = t01 / total_prior_0 if total_prior_0 > 0 else 0.0
    pi_11 = t11 / total_prior_1 if total_prior_1 > 0 else 0.0
    pi = (t01 + t11) / (total_prior_0 + total_prior_1)

    # Edge cases: no exceptions observed in transitions
    if pi_01 == 0.0 and pi_11 == 0.0:
        return 0.0, 1.0, False

    # Log-likelihood under Null (Independence: pi_01 == pi_11 == pi)
    log_l_null = (t00 + t10) * np.log(1.0 - pi + 1e-12) + (t01 + t11) * np.log(
        pi + 1e-12
    )

    # Log-likelihood under Alternative (Clustering: pi_01 != pi_11)
    log_l_alt = (
        t00 * np.log(1.0 - pi_01 + 1e-12)
        + t01 * np.log(pi_01 + 1e-12)
        + t10 * np.log(1.0 - pi_11 + 1e-12)
        + t11 * np.log(pi_11 + 1e-12)
    )

    lr_stat = -2.0 * (log_l_null - log_l_alt)
    lr_stat = max(0.0, float(lr_stat))

    p_val = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    reject_null = p_val < critical_p_value
    return lr_stat, p_val, reject_null


def run_full_var_backtest(
    returns: npt.NDArray[np.float64],
    var_thresholds: npt.NDArray[np.float64] | float,
    confidence_level: float = 0.95,
    critical_p_value: float = 0.05,
) -> BacktestResult:
    """Executes simultaneous Unconditional, Independence, and Combined Conditional Coverage tests."""
    exceptions = calculate_var_exceptions(returns, var_thresholds)
    total_obs = exceptions.size
    total_exc = int(np.sum(exceptions))

    lr_uc, p_uc, rej_uc = kupiec_pof_test(
        exceptions, confidence_level=confidence_level, critical_p_value=critical_p_value
    )
    lr_ind, p_ind, rej_ind = christoffersen_independence_test(
        exceptions, critical_p_value=critical_p_value
    )

    # Combined conditional coverage: LR_cc = LR_uc + LR_ind ~ chi2(df=2)
    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    rej_cc = p_cc < critical_p_value

    return BacktestResult(
        total_observations=total_obs,
        total_exceptions=total_exc,
        empirical_rate=total_exc / total_obs if total_obs > 0 else 0.0,
        target_rate=1.0 - confidence_level,
        kupiec_stat=lr_uc,
        kupiec_p_value=p_uc,
        kupiec_reject=rej_uc,
        christoffersen_stat=lr_ind,
        christoffersen_p_value=p_ind,
        christoffersen_reject=rej_ind,
        combined_stat=lr_cc,
        combined_p_value=p_cc,
        combined_reject=rej_cc,
    )


# Basel traffic light evaluation functions and dataclasses for regulatory capital multipliers based on VaR backtesting results.

from dataclasses import dataclass
from enum import StrEnum


class BaselZone(StrEnum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True, slots=True)
class BaselTrafficLightResult:
    total_observations: int
    exceptions: int
    zone: BaselZone
    capital_multiplier: float
    cumulative_probability: float


def compute_expected_shortfall(
    returns: npt.NDArray[np.float64],
    var_threshold: float,
) -> float:
    """Calculates empirical Expected Shortfall (CVaR) as a positive loss percentage

    given a negative return VaR threshold (losses beyond cutoff).
    """
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)
    # Mask breaches: returns that fall below negative cutoff
    breaches = returns_arr[returns_arr < var_threshold]

    if breaches.size == 0:
        # If no breaches occurred, ES defaults to the VaR threshold magnitude
        return max(0.0, -var_threshold)

    # Return positive mean loss magnitude
    return float(np.mean(-breaches))


def evaluate_basel_traffic_light(
    exceptions: int,
    total_observations: int = 250,
    confidence_level: float = 0.99,
) -> BaselTrafficLightResult:
    """Classifies a model into the BCBS Basel Traffic Light zones (Green, Yellow, Red).

    The zone cutoffs and capital multipliers are the BCBS table for 250 observations
    at 99% 1-day VaR. Other sample sizes or confidence levels raise an error, because
    the table does not apply to them. The cumulative probability is the binomial CDF
    for the same setting.
    """
    if total_observations != 250 or not np.isclose(confidence_level, 0.99):
        raise ValueError(
            "The BCBS traffic-light table applies only to 250 observations at 99% VaR "
            f"(got {total_observations} observations at {confidence_level:.2%})."
        )
    p = 1.0 - confidence_level
    cum_prob = float(stats.binom.cdf(exceptions, total_observations, p))

    # BCBS standard boundaries for N=250 at 99% VaR:
    # 0-4 breaches: Green (multiplier = 3.00)
    # 5-9 breaches: Yellow (multiplier = 3.40 to 3.85)
    # 10+ breaches: Red (multiplier = 4.00)
    if exceptions <= 4:
        zone = BaselZone.GREEN
        multiplier = 3.00
    elif exceptions <= 9:
        zone = BaselZone.YELLOW
        # Graduated penalty scaling between 3.40 and 3.85
        penalties = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
        multiplier = penalties.get(exceptions, 3.85)
    else:
        zone = BaselZone.RED
        multiplier = 4.00

    return BaselTrafficLightResult(
        total_observations=total_observations,
        exceptions=exceptions,
        zone=zone,
        capital_multiplier=multiplier,
        cumulative_probability=cum_prob,
    )
