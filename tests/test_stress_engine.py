"""Comprehensive mathematical and behavioral unit test suite for stress_engine."""

import numpy as np
import pytest

from stress_engine.backtest import (
    calculate_var_exceptions,
    christoffersen_independence_test,
    kupiec_pof_test,
    run_full_var_backtest,
)
from stress_engine.monte_carlo import run_comprehensive_stress_engine
from stress_engine.portfolio import PortfolioVaR, export_descriptive_statistics
from stress_engine.volatility import compute_ewma_volatility, fit_gjr_garch

# =====================================================================
# 1. REGULATORY BACKTESTING INVARIANTS (KUPIEC & CHRISTOFFERSEN)
# =====================================================================


def test_calculate_var_exceptions_boundary() -> None:
    returns = np.array([-0.05, -0.01, 0.02, -0.04, 0.01], dtype=np.float64)
    var = -0.03
    exceptions = calculate_var_exceptions(returns, var)
    expected = np.array([True, False, False, True, False], dtype=np.bool_)
    assert np.array_equal(exceptions, expected)


def test_kupiec_perfect_calibration() -> None:
    # 1,000 observations with exactly 50 breaches (5.0% empirical on 95% VaR)
    exceptions = np.zeros(1000, dtype=np.bool_)
    exceptions[:50] = True

    stat, p_val, reject = kupiec_pof_test(exceptions, confidence_level=0.95)
    assert np.isclose(stat, 0.0, atol=1e-2)
    assert p_val > 0.90
    assert not reject


def test_kupiec_severe_underestimation() -> None:
    # 1,000 observations with 150 breaches (15% empirical on 95% VaR)
    exceptions = np.zeros(1000, dtype=np.bool_)
    exceptions[:150] = True

    stat, p_val, reject = kupiec_pof_test(exceptions, confidence_level=0.95)
    assert stat > 20.0
    assert p_val < 0.001
    assert reject


def test_christoffersen_independence_clustering() -> None:
    # Heavy clustering: all 10 exceptions occur sequentially
    exceptions = np.zeros(200, dtype=np.bool_)
    exceptions[50:60] = True

    stat, p_val, reject = christoffersen_independence_test(exceptions)
    assert stat > 3.841  # Chi-squared critical threshold at alpha=0.05, df=1
    assert p_val < 0.05
    assert reject


def test_christoffersen_zero_exceptions_boundary() -> None:
    # Zero exceptions across observation horizon: cannot reject independence
    exceptions = np.zeros(250, dtype=np.bool_)
    stat, p_val, reject = christoffersen_independence_test(exceptions)
    assert stat == 0.0
    assert p_val == 1.0
    assert not reject


def test_full_backtest_dataclass_contract() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=500)
    var = -0.02
    result = run_full_var_backtest(returns, var, confidence_level=0.95)

    assert result.total_observations == 500
    assert 0.0 <= result.empirical_rate <= 1.0
    assert isinstance(result.kupiec_reject, bool)
    assert isinstance(result.christoffersen_reject, bool)
    assert isinstance(result.combined_reject, bool)


# =====================================================================
# 2. DYNAMIC VOLATILITY ENGINES (EWMA & GJR-GARCH)
# =====================================================================


def test_ewma_monotonic_decay_under_zero_shock() -> None:
    # A single shock followed by a zero-return regime must decay exponentially
    returns = np.zeros(100, dtype=np.float64)
    returns[0] = 0.10  # Initial 10% shock

    vol = compute_ewma_volatility(returns, decay_factor=0.94, initial_variance=0.01)

    # From t=2 onward (where r_{t-1}=0), variance = lambda * variance_{t-1}
    # Therefore, consecutive conditional sigma must be strictly decreasing
    diffs = np.diff(vol[2:])
    assert (diffs < 0.0).all()


def test_ewma_output_shape_and_positivity() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.02, size=300)
    vol = compute_ewma_volatility(returns, decay_factor=0.94)

    assert vol.shape == (300,)
    assert (vol > 0.0).all()


def test_gjr_garch_stationarity_and_parameter_bounds() -> None:
    rng = np.random.default_rng(42)
    # Generate synthetic stationary fat-tailed returns
    synthetic_returns = rng.standard_t(df=5.0, size=1000) * 0.015

    params, cond_sigma = fit_gjr_garch(synthetic_returns)

    # Positivity bounds
    assert params.omega > 0.0
    assert params.alpha >= 0.0
    assert params.gamma >= 0.0
    assert params.beta >= 0.0

    # Stationarity requirement: alpha + beta + 0.5 * gamma < 1.0
    assert params.persistence < 1.0
    assert len(cond_sigma) == len(synthetic_returns)
    assert (cond_sigma > 0.0).all()


# =====================================================================
# 3. 4D TENSOR MONTE CARLO INVARIANTS
# =====================================================================


def test_monte_carlo_tensor_geometry_and_bounds() -> None:
    # Lightweight smoke run: 100 paths, 10 days
    df_results, master_tensor = run_comprehensive_stress_engine(n_paths=100, n_days=10)

    # Assert Cartesian grid geometry: 3 x 3 x 3 x 3 = 81 nodes
    assert len(df_results) == 81
    assert master_tensor.shape == (81, 100, 10)
    assert master_tensor.dtype == np.float32

    # Drawdown probabilities must be bounded strictly within [0.0, 1.0]
    assert (df_results["Distress_Probability"] >= 0.0).all()
    assert (df_results["Distress_Probability"] <= 1.0).all()

    # Drawdown metrics must be negative or zero (peak-to-trough)
    assert (df_results["Mean_Max_DD"] <= 0.0).all()
    assert (df_results["P95_Max_DD"] <= 0.0).all()


# =====================================================================
# 4. PORTFOLIO ENGINE CONTRACTS & INPUT VALIDATION
# =====================================================================


def test_portfolio_weight_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Dimension mismatch"):
        PortfolioVaR(
            name="Mismatch Port",
            tickers=["AAPL", "MSFT"],
            weights=[0.5, 0.3, 0.2],  # 2 assets, 3 weights
            start_date="2020-01-01",
            end_date="2021-01-01",
        )


def test_portfolio_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        PortfolioVaR(
            name="Unnormalized Port",
            tickers=["AAPL", "MSFT"],
            weights=[0.5, 0.4],  # Sum = 0.9
            start_date="2020-01-01",
            end_date="2021-01-01",
        )


def test_portfolio_lookback_exceeding_data_length() -> None:
    port = PortfolioVaR(
        name="Short Horizon",
        tickers=["SPY"],
        weights=[1.0],
        start_date="2020-01-01",
        end_date="2020-02-01",  # ~21 trading days
    )
    with pytest.raises(ValueError, match="must exceed lookback window"):
        port.run_rolling_out_of_sample_backtest(lookback_window=252)


from stress_engine.backtest import (
    BaselZone,
    compute_expected_shortfall,
    evaluate_basel_traffic_light,
)


def test_expected_shortfall_strictly_exceeds_var() -> None:
    # Coherence invariant: Expected Shortfall must be >= VaR for non-degenerate tail
    rng = np.random.default_rng(42)
    returns = rng.standard_t(df=4.0, size=1000) * 0.02
    var_cutoff = float(np.percentile(returns, 5.0))  # 95% cutoff (negative)

    es = compute_expected_shortfall(returns, var_cutoff)
    var_mag = -var_cutoff

    assert es > var_mag


def test_basel_traffic_light_regulatory_boundaries() -> None:
    # 250 observations at 99% VaR:
    # Green zone: 0 to 4
    green = evaluate_basel_traffic_light(exceptions=3, total_observations=250)
    assert green.zone == BaselZone.GREEN
    assert green.capital_multiplier == 3.00

    # Yellow zone: 5 to 9
    yellow = evaluate_basel_traffic_light(exceptions=6, total_observations=250)
    assert yellow.zone == BaselZone.YELLOW
    assert yellow.capital_multiplier == 3.50

    # Red zone: >= 10
    red = evaluate_basel_traffic_light(exceptions=11, total_observations=250)
    assert red.zone == BaselZone.RED
    assert red.capital_multiplier == 4.00


# =====================================================================
# 5. EXPORT FUNCTIONALITY
# =====================================================================


def test_export_descriptive_statistics() -> None:
    from pathlib import Path

    import pandas as pd

    from stress_engine.portfolio import PortfolioVaR

    port = PortfolioVaR(
        name="Test Portfolio",
        tickers=["SPY", "TLT"],
        weights=[0.6, 0.4],
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    df = export_descriptive_statistics([port])

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Annualized Volatility" in df.columns
    assert "FHS Breaches" in df.columns
    assert Path("data/raw/week10_descriptive_summary.csv").exists()
