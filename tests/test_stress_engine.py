"""Comprehensive mathematical and behavioral unit test suite for stress_engine.

Portfolio tests construct synthetic return series directly instead of downloading
market data, so the suite runs offline and does not write to tracked output folders.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import stress_engine.volatility as volatility
from stress_engine.backtest import (
    BaselZone,
    calculate_var_exceptions,
    christoffersen_independence_test,
    compute_expected_shortfall,
    evaluate_basel_traffic_light,
    kupiec_pof_test,
    run_full_var_backtest,
)
from stress_engine.monte_carlo import run_comprehensive_stress_engine
from stress_engine.portfolio import (
    PortfolioVaR,
    export_descriptive_statistics,
    summarize_return_series,
)
from stress_engine.visualization import plot_monte_carlo_drawdown_surface
from stress_engine.volatility import compute_ewma_volatility, fit_gjr_garch

# =====================================================================
# Helpers
# =====================================================================


def _synthetic_portfolio(returns: np.ndarray, name: str = "Synthetic") -> PortfolioVaR:
    """Portfolio whose return series is supplied directly (no network access)."""
    port = PortfolioVaR(
        name=name,
        tickers=["A"],
        weights=[1.0],
        start_date="2020-01-01",
        end_date="2021-01-01",
    )
    port.portfolio_returns = np.ascontiguousarray(returns, dtype=np.float64)
    return port


def _run_thresholds(method: str, returns: np.ndarray, lookback: int) -> np.ndarray:
    port = _synthetic_portfolio(returns)
    if method == "ewma":
        _, _, thresholds = port.run_ewma_out_of_sample_backtest(lookback_window=lookback)
    elif method == "gjr":
        _, _, thresholds = port.run_gjr_garch_out_of_sample_backtest(
            lookback_window=lookback, refit_every=21
        )
    elif method == "fhs":
        _, _, thresholds = port.run_fhs_out_of_sample_backtest(
            lookback_window=lookback, refit_every=21
        )
    else:
        raise ValueError(method)
    return thresholds


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


def test_gjr_fit_raises_when_optimizer_does_not_converge(monkeypatch) -> None:
    # The starting guess must never be returned as if it were a fitted model
    def failed_minimize(*args, **kwargs):
        return SimpleNamespace(
            success=False,
            x=np.zeros(4),
            fun=np.inf,
            message="Positive directional derivative for linesearch",
        )

    monkeypatch.setattr(volatility, "minimize", failed_minimize)
    returns = np.random.default_rng(0).normal(0.0, 0.01, size=300)

    with pytest.raises(RuntimeError, match="failed to converge"):
        fit_gjr_garch(returns)


# =====================================================================
# 3. OUT-OF-SAMPLE ALIGNMENT & LOOK-AHEAD INVARIANTS
# =====================================================================


def test_ewma_threshold_matches_causal_sigma_for_same_day() -> None:
    # Threshold i is the forecast for day W + i, built from returns through W + i - 1
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.02, size=400)
    lookback = 100

    _, realized, thresholds = _synthetic_portfolio(returns).run_ewma_out_of_sample_backtest(
        lookback_window=lookback
    )
    sigma = compute_ewma_volatility(returns, decay_factor=0.94)
    z = stats.norm.ppf(0.95)

    assert np.array_equal(realized, returns[lookback:])
    assert np.allclose(thresholds, -z * sigma[lookback:], rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("method", ["ewma", "gjr", "fhs"])
def test_forecasts_do_not_depend_on_future_returns(method: str) -> None:
    rng = np.random.default_rng(11)
    returns = rng.standard_t(df=5.0, size=400) * 0.015
    lookback = 100
    cut = 250  # Days from 250 onward are altered

    perturbed = returns.copy()
    perturbed[cut:] = perturbed[cut:] * -3.0 + 0.05

    thr_original = _run_thresholds(method, returns, lookback)
    thr_perturbed = _run_thresholds(method, perturbed, lookback)

    # thresholds[i] is the forecast for day lookback + i. Forecasts for days before
    # `cut` must be identical, because they use only returns before `cut`.
    k = cut - lookback
    assert np.allclose(thr_original[:k], thr_perturbed[:k], rtol=0.0, atol=1e-12)


# =====================================================================
# 4. 4D TENSOR MONTE CARLO INVARIANTS
# =====================================================================


def test_monte_carlo_is_seed_reproducible(tmp_path) -> None:
    # Same seed, same grid: metrics and path tensor must match exactly
    df_a, tensor_a = run_comprehensive_stress_engine(
        n_paths=200, n_days=40, random_seed=42, output_dir=tmp_path / "a"
    )
    df_b, tensor_b = run_comprehensive_stress_engine(
        n_paths=200, n_days=40, random_seed=42, output_dir=tmp_path / "b"
    )
    pd.testing.assert_frame_equal(df_a, df_b)
    assert np.array_equal(tensor_a, tensor_b)


def test_stress_surface_regenerates_missing_results(tmp_path) -> None:
    # The results parquet is not tracked in git. If it is missing, the plot step must
    # regenerate it instead of failing, and write everything under tmp_path.
    out = plot_monte_carlo_drawdown_surface(
        output_filename=tmp_path / "surface.png",
        results_path=tmp_path / "synthetic_4d_monte_carlo_results.parquet",
    )
    assert out.exists()
    assert (tmp_path / "synthetic_4d_monte_carlo_results.parquet").exists()


def test_monte_carlo_tensor_geometry_and_bounds(tmp_path) -> None:
    # Lightweight smoke run: 100 paths, 10 days. Output goes to a temp dir, not data/raw.
    df_results, master_tensor = run_comprehensive_stress_engine(
        n_paths=100, n_days=10, output_dir=tmp_path
    )

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
# 5. PORTFOLIO ENGINE CONTRACTS & INPUT VALIDATION
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
    port = _synthetic_portfolio(np.random.default_rng(1).normal(0.0, 0.01, size=21))
    with pytest.raises(ValueError, match="must exceed lookback window"):
        port.run_rolling_out_of_sample_backtest(lookback_window=252)


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
# 6. DESCRIPTIVE STATISTICS & EXPORT
# =====================================================================


def test_summary_max_drawdown_uses_compounded_log_wealth() -> None:
    # Log returns -0.5 then +0.2: wealth 1.0 -> exp(-0.5), trough measured from 1.0
    out = summarize_return_series(np.array([-0.5, 0.2]), log_returns=True)
    assert np.isclose(out["Maximum Drawdown"], np.exp(-0.5) - 1.0)


def test_summary_max_drawdown_uses_compounded_simple_wealth() -> None:
    # Simple returns -50% then +100%: wealth 1.0 -> 0.5 -> 1.0, drawdown -50%
    out = summarize_return_series(np.array([-0.5, 1.0]), log_returns=False)
    assert np.isclose(out["Maximum Drawdown"], -0.5)


def test_export_descriptive_statistics_writes_to_requested_dir(tmp_path) -> None:
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    log_returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n, 2)), index=idx, columns=["SPY", "TLT"]
    )

    port = PortfolioVaR(
        name="Test Portfolio",
        tickers=["SPY", "TLT"],
        weights=[0.6, 0.4],
        start_date="2022-01-01",
        end_date="2023-12-31",
    )
    port.returns = log_returns  # Supply returns directly; no download

    df = export_descriptive_statistics([port], output_dir=tmp_path, lookback_window=120)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Annualized Volatility" in df.columns
    assert "FHS Breaches" in df.columns
    assert (tmp_path / "week10_descriptive_summary.csv").exists()


def _two_asset_market_data(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(n, 2)), axis=0))
    cols = pd.MultiIndex.from_product([["Adj Close"], ["AAA", "BBB"]])
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_input_fingerprint_is_stable_and_detects_price_changes() -> None:
    port = PortfolioVaR(
        name="Fingerprint", tickers=["AAA", "BBB"], weights=[0.5, 0.5],
        start_date="2020-01-01", end_date="2020-03-31",
    )
    port.market_data = _two_asset_market_data()
    fp = port.input_fingerprint()
    assert fp == port.input_fingerprint()

    # Moving one adjusted price by 0.01% must change the fingerprint
    changed = _two_asset_market_data()
    changed.iloc[10, 0] *= 1.0001
    port.market_data = changed
    assert port.input_fingerprint() != fp
