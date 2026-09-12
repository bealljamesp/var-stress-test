"""Portfolio Value-at-Risk (VaR) calculation and risk analysis engine."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance as yf
from scipy import stats

from stress_engine.backtest import BacktestResult, run_full_var_backtest


class PortfolioVaR:
    """Multi-asset portfolio VaR calculator supporting Historical, Parametric, and Rolling OOS methods."""

    def __init__(
        self,
        name: str,
        tickers: list[str],
        weights: list[float] | npt.NDArray[np.float64],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        confidence_level: float = 0.95,
    ) -> None:
        self.name = name
        self.tickers = tickers
        self.weights: npt.NDArray[np.float64] = np.ascontiguousarray(
            weights, dtype=np.float64
        )

        if len(self.tickers) != self.weights.size:
            raise ValueError(
                f"Dimension mismatch: {len(self.tickers)} tickers provided, "
                f"but received {self.weights.size} weights."
            )

        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-4):
            raise ValueError(
                f"Portfolio weights must sum to 1.0. Current sum for '{self.name}': "
                f"{np.sum(self.weights):.4f}"
            )

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.confidence_level = confidence_level

        self.market_data: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.portfolio_returns: npt.NDArray[np.float64] | None = None

    def download_data(self) -> pd.DataFrame:
        """Downloads historical market data for the target tickers."""
        self.market_data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
        )
        return self.market_data

    def calculate_returns(self, method: str = "log") -> pd.DataFrame:
        """Calculates asset returns using vectorized pandas transformations."""
        if self.market_data is None or self.market_data.empty:
            raise ValueError("Market data is empty. Call download_data() first.")

        if isinstance(self.market_data.columns, pd.MultiIndex):
            price_data = (
                self.market_data["Adj Close"]
                if "Adj Close" in self.market_data.columns.levels[0]
                else self.market_data["Close"]
            )
        else:
            price_data = (
                self.market_data[["Adj Close"]]
                if "Adj Close" in self.market_data.columns
                else self.market_data[["Close"]]
            )

        price_data = price_data.loc[:, self.tickers]

        if method == "log":
            self.returns = np.log(price_data / price_data.shift(1))
        elif method == "simple":
            self.returns = price_data.pct_change()
        else:
            raise ValueError("Invalid return method. Choose 'log' or 'simple'.")

        self.returns = self.returns.dropna()
        return self.returns

    def compute_portfolio_returns(self) -> npt.NDArray[np.float64]:
        """Computes portfolio returns via zero-copy NumPy matrix dot product."""
        if self.returns is None or self.returns.empty:
            raise ValueError("Returns not computed. Call calculate_returns() first.")

        returns_mat: npt.NDArray[np.float64] = self.returns.to_numpy(
            dtype=np.float64, copy=False
        )
        self.portfolio_returns = np.ascontiguousarray(
            returns_mat @ self.weights, dtype=np.float64
        )
        return self.portfolio_returns

    def calculate_historical_var(self) -> float:
        """Calculates unconditional in-sample Historical VaR expressed as a positive loss percentage."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        alpha = 1.0 - self.confidence_level
        loss_cutoff = float(np.percentile(self.portfolio_returns, alpha * 100))
        return max(0.0, -loss_cutoff)

    def calculate_parametric_var(self) -> float:
        """Calculates unconditional in-sample Parametric VaR assuming normal distribution."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        mu = float(np.mean(self.portfolio_returns))
        sigma = float(np.std(self.portfolio_returns, ddof=1))
        z_score = float(stats.norm.ppf(self.confidence_level))

        var_pct = (z_score * sigma) - mu
        return max(0.0, var_pct)

    def run_rolling_out_of_sample_backtest(
        self,
        lookback_window: int = 252,
        method: str = "historical",
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes a fully vectorized, out-of-sample dynamic VaR backtest using sliding strided views."""
        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        total_obs = self.portfolio_returns.size

        if total_obs <= lookback_window:
            raise ValueError(
                f"Total observations ({total_obs}) must exceed lookback window ({lookback_window})."
            )

        # Extract pre-realization training slices via O(1) strided window: (T - W, W)
        training_windows = np.lib.stride_tricks.sliding_window_view(
            self.portfolio_returns[:-1], window_shape=lookback_window
        )
        # Out-of-sample realized returns: vector of length T - W
        realized_returns = self.portfolio_returns[lookback_window:]

        alpha = 1.0 - self.confidence_level

        if method == "historical":
            # Vectorized percentile computation along lookback axis (axis 1)
            var_thresholds = np.percentile(training_windows, alpha * 100.0, axis=1)
        elif method == "parametric":
            # Vectorized mean and standard deviation along lookback axis
            mu = np.mean(training_windows, axis=1)
            sigma = np.std(training_windows, axis=1, ddof=1)
            z_score = float(stats.norm.ppf(self.confidence_level))
            # Loss cutoff threshold in return space (negative value)
            var_thresholds = mu - (z_score * sigma)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'.")

        # Run Kupiec & Christoffersen against realized out-of-sample series
        backtest = run_full_var_backtest(
            returns=realized_returns,
            var_thresholds=var_thresholds,
            confidence_level=self.confidence_level,
        )

        return backtest, realized_returns, var_thresholds

    def run_analysis(self) -> dict[str, float | str]:
        """Executes the pipeline and returns a summary dictionary of metrics."""
        self.download_data()
        self.calculate_returns(method="log")
        self.compute_portfolio_returns()

        hist_var = self.calculate_historical_var()
        param_var = self.calculate_parametric_var()
        hist_es = self.calculate_historical_expected_shortfall()
        param_es = self.calculate_parametric_expected_shortfall()

        return {
            "name": self.name,
            "period": f"{self.start_date} to {self.end_date}",
            "hist_pct": hist_var,
            "hist_dollar": hist_var * self.initial_capital,
            "hist_es_pct": hist_es,
            "hist_es_dollar": hist_es * self.initial_capital,
            "param_pct": param_var,
            "param_dollar": param_var * self.initial_capital,
            "param_es_pct": param_es,
            "param_es_dollar": param_es * self.initial_capital,
        }

    def run_ewma_out_of_sample_backtest(
        self,
        lookback_window: int = 252,
        decay_factor: float = 0.94,
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes out-of-sample backtest using dynamic EWMA conditional volatility."""
        from stress_engine.volatility import compute_ewma_volatility

        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        sigma_series = compute_ewma_volatility(
            self.portfolio_returns, decay_factor=decay_factor
        )

        # Shift conditional sigma by 1 to make it strictly out-of-sample: sigma_t forecasts r_t
        forecast_sigma = sigma_series[lookback_window - 1 : -1]
        realized_returns = self.portfolio_returns[lookback_window:]

        z_score = float(stats.norm.ppf(self.confidence_level))
        var_thresholds = -(z_score * forecast_sigma)

        backtest = run_full_var_backtest(
            returns=realized_returns,
            var_thresholds=var_thresholds,
            confidence_level=self.confidence_level,
        )
        return backtest, realized_returns, var_thresholds

    def run_gjr_garch_out_of_sample_backtest(
        self,
        lookback_window: int = 252,
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes out-of-sample backtest using dynamic GJR-GARCH(1,1) asymmetric conditional volatility."""
        from stress_engine.volatility import fit_gjr_garch

        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        _, cond_sigma = fit_gjr_garch(self.portfolio_returns)

        # Shift conditional sigma by 1 to maintain strictly out-of-sample alignment: sigma_t forecasts r_t
        forecast_sigma = cond_sigma[lookback_window - 1 : -1]
        realized_returns = self.portfolio_returns[lookback_window:]

        z_score = float(stats.norm.ppf(self.confidence_level))
        var_thresholds = -(z_score * forecast_sigma)

        backtest = run_full_var_backtest(
            returns=realized_returns,
            var_thresholds=var_thresholds,
            confidence_level=self.confidence_level,
        )
        return backtest, realized_returns, var_thresholds

    def run_fhs_out_of_sample_backtest(
        self,
        lookback_window: int = 252,
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes Filtered Historical Simulation (FHS) backtest combining GJR-GARCH(1,1)

        conditional volatility with empirical standardized residual quantiles.
        """
        from stress_engine.volatility import fit_gjr_garch

        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        _, cond_sigma = fit_gjr_garch(self.portfolio_returns)

        # De-volatilize returns into standardized innovations
        standardized_residuals = self.portfolio_returns / np.maximum(cond_sigma, 1e-8)

        # Stride training windows over standardized residuals to compute rolling empirical tail quantiles
        # Shape: (T - W, W)
        res_windows = np.lib.stride_tricks.sliding_window_view(
            standardized_residuals[:-1], window_shape=lookback_window
        )

        alpha = 1.0 - self.confidence_level
        # Empirical innovations quantile along lookback window: vector of length T - W
        empirical_q = np.percentile(res_windows, alpha * 100.0, axis=1)

        # Strictly out-of-sample sigma forecast
        forecast_sigma = cond_sigma[lookback_window - 1 : -1]
        realized_returns = self.portfolio_returns[lookback_window:]

        # Dynamic FHS threshold in return space (negative loss cutoff)
        var_thresholds = empirical_q * forecast_sigma

        backtest = run_full_var_backtest(
            returns=realized_returns,
            var_thresholds=var_thresholds,
            confidence_level=self.confidence_level,
        )
        return backtest, realized_returns, var_thresholds

    def calculate_historical_expected_shortfall(self) -> float:
        """Calculates in-sample Historical Expected Shortfall (CVaR) as a positive loss percentage."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        alpha = 1.0 - self.confidence_level
        loss_cutoff = float(np.percentile(self.portfolio_returns, alpha * 100.0))
        breaches = self.portfolio_returns[self.portfolio_returns < loss_cutoff]

        if breaches.size == 0:
            return max(0.0, -loss_cutoff)

        return float(np.mean(-breaches))

    def calculate_parametric_expected_shortfall(self) -> float:
        """Calculates Parametric Expected Shortfall (CVaR) assuming normal distribution."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        mu = float(np.mean(self.portfolio_returns))
        sigma = float(np.std(self.portfolio_returns, ddof=1))
        alpha = 1.0 - self.confidence_level

        # ES = -mu + sigma * (phi(Phi^-1(alpha)) / alpha)
        z_cutoff = stats.norm.ppf(alpha)
        es_loss = -mu + sigma * (stats.norm.pdf(z_cutoff) / alpha)
        return max(0.0, float(es_loss))


def export_descriptive_statistics(
    portfolio_list: list, output_dir: str | Path = "data/raw"
) -> pd.DataFrame:
    """Computes vectorized descriptive statistics and out-of-sample exception
    counts for multiple portfolios, exporting the master summary to CSV
    for Week 10 MS Excel chart generation.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for port in portfolio_list:
        # Ensure data is downloaded and returns are computed if not already done
        if port.returns is None or port.returns.empty:
            port.run_analysis()

        # Pull returns using zero-copy contiguous access
        returns: NDArray[np.float64] = port.returns.to_numpy(
            dtype=np.float64, copy=False
        )

        # Vectorized higher-order moments & metrics
        mean_ret = float(np.mean(returns))
        vol_ann = float(np.std(returns, ddof=1) * np.sqrt(252))

        # Higher-order standardized moments (skewness and excess kurtosis)
        std_dev = np.std(returns, ddof=1)
        z_scores = (returns - mean_ret) / std_dev
        skewness = float(np.mean(z_scores**3))
        excess_kurtosis = float(np.mean(z_scores**4) - 3.0)

        # Maximum Drawdown calculation via cumulative product vectorization
        wealth_index = 1.0 + np.cumsum(returns)
        peak = np.maximum.accumulate(wealth_index)
        drawdown = (wealth_index - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # Run out-of-sample backtests to capture exact breach counts[cite: 1]
        hist_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=252, method="historical"
        )
        param_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=252, method="parametric"
        )
        ewma_bt, _, _ = port.run_ewma_out_of_sample_backtest(
            lookback_window=252, decay_factor=0.94
        )
        garch_bt, _, _ = port.run_gjr_garch_out_of_sample_backtest(lookback_window=252)
        fhs_bt, _, _ = port.run_fhs_out_of_sample_backtest(lookback_window=252)

        record = {
            "Portfolio": port.name,
            "Observations": len(returns),
            "Daily Mean Return": mean_ret,
            "Annualized Volatility": vol_ann,
            "Skewness": skewness,
            "Excess Kurtosis": excess_kurtosis,
            "Maximum Drawdown": max_drawdown,
            "Hist VaR Breaches": int(hist_bt.total_exceptions),
            "Param VaR Breaches": int(param_bt.total_exceptions),
            "EWMA Breaches": int(ewma_bt.total_exceptions),
            "GARCH Breaches": int(garch_bt.total_exceptions),
            "FHS Breaches": int(fhs_bt.total_exceptions),
        }
        summary_records.append(record)

    df_summary = pd.DataFrame(summary_records)
    csv_path = path / "week10_descriptive_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[Export Complete] Descriptive statistics saved to: {csv_path}")

    return df_summary
