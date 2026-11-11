"""Portfolio Value-at-Risk (VaR) calculation and risk analysis engine."""

import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance as yf
from scipy import stats

from stress_engine.backtest import BacktestResult, run_full_var_backtest
from stress_engine.volatility import compute_ewma_volatility, fit_gjr_garch

TRADING_DAYS = 252
SNAPSHOT_DIR = Path("data/raw/snapshots")


def summarize_return_series(
    portfolio_returns: npt.NDArray[np.float64],
    log_returns: bool = True,
) -> dict[str, float]:
    """Descriptive statistics and maximum drawdown for a single portfolio return series.

    The drawdown is computed on the compounded wealth path starting at 1.0. For log
    returns the wealth path is exp(cumsum(r)); for simple returns it is cumprod(1 + r).
    """
    r = np.ascontiguousarray(portfolio_returns, dtype=np.float64)
    mean_ret = float(np.mean(r))
    std_dev = float(np.std(r, ddof=1))
    z_scores = (r - mean_ret) / std_dev

    if log_returns:
        wealth = np.exp(np.concatenate(([0.0], np.cumsum(r))))
    else:
        wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))

    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak

    return {
        "Daily Mean Return": mean_ret,
        "Annualized Volatility": float(std_dev * np.sqrt(TRADING_DAYS)),
        "Skewness": float(np.mean(z_scores**3)),
        "Excess Kurtosis": float(np.mean(z_scores**4) - 3.0),
        "Maximum Drawdown": float(np.min(drawdown)),
    }


def _rolling_gjr_forecasts(
    returns: npt.NDArray[np.float64],
    lookback_window: int,
    refit_every: int,
    alpha: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Strictly out-of-sample one-step-ahead GJR-GARCH(1,1) volatility forecasts.

    Parameters are re-estimated every ``refit_every`` days, each time on only the
    trailing ``lookback_window`` returns that end the day before the block starts.
    Within a block, the variance recursion runs forward through realized returns,
    so the forecast for day t depends only on returns dated t-1 and earlier.

    Returns:
        sigma_forecast: forecast volatility for days W..T-1, length T - W.
        standardized_quantile: alpha-quantile of the in-window standardized residuals
            for the block containing each day, length T - W. Used by FHS.
    """
    r = np.ascontiguousarray(returns, dtype=np.float64)
    n = r.size
    n_forecasts = n - lookback_window
    sigma_forecast = np.empty(n_forecasts, dtype=np.float64)
    standardized_quantile = np.empty(n_forecasts, dtype=np.float64)

    for block_start in range(lookback_window, n, refit_every):
        block_end = min(block_start + refit_every, n)

        # Training data ends at day block_start - 1: nothing from the block is used
        train = r[block_start - lookback_window : block_start]
        params, train_sigma = fit_gjr_garch(train)

        z_train = train / np.maximum(train_sigma, 1e-8)
        block_quantile = float(np.percentile(z_train, alpha * 100.0))

        # One-step-ahead variance for day block_start, built from the last training day
        last_r = float(train[-1])
        var = (
            params.omega
            + (params.alpha + params.gamma * float(last_r < 0.0)) * last_r**2
            + params.beta * float(train_sigma[-1]) ** 2
        )

        for t in range(block_start, block_end):
            idx = t - lookback_window
            sigma_forecast[idx] = np.sqrt(var)
            standardized_quantile[idx] = block_quantile

            # Advance the recursion with the realized return on day t
            r_t = float(r[t])
            var = max(
                1e-8,
                params.omega
                + (params.alpha + params.gamma * float(r_t < 0.0)) * r_t**2
                + params.beta * var,
            )

    return sigma_forecast, standardized_quantile


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
        self.return_method: str = "simple"
        self.portfolio_returns: npt.NDArray[np.float64] | None = None

    def snapshot_path(self) -> Path:
        """Cache file for this portfolio's price snapshot, keyed by tickers and dates."""
        key = f"{'_'.join(self.tickers)}_{self.start_date}_{self.end_date}"
        return SNAPSHOT_DIR / f"{key}.pkl"

    def download_data(self, refresh: bool = False) -> pd.DataFrame:
        """Loads the cached price snapshot, or downloads and caches it if none exists.

        Yahoo Finance revises adjusted prices over time, so a fresh download can change
        results near a threshold. Pass refresh=True only when a new snapshot is wanted,
        and then record its fingerprint.
        """
        path = self.snapshot_path()
        if path.exists() and not refresh:
            self.market_data = pd.read_pickle(path)
            print(f"[data] Loaded cached price snapshot: {path}")
            return self.market_data

        downloaded = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
        )
        if downloaded.empty:
            raise ValueError(
                f"No price data returned for {self.tickers} between "
                f"{self.start_date} and {self.end_date}."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded.to_pickle(path)
        self.market_data = downloaded
        print(f"[data] Downloaded and cached new price snapshot: {path}")
        return self.market_data

    def data_fingerprint(self) -> str:
        """Short hash of the portfolio return series, for reporting which snapshot was used."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")
        return hashlib.sha256(self.portfolio_returns.tobytes()).hexdigest()[:12]

    def calculate_returns(self, method: str = "simple") -> pd.DataFrame:
        """Calculates asset returns using vectorized pandas transformations.

        Defaults to simple returns. The portfolio return is ``weights @ asset_returns``,
        which is exact only for simple returns under constant weights (a portfolio
        rebalanced to target every day). Summing log returns across assets gives the
        log of a weighted geometric mean, not the log of the portfolio return, so it
        is not used for aggregation. See ``compute_portfolio_returns``.
        """
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
        self.return_method = method
        # Portfolio returns depend on the asset returns, so they must be rebuilt
        self.portfolio_returns = None
        return self.returns

    def compute_portfolio_returns(self) -> npt.NDArray[np.float64]:
        """Computes portfolio returns as the weighted sum of asset simple returns.

        Under constant weights (daily rebalancing to target), the portfolio simple return
        is exactly ``sum_i w_i * R_i``. This method requires simple returns; calling it
        after a log-return calculation raises an error instead of silently mixing units.
        """
        if self.return_method != "simple":
            raise ValueError(
                "Portfolio aggregation requires simple returns. Call "
                "calculate_returns(method='simple') first."
            )
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

        # Row i is returns[i : i + W]; it forecasts returns[i + W], so the last
        # training row ends on the day before the forecast day.
        training_windows = np.lib.stride_tricks.sliding_window_view(
            self.portfolio_returns[:-1], window_shape=lookback_window
        )
        realized_returns = self.portfolio_returns[lookback_window:]

        alpha = 1.0 - self.confidence_level

        if method == "historical":
            var_thresholds = np.percentile(training_windows, alpha * 100.0, axis=1)
        elif method == "parametric":
            mu = np.mean(training_windows, axis=1)
            sigma = np.std(training_windows, axis=1, ddof=1)
            z_score = float(stats.norm.ppf(self.confidence_level))
            var_thresholds = mu - (z_score * sigma)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'.")

        backtest = run_full_var_backtest(
            returns=realized_returns,
            var_thresholds=var_thresholds,
            confidence_level=self.confidence_level,
        )

        return backtest, realized_returns, var_thresholds

    def run_analysis(self) -> dict[str, float | str]:
        """Executes the pipeline and returns a summary dictionary of metrics."""
        self.download_data()
        self.calculate_returns(method="simple")
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
        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None

        # compute_ewma_volatility sets sigma[t] from returns through t-1, so sigma[t]
        # is already the forecast for returns[t]. No shift is applied.
        sigma_series = compute_ewma_volatility(
            self.portfolio_returns, decay_factor=decay_factor
        )
        forecast_sigma = sigma_series[lookback_window:]
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
        refit_every: int = 21,
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes out-of-sample backtest using rolling-refit GJR-GARCH(1,1) volatility."""
        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        alpha = 1.0 - self.confidence_level
        forecast_sigma, _ = _rolling_gjr_forecasts(
            self.portfolio_returns, lookback_window, refit_every, alpha
        )
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
        refit_every: int = 21,
    ) -> tuple[BacktestResult, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Executes Filtered Historical Simulation (FHS) backtest.

        Combines rolling-refit GJR-GARCH(1,1) conditional volatility with the empirical
        quantile of standardized residuals from the same trailing training window.
        """
        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        alpha = 1.0 - self.confidence_level
        forecast_sigma, std_quantile = _rolling_gjr_forecasts(
            self.portfolio_returns, lookback_window, refit_every, alpha
        )
        realized_returns = self.portfolio_returns[lookback_window:]

        # Negative loss cutoff: empirical standardized quantile scaled by forecast sigma
        var_thresholds = std_quantile * forecast_sigma

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
    portfolio_list: list,
    output_dir: str | Path = "data/output",
    lookback_window: int = 252,
) -> pd.DataFrame:
    """Computes descriptive statistics and out-of-sample breach counts for each portfolio.

    Statistics are computed on the portfolio return series, not the per-asset matrix.
    The master summary is written to CSV in ``output_dir``.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for port in portfolio_list:
        if port.returns is None or port.returns.empty:
            port.run_analysis()
        if port.portfolio_returns is None:
            port.compute_portfolio_returns()

        portfolio_returns = port.portfolio_returns
        assert portfolio_returns is not None

        stats_row = summarize_return_series(
            portfolio_returns, log_returns=(port.return_method == "log")
        )

        hist_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=lookback_window, method="historical"
        )
        param_bt, _, _ = port.run_rolling_out_of_sample_backtest(
            lookback_window=lookback_window, method="parametric"
        )
        ewma_bt, _, _ = port.run_ewma_out_of_sample_backtest(
            lookback_window=lookback_window, decay_factor=0.94
        )
        garch_bt, _, _ = port.run_gjr_garch_out_of_sample_backtest(
            lookback_window=lookback_window
        )
        fhs_bt, _, _ = port.run_fhs_out_of_sample_backtest(
            lookback_window=lookback_window
        )

        record = {
            "Portfolio": port.name,
            "Data Fingerprint": port.data_fingerprint(),
            "Observations": int(portfolio_returns.size),
            **stats_row,
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
