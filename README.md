# Quantitative Risk & Liquidity Stress Testing Engine (`stress_engine`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/typing-PEP%20585%20%7C%20604%20%7C%20646-informational.svg)]()

A quantitative risk and liquidity stress-testing engine built to institutional risk standards (Basel Committee on Banking Supervision / FRM Part 1).

The engine evaluates historical portfolio drawdowns, calculates dynamic conditional Value-at-Risk (VaR) and Expected Shortfall (ES), runs formal econometric regulatory backtests, and runs vectorized Monte Carlo macroeconomic stress simulations across a 4D parametric state space ($36.45\text{M}$ path coordinates in $<3.0\text{s}$ on the reference machine).

---

## Key Econometric & Engineering Features

* **Vectorized Computation:** Rolling VaR thresholds, portfolio combinations, and simulation paths are computed with NumPy array operations. Rolling windows use `numpy.lib.stride_tricks.sliding_window_view`, which builds the training matrices without copying the data. The recursive volatility filters (EWMA and GJR-GARCH) and the rolling GJR-GARCH refits are Python loops, because each step depends on the previous one.
* **Dynamic Conditional Volatility Engines:**
  * **RiskMetrics EWMA ($\lambda=0.94$):** Exponentially decaying memory footprint.
  * **GJR-GARCH(1,1):** Asymmetric leverage estimated by constrained Sequential Least Squares Programming (SLSQP) Maximum Likelihood Estimation (MLE). Refit every 21 days on the trailing 252 returns, from two starting points, with a tighter-tolerance polish pass. The presample variance follows the convention in the `arch` package. A fit that fails to converge raises an error; the engine never substitutes starting values.
  * **Filtered Historical Simulation (FHS):** Non-parametric standardization using empirical standardized innovations $z_t = r_t / \sigma_t$, with the innovation quantile taken from the same trailing training window as the GJR-GARCH fit.
* **Regulatory Backtesting Suite:**
  * **Kupiec Proportion of Failures (POF) Test ($LR_{\text{uc}}$):** Unconditional coverage testing.
  * **Christoffersen Markov Independence Test ($LR_{\text{ind}}$):** Exception clustering and first-order Markov transition likelihood ratio testing.
  * **Christoffersen Combined Conditional Coverage ($LR_{\text{cc}}$):** Joint evaluation against $\chi^2(2)$.
  * **Basel Committee (BCBS) Traffic Light System:** Exact binomial cumulative density classification into Green, Yellow, and Red capital penalty zones.
* **Out-of-Sample Alignment:** Every forecast for day $t$ uses only returns through day $t-1$. Tests check this directly, including a test that changing future returns leaves earlier forecasts unchanged.
* **Modern Python 3.12+ Standards:** Strict type annotations via `numpy.typing.NDArray[np.float64]`, modern type unions (`|`), explicit axis selections, zero-copy `.to_numpy(dtype=np.float64, copy=False)` operations, and a `src/` layout installed with `pip install -e .`.

---

## System Architecture

```text
var-stress-test/
├── pyproject.toml                     # PEP 517/518 build specification
├── main.py                            # End-to-end execution pipeline
├── src/
│   └── stress_engine/
│       ├── __init__.py                # Package initialization & exports
│       ├── backtest.py                # Kupiec, Christoffersen, & Basel III engines
│       ├── ingestion.py               # Distressed-entity HTML parsing, event-window coverage checks, manifest
│       ├── monte_carlo.py             # 4D tensor stress testing engine
│       ├── portfolio.py               # PortfolioVaR, rolling OOS backtests, GJR-GARCH & FHS engines
│       ├── visualization.py           # VaR diagnostic charts and 4D stress surface
│       └── volatility.py              # EWMA and GJR-GARCH(1,1) MLE solver
├── scripts/
│   ├── compare_gjr_arch.py            # Cross-check of the GJR-GARCH fit against the arch package
│   └── check_presample.py             # Verifies the presample variance convention against arch
├── tests/
│   ├── test_ingestion.py              # 12 ingestion invariants (offline, synthetic HTML)
│   └── test_stress_engine.py          # 23 statistical, alignment, and contract invariants
├── data/
│   ├── patches/                       # Normalized distressed-entity CSVs and manifest.csv
│   ├── output/                        # Descriptive summary CSV and VaR diagnostic charts
│   ├── plots/                         # 4D Monte Carlo stress surface (300 DPI)
│   └── raw/                           # Raw HTML scrapes, Monte Carlo parquet & tensors
└── research/
    └── archive/                       # Deprecated scratchpads & experiments
```

---

## Mathematical Specification

### Dynamic Conditional Volatility

* RiskMetrics EWMA Filter

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2 \quad (\lambda = 0.94)$$

* Asymmetric GJR-GARCH(1,1)

$$\sigma_t^2 = \omega + \left( \alpha + \gamma \cdot \mathbb{I}_{\{r_{t-1} < 0\}} \right) r_{t-1}^2 + \beta \sigma_{t-1}^2$$

* Presample variance, following the `arch` convention. With $b$ the weighted average of the first $\min(75, n)$ squared returns (weights $0.94^k$, normalized):

$$\sigma_0^2 = \omega + \left(\alpha + \tfrac{\gamma}{2}\right) b + \beta b$$

* Subject to parameter constraints:

$$\omega > 0, \quad \alpha \ge 0, \quad \gamma \ge 0, \quad \beta \ge 0, \quad \alpha + \beta + 0.5\gamma < 1.0$$

* Filtered Historical Simulation (FHS). Standardized innovations are filtered from conditional volatility:

$$z_t = \frac{r_t}{\sigma_t}$$

* The out-of-sample loss cutoff is the empirical quantile of the standardized innovations in the training window, scaled by the forecast volatility:

$$\text{VaR}_t^{\text{FHS}}(\alpha) = \text{Quantile}_\alpha(\{z_\tau\}_{\tau=t-W}^{t-1}) \cdot \sigma_t$$

### Regulatory Backtesting Framework

* Kupiec Unconditional Coverage Test ($LR_{\text{uc}}$)

Evaluates whether empirical failure rate $\hat{p} = x / N$ statistically diverges from nominal rate $p = 1 - \alpha$:

$$LR_{\text{uc}} = -2 \ln \left[ \frac{(1 - p)^{N-x} p^x}{(1 - \hat{p})^{N-x} \hat{p}^x} \right] \sim \chi^2(1)$$

* Christoffersen Independence Test ($LR_{\text{ind}}$)

Evaluates exception clustering using a first-order Markov chain:

$$LR_{\text{ind}} = -2 \ln \left[ \frac{L(\hat{\Pi}_1)}{L(\hat{\Pi}_2)} \right] \sim \chi^2(1)$$

where:

$$L(\hat{\Pi}_1) = (1 - \pi)^{T_{00} + T_{10}} \pi^{T_{01} + T_{11}}, \quad L(\hat{\Pi}_2) = (1 - \pi_{01})^{T_{00}} \pi_{01}^{T_{01}} (1 - \pi_{11})^{T_{10}} \pi_{11}^{T_{11}}$$

* Combined Conditional Coverage Test ($LR_{\text{cc}}$)

$$LR_{\text{cc}} = LR_{\text{uc}} + LR_{\text{ind}} \sim \chi^2(2)$$

### Coherent Risk Measures: Expected Shortfall (CVaR)

To address the non-subadditivity of Value-at-Risk, the engine computes Expected Shortfall:

$$\text{ES}_\alpha = \mathbb{E}\left[ -r_t \mid -r_t > \text{VaR}_\alpha \right]$$

---

## Empirical Benchmark Findings (2007–2009 Global Financial Crisis)

Each portfolio is backtested over N = 502 out-of-sample trading days with a W = 252 day lookback, at 95% confidence (p = 0.05, expected breaches = 25.1). The table is generated by `main.py`.

### Concentrated Financials (equal-weight C, BAC, XLF, GS)

| Risk Model Engine | Breaches / Obs | Empirical Rate | Kupiec POF (LR_uc) | Christoffersen (LR_ind) | Regulatory Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rolling Historical (252d) | 46 / 502 | 9.16% | 14.861 (p = 0.0001) | 0.469 (p = 0.4933) | **Reject Unconditional** |
| Rolling Parametric (Normal) | 44 / 502 | 8.76% | 12.355 (p = 0.0004) | 0.248 (p = 0.6183) | **Reject Unconditional** |
| **Dynamic EWMA (λ=0.94)** | 35 / 502 | 6.97% | 3.681 (p = 0.0551) | 1.233 (p = 0.2668) | **Accept Both** |
| Dynamic GJR-GARCH(1,1) | 33 / 502 | 6.57% | 2.392 (p = 0.1220) | 4.658 (p = 0.0309) | **Reject Independence** |
| **Filtered Historical (FHS)** | 30 / 502 | 5.98% | 0.950 (p = 0.3296) | 3.824 (p = 0.0505) | **Accept Both** (borderline) |

### Multi-Asset Diverse (SPY 40%, TLT 30%, GLD 20%, DBC 10%)

| Risk Model Engine | Breaches / Obs | Empirical Rate | Kupiec POF (LR_uc) | Christoffersen (LR_ind) | Regulatory Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rolling Historical (252d) | 29 / 502 | 5.78% | 0.609 (p = 0.4352) | 12.102 (p = 0.0005) | **Reject Independence** |
| Rolling Parametric (Normal) | 31 / 502 | 6.18% | 1.363 (p = 0.2431) | 7.091 (p = 0.0077) | **Reject Independence** |
| **Dynamic EWMA (λ=0.94)** | 25 / 502 | 4.98% | 0.000 (p = 0.9837) | 0.038 (p = 0.8455) | **Accept Both** |
| **Dynamic GJR-GARCH(1,1)** | 27 / 502 | 5.38% | 0.148 (p = 0.7005) | 0.141 (p = 0.7074) | **Accept Both** |
| **Filtered Historical (FHS)** | 25 / 502 | 4.98% | 0.000 (p = 0.9837) | 0.038 (p = 0.8455) | **Accept Both** |

### Takeaways

**Static in-sample VaR is not a backtest.** In-sample historical simulation hits its own quantile by construction, so it reports the nominal rate exactly. That hides any time variation in the exceptions. The rolling out-of-sample rows above are the honest comparison.

**Dynamic volatility reduces breaches in the crisis.** Against the static rolling windows, the dynamic models cut breaches in Concentrated Financials from 44–46 to 30–35. EWMA passes both tests there. GJR-GARCH passes unconditional coverage but its exceptions still cluster in time (Christoffersen p = 0.031). FHS passes both tests, but its independence p-value (0.0505) sits right at the 0.05 cutoff, so that pass is borderline.

**Multi-asset results are mixed, and some matches are coincidental.** Both rolling windows reject independence in the multi-asset portfolio. EWMA, GJR-GARCH, and FHS pass both tests there. EWMA and FHS produce identical breach counts and identical transition counts (T00 = 453, T01 = 23, T10 = 24, T11 = 1), so their matching statistics follow from the counts. Their thresholds differ, with a maximum gap of 0.79 percentage points, and their breach flags disagree on 8 of 502 days. The 25-breach count and the single consecutive breach pair (T11 = 1) mean the independence evidence in this portfolio is weak.

---

## Distressed-Entity Patches (Event-Window Coverage)

`ingestion.py` parses the raw HTML scrapes for six distressed financial and energy names and writes normalized patches. A patch is written only when its source covers the entity's event window. Otherwise the entity is recorded as skipped in `data/patches/manifest.csv`.

| Ticker | Event Window | Source Cadence | Obs in Window | Status |
| :--- | :--- | :--- | :--- | :--- |
| LEHMQ | 2007-01 to 2008-12 | Monthly | 24 | Written |
| BSC | 2007-01 to 2008-06 | Monthly | 17 | Written |
| SIVB | 2023-01 to 2023-05 | Monthly | 5 | Written |
| PACW | 2023-01 to 2023-12 | Monthly | 12 | Written |
| SBNY | 2023-01 to 2023-06 | Daily (2024–2025 only) | 0 | Skipped: no coverage |
| CHK | 2020-01 to 2021-06 | Monthly (from 2021-02) | 5 | Skipped: partial coverage |

These patches are not yet used by the analysis. The windows contain 5 to 24 monthly observations, which is too few for the 252-day backtests.

---

## Validation of the GJR-GARCH Fit

The in-house GJR-GARCH fit is cross-checked against the `arch` package on three one-year windows of the four-asset portfolio (`scripts/compare_gjr_arch.py`). Both use a zero mean and Gaussian innovations.

* Parameters agree to four decimal places in all three windows.
* Conditional volatility paths have correlation 1.00000 with relative differences near 0.00%.
* Log-likelihoods agree to about 0.001 once the $-\tfrac{1}{2} n \log(2\pi)$ constant is included.

This is a check on three windows, not a proof of correctness across the full sample. The earlier versions of the fit did not match `arch`, and the mismatch came from the presample variance convention and the scaling of the optimizer, both now fixed.

---

## 4D Tensor Monte Carlo Simulation Architecture

The macroeconomic stress module simulates an 81-node Cartesian parameter space across:

* Annualized Volatility Tiers ($\sigma_{\text{ann}}$): 10%, 20%, 35%

* Student-$t$ Tail Thickness ($\nu$): 3.5 (Fat), 8.0 (Normal), 30.0 (Thin)

* GJR Asymmetric Leverage ($\gamma$): 0.01, 0.05, 0.15

* Exogenous Macro Shocks ($s$): -15%, -30%, -50% at $t = 30$

The grid values are chosen by hand. They are not calibrated to historical crises, so the stress surface is an illustrative sensitivity analysis, not a forecast of crisis probabilities.

Memory & Execution Profile

* Tensor Dimensions: $(K=81\text{ nodes}) \times (P=5,000\text{ paths}) \times (T=90\text{ days})$

* Total Points Evaluated: 36,450,000 single-precision float32 values.

* Contiguous Memory Footprint: $\approx 145.8\text{ MB}$.

* Execution Latency: under 3 seconds on the reference machine, using broadcasted NumPy array operations. Not re-measured on other hardware.

$$\mathbf{S}_{k, p, t} = 100 \cdot \exp\left( \sum_{\tau=1}^t \mathbf{R}_{k, p, \tau} \right)$$

$$\mathbf{MDD}_{k, p} = \min_{t \in [1, T]} \left( \frac{\mathbf{S}_{k, p, t} - \max_{\tau \le t} \mathbf{S}_{k, p, \tau}}{\max_{\tau \le t} \mathbf{S}_{k, p, \tau}} \right)$$

The stress surface plots distress probability over the volatility and shock-severity plane. Tail thickness and asymmetry are averaged out, and the figure title says so.

---

## Diagnostic Figures

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **VaR Backtest Diagnostics** | `data/output/var_diagnostics_*.png` | One chart per risk model: daily returns, the VaR threshold, and the breach points, with Kupiec and Christoffersen p-values in the title. |
| **4D Stress Surface Matrix** | `data/plots/stress_matrix_4d_surface.png` | Distress probability ($DD \le -40\%$) across annualized volatility and shock severity, averaged over tail tier and asymmetry. |

---

## Getting Started

### Environment Setup

```
# Clone the repository
git clone https://github.com/your-username/VaR-Simulation-Stress-Test.git
cd VaR-Simulation-Stress-Test

# Create and activate Conda environment
conda create -n var-simulation python=3.12 -y
conda activate var-simulation

# Install development dependencies
conda install pytest -y
pip install -e .
```

### Execution

Run the econometric analysis, backtesting suite, and diagnostic chart generator. This downloads price data, so it needs network access:

```
python main.py
```

Rebuild the distressed-entity patches from the raw HTML (no network access needed):

```
rm -f data/patches/*.csv
python src/stress_engine/ingestion.py
```

To run the standalone 4D Monte Carlo tensor stress simulation:

```
python src/stress_engine/monte_carlo.py
```

To cross-check the GJR-GARCH fit against the `arch` package (needs network access):

```
python scripts/compare_gjr_arch.py
```

### Verification

Run the full test suite:

```
pytest -v
```

The tests use synthetic data and do not download prices, so they run offline.

---

## Technical Invariants Tested

### The test suite asserts 35 mechanical, statistical, and data-contract guarantees across `tests/`:

* **Kupiec POF Bounds:** Verifies statistical non-rejection under perfect calibration (50 / 1000 breaches) and deterministic rejection under severe underestimation (150 / 1000 breaches).

* **Christoffersen Markov Transitions:** Verifies clustering detection (p < 0.05) on sequential exceptions and graceful handling of zero-exception boundary states.

* **EWMA Monotonic Decay:** Enforces that conditional variance strictly decreases across consecutive zero-return periods following an initial shock.

* **GJR-GARCH Stationarity and Failure Handling:** Validates non-negativity bounds (ω, α, γ, β > 0), the stationarity constraint (α + β + 0.5γ < 1.0), and that a non-converged optimizer raises an error instead of returning starting values.

* **Out-of-Sample Alignment:** Confirms that the EWMA threshold for day $t$ equals the causal volatility forecast for day $t$, and that GJR-GARCH and FHS thresholds for days before a change in the return series are unaffected by that change.

* **Descriptive Statistics:** Checks that drawdown is computed on compounded wealth (for both log and simple returns), and that the export writes only to the requested directory.

* **Expected Shortfall Coherence:** Asserts that Expected Shortfall strictly exceeds Value-at-Risk (ES > VaR) on non-degenerate distributions.

* **BCBS Basel III Boundaries:** Confirms exact regulatory traffic light zone assignments (Green: 0–4 breaches, Yellow: 5–9 breaches, Red: 10+ breaches) and corresponding capital multiplier penalties (3.00 to 4.00).

* **Tensor Geometry:** Enforces valid probability bounds [0.0, 1.0], negative peak-to-trough drawdowns, and strict shape preservation across the (81, 5000, 90) float32 state space.

* **Ingestion Contracts:** Checks that raw records are parsed (quoted and unquoted prices), malformed prices are rejected, non-positive values and duplicate dates are dropped and counted, monthly and daily cadence are classified correctly, coverage is classified as full, partial, or none, and only covered entities are written.

---

## Known Limitations

* **GJR-GARCH boundary solutions.** On real data, alpha often sits at its lower bound (1e-6) in parts of the sample, and persistence reaches about 0.99 in some windows. The asymmetric term carries most of the shock response there. This is a property of the fitted data, not a convergence failure, but it should be stated wherever GJR-GARCH results are reported.
* **Borderline FHS pass.** The concentrated-financials FHS independence test sits at p = 0.0505, so its "Accept Both" verdict is marginal.
* **Coincident statistics in the multi-asset portfolio.** EWMA and FHS share breach and transition counts, so their independence statistics follow from those counts and are not independent confirmations of each other.
* **Thin evidence in the multi-asset Christoffersen tests.** The independence results rest on a single pair of consecutive breaches in the EWMA and FHS cases.
* **Cross-check limited to three windows.** The `arch` comparison covers three one-year windows, not the full backtest sample.
* **Small event-window samples.** The distressed-entity patches contain 5 to 24 monthly observations per event. No backtest in this repository runs on them yet.
* **Two entities lack event data.** The Signature Bank (SBNY) source covers 2024–2025, not the 2023 failure, and the Chesapeake (CHK) source starts after its 2020 bankruptcy. Both need new sources.
* **Monthly cadence in the distressed data.** Five of six sources are monthly despite their `_daily` file names. Any analysis that uses them has to work at a monthly horizon.
* **Hand-chosen Monte Carlo grid.** The stress tiers are not calibrated to history, so the 4D surface is illustrative only.

---

## Author

James Beall

M.S. Applied Business Analytics | B.S. Theoretical Mathematics

Quantitative Risk Management & Quantitative Finance Portfolio Project