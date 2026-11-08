# Quantitative Risk & Liquidity Stress Testing Engine (`stress_engine`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/typing-PEP%20585%20%7C%20604%20%7C%20646-informational.svg)]()

A high-performance quantitative risk and liquidity stress-testing engine developed to institutional risk standards (Basel Committee on Banking Supervision / FRM Part 1).

The platform evaluates historical asset and portfolio drawdowns, calculates dynamic conditional Value-at-Risk (VaR) and Expected Shortfall (ES), runs formal econometric regulatory backtests, and conducts vectorized Monte Carlo macroeconomic stress simulations across a 4D parametric state space ($36.45\text{M}$ path coordinates in $<3.0\text{s}$).

---

## Key Econometric & Engineering Features

* **Anti-Loop Mandate & Tensor Vectorization:** Zero iterative loops for mathematical transformations, portfolio combinations, simulation paths, or rolling windows. Employs SIMD-aligned C-contiguous memory structures via NumPy and SciPy.
* **Rolling Dynamic Strided Windows ($O(1)$ Allocation):** Leverages `numpy.lib.stride_tricks.sliding_window_view` to construct out-of-sample training/evaluation matrices without memory duplication.
* **Dynamic Conditional Volatility Engines:**
  * **RiskMetrics EWMA ($\lambda=0.94$):** Exponentially decaying memory footprint.
  * **GJR-GARCH(1,1):** Asymmetric leverage estimation via constrained Sequential Least Squares Programming (SLSQP) Maximum Likelihood Estimation (MLE).
  * **Filtered Historical Simulation (FHS):** Non-parametric standardization using empirical standardized innovations $z_t = r_t / \sigma_t$.
* **Regulatory Backtesting Suite:**
  * **Kupiec Proportion of Failures (POF) Test ($LR_{\text{uc}}$):** Unconditional coverage testing.
  * **Christoffersen Markov Independence Test ($LR_{\text{ind}}$):** Exception clustering and first-order Markov transition likelihood ratio testing.
  * **Christoffersen Combined Conditional Coverage ($LR_{\text{cc}}$):** Joint evaluation against $\chi^2(2)$.
  * **Basel Committee (BCBS) Traffic Light System:** Exact binomial cumulative density classification into Green, Yellow, and Red capital penalty zones.
* **Modern Python 3.12+ Standards:** Strict type annotations via `numpy.typing.NDArray[np.float64]`, modern type unions (`|`), explicit axis selections, zero-copy `.to_numpy(dtype=np.float64, copy=False)` operations, and clean `src/` layout installation.

---

## System Architecture

```text
VaR-Simulation-Stress-Test/
├── pyproject.toml                     # PEP 517/518 build specification
├── main.py                            # End-to-end execution pipeline
├── src/
│   └── stress_engine/
│       ├── __init__.py                # Package initialization & exports
│       ├── backtest.py                # Kupiec, Christoffersen, & Basel III engines
│       ├── ingestion.py               # Vectorized HTML/distressed patch parsing
│       ├── monte_carlo.py             # 4D tensor stress testing engine
│       ├── portfolio.py               # PortfolioVaR, rolling OOS, & FHS engines
│       ├── visualization.py           # Publication-grade diagnostic charting
│       └── volatility.py              # EWMA and GJR-GARCH(1,1) MLE solvers
├── tests/
│   └── test_stress_engine.py          # 15 mathematical & structural invariants
├── data/
│   ├── patches/                       # Standardized CSV patches
│   ├── plots/                         # 300 DPI exported diagnostics
│   └── raw/                           # Parquet metrics & master numpy tensors
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

* Subject to parameter constraints:

$$\omega > 0, \quad \alpha \ge 0, \quad \gamma \ge 0, \quad \beta \ge 0, \quad \alpha + \beta + 0.5\gamma < 1.0$$

* Filtered Historical Simulation (FHS)Standardized innovations are filtered from conditional volatility:

$$z_t = \frac{r_t}{\sigma_t}$$

* The dynamic out-of-sample loss cutoff is computed via the empirical quantile of historical residuals:

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

## Empirical Benchmark Findings (2008 Crisis Stress Horizon)

Tested across N = 502 out-of-sample trading days (W = 252 lookback window) during the 2007–2009 Global Financial Crisis at 95% confidence level (p = 0.05, Target Breaches = 25.1):

| Portfolio Regime | Risk Model Engine | Breaches / Obs | Empirical Rate | Kupiec POF (LR_uc) | Christoffersen (LR_ind) | Regulatory Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Financials** *(Sector Meltdown)* | Rolling Historical (252d) | 46 / 502 | 9.16% | 14.861 (*p* &lt; 0.001) | 0.469 (*p* = 0.493) | **Reject Unconditional** |
| | Rolling Parametric (Normal) | 44 / 502 | 8.76% | 12.355 (*p* &lt; 0.001) | 0.248 (*p* = 0.618) | **Reject Unconditional** |
| | **Dynamic EWMA (&lambda;=0.94)** | 34 / 502 | 6.77% | 3.005 (*p* = 0.083) | 1.053 (*p* = 0.305) | **Accept Both** |
| | **Dynamic GJR-GARCH(1,1)** | 34 / 502 | 6.77% | 3.005 (*p* = 0.083) | 1.207 (*p* = 0.272) | **Accept Both** |
| | Filtered Historical (FHS) | 37 / 502 | 7.37% | 5.215 (*p* = 0.022) | 0.030 (*p* = 0.863) | **Reject Unconditional** |
| **Multi-Asset** *(Flight to Safety)* | Rolling Historical (252d) | 29 / 502 | 5.78% | 0.609 (*p* = 0.435) | 12.102 (*p* &lt; 0.001) | **Reject Independence** |
| | Rolling Parametric (Normal) | 31 / 502 | 6.18% | 1.363 (*p* = 0.243) | 7.091 (*p* = 0.008) | **Reject Independence** |
| | **Dynamic EWMA (&lambda;=0.94)** | 26 / 502 | 5.18% | 0.034 (*p* = 0.855) | 0.368 (*p* = 0.544) | **Accept Both** |
| | Dynamic GJR-GARCH(1,1) | 39 / 502 | 7.77% | 6.983 (*p* = 0.008) | 0.001 (*p* = 0.979) | **Reject Unconditional** |
| | **Filtered Historical (FHS)** | 30 / 502 | 5.98% | 0.950 (*p* = 0.330) | 0.043 (*p* = 0.835) | **Accept Both** |

### Takeaways

The In-Sample Quantile Fallacy: In-sample static historical simulation guarantees an exact 5.00% breach rate by mathematical construction, creating a false sense of security while masking severe temporal shock clustering.

Dynamic Volatility Responsiveness: Unweighted rolling windows lag violent market shifts. Dynamic EWMA and GJR-GARCH rapidly scale conditional variance upward, mitigating breach clustering during systemic market panics.

Distributional Breakdown in Multi-Asset Portfolios: Standard GJR-GARCH under normal innovations failed on multi-asset portfolios (7.77% breach rate, p = 0.0082) due to tail thinness. Filtered Historical Simulation resolved the failure (5.98% breach rate, p = 0.3296) by combining GJR-GARCH variance scaling with empirical innovation tails.

---

## 4D Tensor Monte Carlo Simulation Architecture

The macroeconomic stress module simulates an 81-node Cartesian parameter space across:

* Annualized Volatility Tiers ($\sigma_{\text{ann}}$): 10%, 20%, 35%

* Student-$t$ Tail Thickness ($\nu$): 3.5 (Fat), 8.0 (Normal), 30.0 (Thin)

* GJR Asymmetric Leverage ($\gamma$): 0.01, 0.05, 0.15

* Exogenous Macro Shocks ($s$): -15%, -30%, -50% at $t = 30$

Memory & Execution Profile

* Tensor Dimensions: $(K=81\text{ nodes}) \times (P=5,000\text{ paths}) \times (T=90\text{ days})$

* Total Points Evaluated: 36,450,000 single-precision float32 values.

* Contiguous Memory Footprint: $\approx 145.8\text{ MB}$.

* Execution Latency: $< 3.0\text{ seconds}$ on modern multi-core hardware via broadcasted SIMD vectorization.

$$\mathbf{S}_{k, p, t} = 100 \cdot \exp\left( \sum_{\tau=1}^t \mathbf{R}_{k, p, \tau} \right)$$

$$\mathbf{MDD}_{k, p} = \min_{t \in [1, T]} \left( \frac{\mathbf{S}_{k, p, t} - \max_{\tau \le t} \mathbf{S}_{k, p, \tau}}{\max_{\tau \le t} \mathbf{S}_{k, p, \tau}} \right)$$

---

## Diagnostic Figures

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **VaR Backtest Diagnostics** | `data/plots/var_diagnostics_*.png` | Time-series overlay of daily returns, time-varying VaR boundary bands, and exception breach clusters. |
| **4D Stress Surface Matrix** | `data/plots/stress_matrix_4d_surface.png` | Faceted categorical surface contrasting shock severity vs. distress probability ($DD \le -40\%$). |

The visualization pipeline produces publication-quality figures formatted with matplotlib and seaborn:

---

## Getting Started

### Environment Setup
```
# Clone the repository
git clone [https://github.com/your-username/VaR-Simulation-Stress-Test.git](https://github.com/your-username/VaR-Simulation-Stress-Test.git)
cd VaR-Simulation-Stress-Test

# Create and activate Conda environment
conda create -n var-simulation python=3.12 -y
conda activate var-simulation

# Install development dependencies
conda install pytest -y
pip install -e .
```
### Execution
Run the complete econometric analysis, backtesting suite, tensor simulation, and visualization generator:
```
python main.py
```
To run the standalone 4D Monte Carlo tensor stress simulation:
```
python src/stress_engine/monte_carlo.py
```
### Verification
Execute the test suite verifying all 15 mathematical and behavioral invariants:
```
pytest -v
```

---

## Technical Invariants Tested

### The test suite in tests/test_stress_engine.py asserts 15 mechanical and mathematical guarantees:

* Kupiec POF Bounds: Verifies statistical non-rejection under perfect calibration (50 / 1000 breaches) and deterministic rejection under severe underestimation (150 / 1000 breaches).

* Christoffersen Markov Transitions: Verifies clustering detection (p < 0.05) on sequential exceptions and graceful handling of zero-exception boundary states.

* EWMA Monotonic Decay: Enforces that conditional variance strictly decreases across consecutive zero-return periods following an initial shock.

* GJR-GARCH Stationarity: Validates non-negativity bounds (ω, α, γ, β > 0) and verifies the stationarity constraint (α + β + 0.5γ < 1.0).

* Expected Shortfall Coherence: Asserts the subadditivity and severity invariant where Expected Shortfall strictly exceeds Value-at-Risk (ES > VaR) on non-degenerate distributions.

* BCBS Basel III Boundaries: Confirms exact regulatory traffic light zone assignments (Green: 0–4 breaches, Yellow: 5–9 breaches, Red: 10+ breaches) and corresponding capital multiplier penalties (3.00 to 4.00).

* Tensor Geometry: Enforces valid probability bounds [0.0, 1.0], negative peak-to-trough drawdowns, and strict shape preservation across the (81, 5000, 90) float32 state space.

---

## Author

James Beall

M.S. Applied Business Analytics | B.S. Theoretical Mathematics

Quantitative Risk Management & Quantitative Finance Portfolio Project
