"""Monte Carlo Stress Testing Engine (4D Parametric Grid with Full Tensor Broadcast)."""

import itertools
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_parametric_grid_arrays() -> tuple[
    pd.DataFrame,
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Generates the Cartesian parameter space, returning metadata and 3D broadcastable parameter vectors."""
    vol_tiers = {"Vol-Low": 0.10, "Vol-Norm": 0.20, "Vol-High": 0.35}
    tail_tiers = {"Tail-Fat": 3.5, "Tail-Norm": 8.0, "Tail-Thin": 30.0}
    asym_tiers = {"Asym-Low": 0.01, "Asym-Norm": 0.05, "Asym-High": 0.15}
    shock_tiers = {"Shock-Mild": -0.15, "Shock-Mod": -0.30, "Shock-Sev": -0.50}

    product_generator = itertools.product(
        vol_tiers.items(),
        tail_tiers.items(),
        asym_tiers.items(),
        shock_tiers.items(),
    )

    records = [
        {
            "Node_Index": idx,
            "Institution_Key": f"{v_k}_{t_k}_{a_k}_{s_k}",
            "Volatility": v_k,
            "annual_vol": v_v,
            "Tail_Tier": t_k,
            "t_df": t_v,
            "Asymmetry": a_k,
            "gjr_gamma": a_v,
            "Shock_Severity": s_k,
            "base_shock": s_v,
        }
        for idx, ((v_k, v_v), (t_k, t_v), (a_k, a_v), (s_k, s_v)) in enumerate(
            product_generator
        )
    ]

    df_meta = pd.DataFrame(records)

    # Reshape parameter arrays to (K, 1, 1) for contiguous 3D broadcasting
    k_nodes = len(df_meta)
    vol_arr = df_meta["annual_vol"].to_numpy(dtype=np.float32).reshape(k_nodes, 1, 1)
    tail_arr = df_meta["t_df"].to_numpy(dtype=np.float32).reshape(k_nodes, 1, 1)
    asym_arr = df_meta["gjr_gamma"].to_numpy(dtype=np.float32).reshape(k_nodes, 1, 1)
    shock_arr = df_meta["base_shock"].to_numpy(dtype=np.float32).reshape(k_nodes, 1, 1)

    return df_meta, vol_arr, tail_arr, asym_arr, shock_arr


def run_comprehensive_stress_engine(
    n_paths: int = 5000,
    n_days: int = 90,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, npt.NDArray[np.float32]]:
    """Simulates the entire 4D parameter space simultaneously via tensor broadcasting.

    Writes the metrics parquet and the path tensor to ``output_dir`` (default: the
    project's ``data/raw``). Tests pass a temporary directory so they never touch
    tracked files.
    """
    start_time = time.time()
    df_meta, vol_3d, tail_3d, asym_3d, shock_3d = build_parametric_grid_arrays()
    n_nodes = len(df_meta)

    print(
        f"Simulating 4D Tensor ({n_nodes} nodes x {n_paths} paths x {n_days} days) "
        f"via broadcasted SIMD operations..."
    )

    rng = np.random.default_rng(random_seed)

    # 1. Vectorized diffusion generation: Shape (81, 5000, 90)
    daily_vol = vol_3d / np.float32(np.sqrt(252.0))
    # rng.standard_t accepts (81, 1, 1) df array and broadcasts across (81, 5000, 90)
    innovations = rng.standard_t(df=tail_3d, size=(n_nodes, n_paths, n_days)).astype(
        np.float32
    )
    returns_tensor = daily_vol * innovations

    # 2. Vectorized shock injection at day 30: Shape (81, 1, 1)
    shock_day = 30
    if shock_day < n_days:
        scaled_shock = (
            shock_3d
            * (np.float32(1.0) + asym_3d * np.float32(10.0))
            * (np.float32(6.0) / tail_3d)
        )
        returns_tensor[:, :, shock_day] += scaled_shock[:, :, 0]

    # 3. Geometric price evolution: Shape (81, 5000, 90)
    price_paths = np.float32(100.0) * np.exp(np.cumsum(returns_tensor, axis=2))

    # 4. Maximum Drawdown Tensor Reduction
    running_max = np.maximum.accumulate(price_paths, axis=2)
    drawdowns = (price_paths - running_max) / running_max
    max_dds = np.min(drawdowns, axis=2)  # Shape: (81, 5000)

    # 5. Summary Statistics computed across paths (axis 1)
    df_meta["Mean_Max_DD"] = np.mean(max_dds, axis=1)
    df_meta["P95_Max_DD"] = np.percentile(max_dds, 5.0, axis=1)
    df_meta["Distress_Probability"] = np.mean(max_dds <= -0.40, axis=1)

    # Canonical projection of results
    output_columns = [
        "Node_Index",
        "Institution_Key",
        "Volatility",
        "Tail_Tier",
        "Asymmetry",
        "Shock_Severity",
        "Mean_Max_DD",
        "P95_Max_DD",
        "Distress_Probability",
    ]
    df_results = df_meta.loc[:, output_columns]

    # Save artifacts to output_dir (default: data/raw/ in the project root)
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "raw"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_output_path = output_dir / "synthetic_4d_monte_carlo_results.parquet"
    tensor_output_path = output_dir / "master_path_tensor.npy"

    df_results.to_parquet(df_output_path, index=False)
    np.save(tensor_output_path, price_paths)

    elapsed = time.time() - start_time
    print(
        f"Simulation complete in {elapsed:.2f}s.\n"
        f"  - Metrics Parquet: {df_output_path} ({len(df_results)} rows)\n"
        f"  - Master Tensor:   {tensor_output_path} (Shape: {price_paths.shape}, Dtype: {price_paths.dtype})"
    )

    return df_results, price_paths


if __name__ == "__main__":
    run_comprehensive_stress_engine()
