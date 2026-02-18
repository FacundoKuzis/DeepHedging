"""
Interactive diffusion simulator runner using strict JSON configs.

How it works:
1. Ask by console for a JSON file name (name only) in ./gan_training_configs.
2. Validate config completeness (no defaults, no missing keys).
3. Copy that same JSON file into the run output folder.
4. Train/load diffusion model on train range, evaluate against test range, and save outputs.

All artifacts are saved under:
G:/Mi unidad/Tesis2026/TimeGanTraining/<json_file_stem>/
"""

import json
import os
import shutil
import sys
import time
import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work without installing the package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.HedgingInstruments import DiffusionStock
from DeepHedging.utils.timegan_metrics import (
    cumulative_log_returns,
    dependence_metrics_df,
    distribution_vectors_df,
    max_drawdown,
    path_risk_metrics_df,
    tail_metrics_df,
)


def get_output_root() -> str:
    return os.path.normpath(
        os.environ.get("TIMEGAN_OUTPUT_ROOT", r"G:/Mi unidad/Tesis2026/TimeGanTraining")
    )


def normalize_paths(paths: np.ndarray, base: float) -> np.ndarray:
    paths = np.asarray(paths, dtype=np.float64)
    first = np.maximum(paths[:, :1], 1e-8)
    out = base * (paths / first)
    out[:, 0] = base
    return out


def one_step_returns(paths: np.ndarray) -> np.ndarray:
    return paths[:, 1:] / np.maximum(paths[:, :-1], 1e-8) - 1.0


def one_step_log_returns(paths: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(paths[:, 1:], 1e-8) / np.maximum(paths[:, :-1], 1e-8))


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    return paths[:, -1] / np.maximum(paths[:, 0], 1e-8) - 1.0


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return float("nan")
    m = np.mean(x)
    s = np.std(x)
    if s <= 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def kurtosis_excess(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4:
        return float("nan")
    m = np.mean(x)
    s = np.std(x)
    if s <= 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def rolling_vol_distribution(paths: np.ndarray, window: int) -> np.ndarray:
    log_r = one_step_log_returns(paths)
    if log_r.shape[1] < window:
        return np.array([], dtype=np.float64)

    vols = []
    for row in log_r:
        row_series = pd.Series(row)
        rolling = row_series.rolling(window=window).std().dropna().to_numpy()
        vols.append(rolling)

    if not vols:
        return np.array([], dtype=np.float64)
    return np.concatenate(vols).astype(np.float64)


def acf_for_lags(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return np.zeros(max_lag, dtype=np.float64)
    out = np.zeros(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        if lag >= x.size:
            out[lag - 1] = np.nan
            continue
        out[lag - 1] = np.mean(x[:-lag] * x[lag:]) / var
    return out


def mean_acf(paths: np.ndarray, max_lag: int) -> np.ndarray:
    log_r = one_step_log_returns(paths)
    all_acf = np.array([acf_for_lags(row, max_lag) for row in log_r], dtype=np.float64)
    return np.nanmean(all_acf, axis=0)


def histogram_l1_distance(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    lower = min(np.min(a), np.min(b))
    upper = max(np.max(a), np.max(b))
    hist_a, edges = np.histogram(a, bins=bins, range=(lower, upper), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.abs(hist_a - hist_b) * widths))


def plot_overlay_paths(
    real_paths_norm: np.ndarray,
    synth_paths_norm: np.ndarray,
    output_path: str,
    n_plot_paths: int,
) -> None:
    plt.figure(figsize=(10, 6))
    for i in range(min(n_plot_paths, real_paths_norm.shape[0])):
        plt.plot(real_paths_norm[i], color="tab:blue", alpha=0.35, linewidth=1)
    for i in range(min(n_plot_paths, synth_paths_norm.shape[0])):
        plt.plot(synth_paths_norm[i], color="tab:orange", alpha=0.35, linewidth=1, linestyle="--")
    plt.title("Normalized Paths Overlay (base=100)")
    plt.xlabel("Time step")
    plt.ylabel("Normalized price")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hist(
    real: np.ndarray,
    synth: np.ndarray,
    title: str,
    xlabel: str,
    output_path: str,
    hist_bins: int,
) -> None:
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    synth = np.asarray(synth, dtype=np.float64).reshape(-1)
    lower = min(np.min(real), np.min(synth))
    upper = max(np.max(real), np.max(synth))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Non-finite values found while plotting histogram '{title}'.")
    if upper <= lower:
        upper = lower + 1e-12
    edges = np.linspace(lower, upper, hist_bins + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(real, bins=edges, alpha=0.6, density=True, label="Real")
    plt.hist(synth, bins=edges, alpha=0.6, density=True, label="Synthetic")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_acf(real_acf: np.ndarray, synth_acf: np.ndarray, output_path: str) -> None:
    lags = np.arange(1, len(real_acf) + 1)
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(lags - width / 2, real_acf, width=width, alpha=0.8, label="Real")
    plt.bar(lags + width / 2, synth_acf, width=width, alpha=0.8, label="Synthetic")
    plt.title("Mean ACF of One-Step Log Returns")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.xticks(lags)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_summary_metrics(
    real_paths: np.ndarray,
    synth_paths: np.ndarray,
    acf_max_lag: int,
    hist_bins: int,
) -> pd.DataFrame:
    real_r = one_step_returns(real_paths).reshape(-1)
    synth_r = one_step_returns(synth_paths).reshape(-1)
    real_term = terminal_returns(real_paths)
    synth_term = terminal_returns(synth_paths)
    real_acf = mean_acf(real_paths, acf_max_lag)
    synth_acf = mean_acf(synth_paths, acf_max_lag)

    rows = [
        {"metric": "daily_return_mean", "real": float(np.mean(real_r)), "synthetic": float(np.mean(synth_r))},
        {"metric": "daily_return_std", "real": float(np.std(real_r)), "synthetic": float(np.std(synth_r))},
        {"metric": "daily_return_skew", "real": skewness(real_r), "synthetic": skewness(synth_r)},
        {
            "metric": "daily_return_kurtosis_excess",
            "real": kurtosis_excess(real_r),
            "synthetic": kurtosis_excess(synth_r),
        },
        {"metric": "terminal_return_mean", "real": float(np.mean(real_term)), "synthetic": float(np.mean(synth_term))},
        {"metric": "terminal_return_std", "real": float(np.std(real_term)), "synthetic": float(np.std(synth_term))},
        {"metric": "terminal_hist_l1", "real": 0.0, "synthetic": histogram_l1_distance(real_term, synth_term, bins=hist_bins)},
        {
            "metric": "one_step_log_return_hist_l1",
            "real": 0.0,
            "synthetic": histogram_l1_distance(
                one_step_log_returns(real_paths).reshape(-1),
                one_step_log_returns(synth_paths).reshape(-1),
                bins=hist_bins,
            ),
        },
    ]

    acf_abs_err = np.abs(real_acf - synth_acf)
    rows.append(
        {"metric": "acf_abs_error_mean_lag_1_10", "real": 0.0, "synthetic": float(np.nanmean(acf_abs_err))}
    )
    for lag_idx, lag_err in enumerate(acf_abs_err, start=1):
        rows.append({"metric": f"acf_abs_error_lag_{lag_idx}", "real": 0.0, "synthetic": float(lag_err)})
    return pd.DataFrame(rows)


def matrix_to_long_df(
    matrix: np.ndarray,
    split: str,
    source: str,
    value_name: str,
    step_name: str = "timestep",
    step_start: int = 0,
) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D matrix for '{value_name}', got shape {values.shape}.")

    n_paths, n_steps = values.shape
    return pd.DataFrame(
        {
            "split": split,
            "source": source,
            "path_id": np.repeat(np.arange(n_paths), n_steps).astype(np.int64),
            step_name: np.tile(np.arange(step_start, step_start + n_steps), n_paths).astype(np.int64),
            value_name: values.reshape(-1),
        }
    )


def vector_to_df(
    values: np.ndarray,
    split: str,
    source: str,
    value_name: str,
    index_name: str,
    index_start: int = 0,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return pd.DataFrame(
        {
            "split": split,
            "source": source,
            index_name: np.arange(index_start, index_start + arr.size, dtype=np.int64),
            value_name: arr,
        }
    )


def histogram_density_df(real: np.ndarray, synth: np.ndarray, bins: int, split: str, metric_name: str) -> pd.DataFrame:
    real_v = np.asarray(real, dtype=np.float64).reshape(-1)
    synth_v = np.asarray(synth, dtype=np.float64).reshape(-1)
    lower = min(np.min(real_v), np.min(synth_v))
    upper = max(np.max(real_v), np.max(synth_v))
    hist_real, edges = np.histogram(real_v, bins=bins, range=(lower, upper), density=True)
    hist_synth, _ = np.histogram(synth_v, bins=edges, density=True)
    return pd.DataFrame(
        {
            "split": split,
            "metric": metric_name,
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "real_density": hist_real,
            "synthetic_density": hist_synth,
            "abs_diff_density": np.abs(hist_real - hist_synth),
        }
    )


def evaluate_split_artifacts(
    real_paths_norm: np.ndarray,
    synth_paths_norm: np.ndarray,
    split: str,
    acf_max_lag: int,
    hist_bins: int,
    rolling_vol_window: int,
) -> dict[str, Any]:
    real_simple_returns = one_step_returns(real_paths_norm)
    synth_simple_returns = one_step_returns(synth_paths_norm)
    real_log_returns = one_step_log_returns(real_paths_norm)
    synth_log_returns = one_step_log_returns(synth_paths_norm)
    real_terminal = terminal_returns(real_paths_norm)
    synth_terminal = terminal_returns(synth_paths_norm)
    real_roll_vol = rolling_vol_distribution(real_paths_norm, window=rolling_vol_window)
    synth_roll_vol = rolling_vol_distribution(synth_paths_norm, window=rolling_vol_window)
    real_acf = mean_acf(real_paths_norm, acf_max_lag)
    synth_acf = mean_acf(synth_paths_norm, acf_max_lag)

    summary = build_summary_metrics(
        real_paths=real_paths_norm,
        synth_paths=synth_paths_norm,
        acf_max_lag=acf_max_lag,
        hist_bins=hist_bins,
    )
    summary.insert(0, "split", split)

    acf_df = pd.DataFrame(
        {
            "split": split,
            "lag": np.arange(1, len(real_acf) + 1, dtype=np.int64),
            "real_acf": real_acf,
            "synthetic_acf": synth_acf,
            "abs_error": np.abs(real_acf - synth_acf),
        }
    )

    return {
        "summary": summary,
        "real_paths_norm": real_paths_norm,
        "synth_paths_norm": synth_paths_norm,
        "real_simple_returns": real_simple_returns,
        "synth_simple_returns": synth_simple_returns,
        "real_log_returns": real_log_returns,
        "synth_log_returns": synth_log_returns,
        "real_terminal_returns": real_terminal,
        "synth_terminal_returns": synth_terminal,
        "real_rolling_vol": real_roll_vol,
        "synth_rolling_vol": synth_roll_vol,
        "real_acf": real_acf,
        "synth_acf": synth_acf,
        "acf_df": acf_df,
        "terminal_hist_df": histogram_density_df(real_terminal, synth_terminal, bins=hist_bins, split=split, metric_name="terminal_returns"),
        "log_return_hist_df": histogram_density_df(
            real_log_returns.reshape(-1),
            synth_log_returns.reshape(-1),
            bins=hist_bins,
            split=split,
            metric_name="one_step_log_returns",
        ),
        "rolling_vol_hist_df": histogram_density_df(
            real_roll_vol,
            synth_roll_vol,
            bins=hist_bins,
            split=split,
            metric_name=f"rolling_vol_window_{rolling_vol_window}",
        ),
    }


def save_split_data(split_data: dict[str, Any], data_dir: str) -> None:
    split = str(split_data["summary"]["split"].iloc[0])
    split_dir = os.path.join(data_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    matrix_to_long_df(
        split_data["real_paths_norm"],
        split=split,
        source="real",
        value_name="price",
    ).to_csv(os.path.join(split_dir, "real_paths_normalized.csv"), index=False)
    matrix_to_long_df(
        split_data["synth_paths_norm"],
        split=split,
        source="synthetic",
        value_name="price",
    ).to_csv(os.path.join(split_dir, "synthetic_paths_normalized.csv"), index=False)

    matrix_to_long_df(
        split_data["real_simple_returns"],
        split=split,
        source="real",
        value_name="simple_return",
        step_name="return_step",
        step_start=1,
    ).to_csv(os.path.join(split_dir, "real_one_step_simple_returns.csv"), index=False)
    matrix_to_long_df(
        split_data["synth_simple_returns"],
        split=split,
        source="synthetic",
        value_name="simple_return",
        step_name="return_step",
        step_start=1,
    ).to_csv(os.path.join(split_dir, "synthetic_one_step_simple_returns.csv"), index=False)

    matrix_to_long_df(
        split_data["real_log_returns"],
        split=split,
        source="real",
        value_name="log_return",
        step_name="return_step",
        step_start=1,
    ).to_csv(os.path.join(split_dir, "real_one_step_log_returns.csv"), index=False)
    matrix_to_long_df(
        split_data["synth_log_returns"],
        split=split,
        source="synthetic",
        value_name="log_return",
        step_name="return_step",
        step_start=1,
    ).to_csv(os.path.join(split_dir, "synthetic_one_step_log_returns.csv"), index=False)

    vector_to_df(
        split_data["real_terminal_returns"],
        split=split,
        source="real",
        value_name="terminal_return",
        index_name="path_id",
    ).to_csv(os.path.join(split_dir, "real_terminal_returns.csv"), index=False)
    vector_to_df(
        split_data["synth_terminal_returns"],
        split=split,
        source="synthetic",
        value_name="terminal_return",
        index_name="path_id",
    ).to_csv(os.path.join(split_dir, "synthetic_terminal_returns.csv"), index=False)

    vector_to_df(
        split_data["real_rolling_vol"],
        split=split,
        source="real",
        value_name="rolling_vol",
        index_name="obs_id",
    ).to_csv(os.path.join(split_dir, "real_rolling_volatility_values.csv"), index=False)
    vector_to_df(
        split_data["synth_rolling_vol"],
        split=split,
        source="synthetic",
        value_name="rolling_vol",
        index_name="obs_id",
    ).to_csv(os.path.join(split_dir, "synthetic_rolling_volatility_values.csv"), index=False)

    split_data["acf_df"].to_csv(os.path.join(split_dir, "acf_values.csv"), index=False)
    split_data["terminal_hist_df"].to_csv(os.path.join(split_dir, "hist_terminal_returns.csv"), index=False)
    split_data["log_return_hist_df"].to_csv(os.path.join(split_dir, "hist_one_step_log_returns.csv"), index=False)
    split_data["rolling_vol_hist_df"].to_csv(os.path.join(split_dir, "hist_rolling_volatility.csv"), index=False)
    split_data["summary"].to_csv(os.path.join(split_dir, "summary_metrics.csv"), index=False)


def save_training_scores_available(
    output_path: str,
    run_name: str,
    config: dict[str, Any],
    model_exists_before: bool,
    model_exists_after: bool,
    train_csv_exists_before: bool,
    train_csv_exists_after: bool,
    test_csv_exists_before: bool,
    test_csv_exists_after: bool,
    train_generate_seconds: float,
    train_instrument: DiffusionStock,
) -> None:
    rows = [
        {"scope": "training", "metric": "model_exists_before_run", "value": int(model_exists_before)},
        {"scope": "training", "metric": "model_exists_after_run", "value": int(model_exists_after)},
        {"scope": "training", "metric": "train_csv_exists_before_run", "value": int(train_csv_exists_before)},
        {"scope": "training", "metric": "train_csv_exists_after_run", "value": int(train_csv_exists_after)},
        {"scope": "testing", "metric": "test_csv_exists_before_run", "value": int(test_csv_exists_before)},
        {"scope": "testing", "metric": "test_csv_exists_after_run", "value": int(test_csv_exists_after)},
        {"scope": "training", "metric": "train_generate_paths_wallclock_seconds", "value": float(train_generate_seconds)},
        {"scope": "training", "metric": "schema_version", "value": int(config["schema_version"])},
        {"scope": "training", "metric": "model_family", "value": str(config["model_family"])},
        {"scope": "training", "metric": "return_transform", "value": str(config["return_transform"])},
        {"scope": "training", "metric": "feature_mode", "value": str(config["feature_mode"])},
        {"scope": "training", "metric": "train_epochs_requested", "value": int(config["train_epochs"])},
        {"scope": "training", "metric": "batch_size_requested", "value": int(config["batch_size"])},
        {"scope": "training", "metric": "learning_rate_requested", "value": float(config["learning_rate"])},
        {"scope": "training", "metric": "weight_decay_requested", "value": float(config["weight_decay"])},
        {"scope": "training", "metric": "grad_clip_norm_requested", "value": float(config["grad_clip_norm"])},
        {"scope": "training", "metric": "diffusion_steps", "value": int(config["diffusion_steps"])},
        {"scope": "training", "metric": "beta_schedule", "value": str(config["beta_schedule"])},
        {"scope": "training", "metric": "sampler_type", "value": str(config["sampler_type"])},
        {"scope": "training", "metric": "sample_steps", "value": int(config["sample_steps"])},
        {"scope": "training", "metric": "stride", "value": int(config["stride"])},
        {"scope": "training", "metric": "min_windows", "value": int(config["min_windows"])},
        {"scope": "training", "metric": "training_target", "value": str(config["training_target"])},
        {"scope": "training", "metric": "estimated_sigma_from_train_series", "value": float(train_instrument.sigma)},
        {"scope": "training", "metric": "train_window_count_used_by_instrument", "value": int(train_instrument._real_windows.shape[0])},
        {"scope": "training", "metric": "train_series_observations", "value": int(train_instrument._series.shape[0])},
        {"scope": "meta", "metric": "run_name", "value": run_name},
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def required_keys_v1() -> set[str]:
    return {
        "schema_version",
        "model_family",
        "s0",
        "t",
        "n",
        "r",
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "interval",
        "price_col",
        "download_if_missing",
        "retrain",
        "stride",
        "min_windows",
        "random_seed",
        "training_target",
        "feature_mode",
        "feature_rolling_vol_window",
        "return_transform",
        "transform_eps",
        "legacy_return_clip",
        "train_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "use_ema",
        "ema_decay",
        "diffusion_steps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "model_hidden_dim",
        "model_num_res_blocks",
        "model_dropout",
        "time_embedding_dim",
        "sampler_type",
        "sample_steps",
        "ddim_eta",
        "n_synth_paths",
        "n_real_windows_compare",
        "n_plot_paths",
        "rolling_vol_window",
        "acf_max_lag",
        "hist_bins",
        "normalization_base",
        "eval_tail_quantiles",
        "eval_exceedance_thresholds",
        "eval_metrics_version",
    }


def validate_float_list(value: Any, key: str, min_len: int, lower_bound: float, upper_bound: float) -> None:
    if not isinstance(value, list) or len(value) < min_len:
        raise ValueError(f"'{key}' must be a list with at least {min_len} values.")
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"'{key}' must contain only numeric values.")
        num = float(item)
        if not (lower_bound < num < upper_bound):
            raise ValueError(f"'{key}' values must satisfy {lower_bound} < x < {upper_bound}. Got {num}.")


def validate_positive_float_list(value: Any, key: str, min_len: int) -> None:
    if not isinstance(value, list) or len(value) < min_len:
        raise ValueError(f"'{key}' must be a list with at least {min_len} values.")
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"'{key}' must contain only numeric values.")
        num = float(item)
        if num <= 0:
            raise ValueError(f"'{key}' values must be > 0. Got {num}.")


def validate_config(config: dict[str, Any]) -> None:
    req = required_keys_v1()
    keys = set(config.keys())

    missing = sorted(req - keys)
    extra = sorted(keys - req)
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if extra:
        raise ValueError(f"Unknown keys in config: {extra}")

    if int(config["schema_version"]) != 1:
        raise ValueError("schema_version must be integer 1 for diffusion runner.")
    if str(config["model_family"]).strip().lower() != "diffusion":
        raise ValueError("model_family must be 'diffusion'.")

    bool_keys = {"download_if_missing", "retrain", "legacy_return_clip", "use_ema"}
    for key in bool_keys:
        if not isinstance(config[key], bool):
            raise ValueError(f"'{key}' must be boolean.")

    int_positive = {
        "n",
        "stride",
        "min_windows",
        "train_epochs",
        "batch_size",
        "random_seed",
        "feature_rolling_vol_window",
        "diffusion_steps",
        "model_hidden_dim",
        "model_num_res_blocks",
        "time_embedding_dim",
        "sample_steps",
        "n_synth_paths",
        "n_real_windows_compare",
        "n_plot_paths",
        "rolling_vol_window",
        "acf_max_lag",
        "hist_bins",
    }
    for key in int_positive:
        value = config[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"'{key}' must be a positive integer.")

    float_positive = {
        "s0",
        "t",
        "learning_rate",
        "grad_clip_norm",
        "ema_decay",
        "beta_start",
        "beta_end",
        "normalization_base",
        "transform_eps",
    }
    for key in float_positive:
        value = config[key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be numeric.")
        if float(value) <= 0:
            raise ValueError(f"'{key}' must be > 0.")
    float_non_negative = {"r", "weight_decay", "model_dropout", "ddim_eta"}
    for key in float_non_negative:
        value = config[key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be numeric.")
        if float(value) < 0:
            raise ValueError(f"'{key}' must be >= 0.")

    string_keys = {
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "interval",
        "price_col",
        "training_target",
        "feature_mode",
        "return_transform",
        "beta_schedule",
        "sampler_type",
        "eval_metrics_version",
    }
    for key in string_keys:
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{key}' must be a non-empty string.")

    if str(config["training_target"]).strip().lower() != "log_returns":
        raise ValueError("training_target must be 'log_returns' for diffusion runner v1.")
    if str(config["feature_mode"]).strip().lower() not in {
        "returns_only",
        "returns_plus_abs_return",
        "returns_plus_rolling_vol",
    }:
        raise ValueError(
            "feature_mode must be one of {'returns_only','returns_plus_abs_return','returns_plus_rolling_vol'}."
        )
    if str(config["return_transform"]).strip().lower() not in {"minmax", "gaussian_cdf", "empirical_cdf"}:
        raise ValueError("return_transform must be one of {'minmax','gaussian_cdf','empirical_cdf'}.")
    if str(config["beta_schedule"]).strip().lower() not in {"linear", "cosine"}:
        raise ValueError("beta_schedule must be one of {'linear','cosine'}.")
    if str(config["sampler_type"]).strip().lower() not in {"ddpm", "ddim"}:
        raise ValueError("sampler_type must be one of {'ddpm','ddim'}.")
    if str(config["eval_metrics_version"]).strip().lower() != "v2":
        raise ValueError("eval_metrics_version must be 'v2'.")

    pd.to_datetime(config["train_start_date"], format="%Y-%m-%d")
    pd.to_datetime(config["train_end_date"], format="%Y-%m-%d")
    pd.to_datetime(config["test_start_date"], format="%Y-%m-%d")
    pd.to_datetime(config["test_end_date"], format="%Y-%m-%d")

    if pd.Timestamp(config["train_start_date"]) > pd.Timestamp(config["train_end_date"]):
        raise ValueError("train_start_date must be <= train_end_date.")
    if pd.Timestamp(config["test_start_date"]) > pd.Timestamp(config["test_end_date"]):
        raise ValueError("test_start_date must be <= test_end_date.")
    if config["acf_max_lag"] >= config["n"]:
        raise ValueError("acf_max_lag must be < n.")

    eps = float(config["transform_eps"])
    if not (0.0 < eps < 0.5):
        raise ValueError("transform_eps must satisfy 0 < transform_eps < 0.5.")
    if float(config["model_dropout"]) >= 1.0:
        raise ValueError("model_dropout must satisfy 0 <= model_dropout < 1.")
    if not (0.0 <= float(config["ddim_eta"]) <= 1.0):
        raise ValueError("ddim_eta must satisfy 0 <= ddim_eta <= 1.")
    if float(config["ema_decay"]) >= 1.0:
        raise ValueError("ema_decay must satisfy 0 < ema_decay < 1.")
    if str(config["beta_schedule"]).strip().lower() == "linear" and float(config["beta_end"]) <= float(config["beta_start"]):
        raise ValueError("For linear schedule, beta_end must be > beta_start.")
    if int(config["sample_steps"]) > int(config["diffusion_steps"]):
        raise ValueError("sample_steps must be <= diffusion_steps.")

    validate_float_list(
        config["eval_tail_quantiles"],
        key="eval_tail_quantiles",
        min_len=2,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    validate_positive_float_list(
        config["eval_exceedance_thresholds"],
        key="eval_exceedance_thresholds",
        min_len=1,
    )


def normalize_config_name(user_input: str) -> str:
    name = str(user_input).strip()
    if not name:
        raise ValueError("Config file name cannot be empty.")
    if os.path.basename(name) != name:
        raise ValueError("Provide only the file name (no paths).")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


def load_config(configs_dir: str, config_name: str | None = None) -> tuple[str, str, dict[str, Any]]:
    if config_name is None:
        user_input = input(
            "Enter JSON config file name from 'gan_training_configs' (example: my_run.json): "
        ).strip()
    else:
        user_input = str(config_name).strip()
    config_filename = normalize_config_name(user_input)
    config_path = os.path.join(configs_dir, config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config JSON must be an object/dictionary.")
    run_name = os.path.splitext(config_filename)[0]
    return run_name, config_path, config


def parse_float_list(value: Any) -> list[float]:
    return [float(x) for x in value]


def run_simulation(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    output_root = get_output_root()
    if not os.path.exists(output_root):
        raise FileNotFoundError(
            f"Output root does not exist: {output_root}. "
            "Mount/create it first and rerun."
        )

    run_dir = os.path.join(output_root, run_name)
    plots_dir = os.path.join(run_dir, "plots")
    csvs_dir = os.path.join(run_dir, "csvs")
    models_dir = os.path.join(run_dir, "models")
    data_dir = os.path.join(csvs_dir, "plot_inputs")
    scores_dir = os.path.join(csvs_dir, "scores")
    market_cache_dir = os.path.join(output_root, "market_data_cache")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(market_cache_dir, exist_ok=True)

    config_copy_path = os.path.join(run_dir, os.path.basename(config_path))
    shutil.copy2(config_path, config_copy_path)
    print(f"[run:{run_name}] Config validated and copied to: {config_copy_path}")
    print(
        f"[run:{run_name}] schema_version={config['schema_version']}, "
        f"model_family={config['model_family']}"
    )

    ticker = config["ticker"]
    interval = config["interval"]
    train_csv_filename = (
        f"{ticker}_{config['train_start_date']}_{config['train_end_date']}_{interval}.csv".replace(":", "-")
    )
    test_csv_filename = (
        f"{ticker}_{config['test_start_date']}_{config['test_end_date']}_{interval}.csv".replace(":", "-")
    )

    train_csv_path = os.path.join(market_cache_dir, train_csv_filename)
    test_csv_path = os.path.join(market_cache_dir, test_csv_filename)
    model_dir = os.path.join(models_dir, "diffusion")
    summary_all_metrics_path = os.path.join(csvs_dir, "summary_metrics_all_splits.csv")
    summary_test_legacy_path = os.path.join(csvs_dir, "summary_metrics.csv")
    train_metrics_path = os.path.join(scores_dir, "summary_metrics_train.csv")
    test_metrics_path = os.path.join(scores_dir, "summary_metrics_test.csv")
    training_scores_path = os.path.join(scores_dir, "training_scores_available.csv")
    train_manifest_path = os.path.join(scores_dir, "train_manifest.csv")
    dep_returns_path = os.path.join(scores_dir, "dependence_metrics_returns.csv")
    dep_squared_path = os.path.join(scores_dir, "dependence_metrics_squared_returns.csv")
    tail_metrics_path = os.path.join(scores_dir, "tail_metrics.csv")
    path_risk_metrics_path = os.path.join(scores_dir, "path_risk_metrics.csv")
    training_history_path = os.path.join(scores_dir, "training_loss_history.csv")
    noise_schedule_path = os.path.join(scores_dir, "noise_schedule.csv")

    model_exists_before = os.path.exists(os.path.join(model_dir, "diffusion_denoiser.keras"))
    train_csv_exists_before = os.path.exists(train_csv_path)
    test_csv_exists_before = os.path.exists(test_csv_path)

    print(f"[run:{run_name}] Using training market data cache: {train_csv_path}")
    print(f"[run:{run_name}] Using test market data cache: {test_csv_path}")
    print(f"[run:{run_name}] Model dir: {model_dir}")

    print(f"[run:{run_name}] Building train instrument and training/loading Diffusion...")
    train_instrument = DiffusionStock(
        S0=float(config["s0"]),
        T=float(config["t"]),
        N=int(config["n"]),
        r=float(config["r"]),
        ticker=ticker,
        start_date=config["train_start_date"],
        end_date=config["train_end_date"],
        interval=interval,
        price_col=config["price_col"],
        csv_path=train_csv_path,
        model_dir=model_dir,
        download_if_missing=bool(config["download_if_missing"]),
        retrain=bool(config["retrain"]),
        stride=int(config["stride"]),
        random_seed=int(config["random_seed"]),
        min_windows=int(config["min_windows"]),
        train_epochs=int(config["train_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        use_ema=bool(config["use_ema"]),
        ema_decay=float(config["ema_decay"]),
        training_target=str(config["training_target"]),
        return_transform=str(config["return_transform"]),
        transform_eps=float(config["transform_eps"]),
        feature_mode=str(config["feature_mode"]),
        feature_rolling_vol_window=int(config["feature_rolling_vol_window"]),
        legacy_return_clip=bool(config["legacy_return_clip"]),
        diffusion_steps=int(config["diffusion_steps"]),
        beta_schedule=str(config["beta_schedule"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
        model_hidden_dim=int(config["model_hidden_dim"]),
        model_num_res_blocks=int(config["model_num_res_blocks"]),
        model_dropout=float(config["model_dropout"]),
        time_embedding_dim=int(config["time_embedding_dim"]),
        sampler_type=str(config["sampler_type"]),
        sample_steps=int(config["sample_steps"]),
        ddim_eta=float(config["ddim_eta"]),
    )
    train_gen_start = time.perf_counter()
    synth_paths = train_instrument.generate_paths(
        int(config["n_synth_paths"]),
        random_seed=int(config["random_seed"]),
    ).numpy()
    train_generate_seconds = time.perf_counter() - train_gen_start
    model_exists_after = os.path.exists(os.path.join(model_dir, "diffusion_denoiser.keras"))
    train_csv_exists_after = os.path.exists(train_csv_path)
    print(f"[run:{run_name}] Synthetic paths generated: {synth_paths.shape}")
    print(f"[run:{run_name}] Training/generation wallclock seconds: {train_generate_seconds:.3f}")

    manifest_df = train_instrument.get_training_manifest()
    manifest_df.to_csv(train_manifest_path, index=False)
    train_instrument.get_training_history().to_csv(training_history_path, index=False)
    train_instrument.get_noise_schedule().to_csv(noise_schedule_path, index=False)

    print(f"[run:{run_name}] Loading train/test real windows for in-sample and out-of-sample comparison...")
    test_instrument = DiffusionStock(
        S0=float(config["s0"]),
        T=float(config["t"]),
        N=int(config["n"]),
        r=float(config["r"]),
        ticker=ticker,
        start_date=config["test_start_date"],
        end_date=config["test_end_date"],
        interval=interval,
        price_col=config["price_col"],
        csv_path=test_csv_path,
        model_dir=model_dir,
        download_if_missing=bool(config["download_if_missing"]),
        retrain=False,
        stride=int(config["stride"]),
        random_seed=int(config["random_seed"]),
        min_windows=int(config["min_windows"]),
        train_epochs=int(config["train_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        use_ema=bool(config["use_ema"]),
        ema_decay=float(config["ema_decay"]),
        training_target=str(config["training_target"]),
        return_transform=str(config["return_transform"]),
        transform_eps=float(config["transform_eps"]),
        feature_mode=str(config["feature_mode"]),
        feature_rolling_vol_window=int(config["feature_rolling_vol_window"]),
        legacy_return_clip=bool(config["legacy_return_clip"]),
        diffusion_steps=int(config["diffusion_steps"]),
        beta_schedule=str(config["beta_schedule"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
        model_hidden_dim=int(config["model_hidden_dim"]),
        model_num_res_blocks=int(config["model_num_res_blocks"]),
        model_dropout=float(config["model_dropout"]),
        time_embedding_dim=int(config["time_embedding_dim"]),
        sampler_type=str(config["sampler_type"]),
        sample_steps=int(config["sample_steps"]),
        ddim_eta=float(config["ddim_eta"]),
    )
    real_paths = test_instrument.get_real_windows(
        int(config["n_real_windows_compare"]),
        random_seed=int(config["random_seed"]),
    )
    test_csv_exists_after = os.path.exists(test_csv_path)
    real_train_paths = train_instrument.get_real_windows(
        int(config["n_real_windows_compare"]),
        random_seed=int(config["random_seed"]),
    )
    print(f"[run:{run_name}] Real train windows sampled: {real_train_paths.shape}")
    print(f"[run:{run_name}] Real test windows sampled: {real_paths.shape}")

    base = float(config["normalization_base"])
    real_train_paths_norm = normalize_paths(real_train_paths, base=base)
    real_paths_norm = normalize_paths(real_paths, base=base)
    synth_paths_norm = normalize_paths(synth_paths, base=base)

    train_split = evaluate_split_artifacts(
        real_paths_norm=real_train_paths_norm,
        synth_paths_norm=synth_paths_norm,
        split="train",
        acf_max_lag=int(config["acf_max_lag"]),
        hist_bins=int(config["hist_bins"]),
        rolling_vol_window=int(config["rolling_vol_window"]),
    )
    test_split = evaluate_split_artifacts(
        real_paths_norm=real_paths_norm,
        synth_paths_norm=synth_paths_norm,
        split="test",
        acf_max_lag=int(config["acf_max_lag"]),
        hist_bins=int(config["hist_bins"]),
        rolling_vol_window=int(config["rolling_vol_window"]),
    )

    print(f"[run:{run_name}] Saving plots for train and test...")
    for split_name, split_data in [("train", train_split), ("test", test_split)]:
        split_plots_dir = os.path.join(plots_dir, split_name)
        os.makedirs(split_plots_dir, exist_ok=True)
        plot_overlay_paths(
            real_paths_norm=split_data["real_paths_norm"],
            synth_paths_norm=split_data["synth_paths_norm"],
            output_path=os.path.join(split_plots_dir, "overlay_paths_normalized.jpg"),
            n_plot_paths=int(config["n_plot_paths"]),
        )
        plot_hist(
            split_data["real_terminal_returns"],
            split_data["synth_terminal_returns"],
            title=f"Terminal Returns: Real {split_name.capitalize()} vs Synthetic",
            xlabel="Terminal return",
            output_path=os.path.join(split_plots_dir, "terminal_returns_hist.jpg"),
            hist_bins=int(config["hist_bins"]),
        )
        plot_hist(
            split_data["real_log_returns"].reshape(-1),
            split_data["synth_log_returns"].reshape(-1),
            title=f"One-Step Log Returns: Real {split_name.capitalize()} vs Synthetic",
            xlabel="Log return",
            output_path=os.path.join(split_plots_dir, "one_step_log_returns_hist.jpg"),
            hist_bins=int(config["hist_bins"]),
        )
        plot_hist(
            split_data["real_rolling_vol"],
            split_data["synth_rolling_vol"],
            title=f"Rolling Volatility ({config['rolling_vol_window']}): Real {split_name.capitalize()} vs Synthetic",
            xlabel="Rolling volatility",
            output_path=os.path.join(split_plots_dir, "rolling_volatility_hist.jpg"),
            hist_bins=int(config["hist_bins"]),
        )
        plot_acf(
            real_acf=split_data["real_acf"],
            synth_acf=split_data["synth_acf"],
            output_path=os.path.join(split_plots_dir, "acf_lags.jpg"),
        )

    print(f"[run:{run_name}] Saving CSV plot-input datasets and scores...")
    save_split_data(train_split, data_dir=data_dir)
    save_split_data(test_split, data_dir=data_dir)

    train_split["summary"].to_csv(train_metrics_path, index=False)
    test_split["summary"].to_csv(test_metrics_path, index=False)
    all_summary = pd.concat([train_split["summary"], test_split["summary"]], ignore_index=True)
    all_summary.to_csv(summary_all_metrics_path, index=False)
    test_split["summary"].to_csv(summary_test_legacy_path, index=False)
    eval_tail_quantiles = parse_float_list(config["eval_tail_quantiles"])
    eval_exceedance_thresholds = parse_float_list(config["eval_exceedance_thresholds"])

    dep_returns = pd.concat(
        [
            dependence_metrics_df(
                train_split["real_paths_norm"],
                train_split["synth_paths_norm"],
                "train",
                int(config["acf_max_lag"]),
                squared=False,
            ),
            dependence_metrics_df(
                test_split["real_paths_norm"],
                test_split["synth_paths_norm"],
                "test",
                int(config["acf_max_lag"]),
                squared=False,
            ),
        ],
        ignore_index=True,
    )
    dep_squared = pd.concat(
        [
            dependence_metrics_df(
                train_split["real_paths_norm"],
                train_split["synth_paths_norm"],
                "train",
                int(config["acf_max_lag"]),
                squared=True,
            ),
            dependence_metrics_df(
                test_split["real_paths_norm"],
                test_split["synth_paths_norm"],
                "test",
                int(config["acf_max_lag"]),
                squared=True,
            ),
        ],
        ignore_index=True,
    )
    tail_df = pd.concat(
        [
            tail_metrics_df(
                train_split["real_paths_norm"],
                train_split["synth_paths_norm"],
                "train",
                eval_tail_quantiles,
                eval_exceedance_thresholds,
            ),
            tail_metrics_df(
                test_split["real_paths_norm"],
                test_split["synth_paths_norm"],
                "test",
                eval_tail_quantiles,
                eval_exceedance_thresholds,
            ),
        ],
        ignore_index=True,
    )
    path_risk_df = pd.concat(
        [
            path_risk_metrics_df(train_split["real_paths_norm"], train_split["synth_paths_norm"], "train"),
            path_risk_metrics_df(test_split["real_paths_norm"], test_split["synth_paths_norm"], "test"),
        ],
        ignore_index=True,
    )

    dep_returns.to_csv(dep_returns_path, index=False)
    dep_squared.to_csv(dep_squared_path, index=False)
    tail_df.to_csv(tail_metrics_path, index=False)
    path_risk_df.to_csv(path_risk_metrics_path, index=False)

    for split_name, split_data in [("train", train_split), ("test", test_split)]:
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        real_cum = cumulative_log_returns(split_data["real_paths_norm"])
        synth_cum = cumulative_log_returns(split_data["synth_paths_norm"])
        real_mdd = max_drawdown(split_data["real_paths_norm"])
        synth_mdd = max_drawdown(split_data["synth_paths_norm"])

        pd.concat(
            [
                distribution_vectors_df(real_cum, split_name, "real", "cumulative_log_return"),
                distribution_vectors_df(synth_cum, split_name, "synthetic", "cumulative_log_return"),
            ],
            ignore_index=True,
        ).to_csv(os.path.join(split_dir, "dist_cumulative_log_return.csv"), index=False)
        pd.concat(
            [
                distribution_vectors_df(real_mdd, split_name, "real", "max_drawdown"),
                distribution_vectors_df(synth_mdd, split_name, "synthetic", "max_drawdown"),
            ],
            ignore_index=True,
        ).to_csv(os.path.join(split_dir, "dist_max_drawdown.csv"), index=False)

    save_training_scores_available(
        output_path=training_scores_path,
        run_name=run_name,
        config=config,
        model_exists_before=model_exists_before,
        model_exists_after=model_exists_after,
        train_csv_exists_before=train_csv_exists_before,
        train_csv_exists_after=train_csv_exists_after,
        test_csv_exists_before=test_csv_exists_before,
        test_csv_exists_after=test_csv_exists_after,
        train_generate_seconds=train_generate_seconds,
        train_instrument=train_instrument,
    )

    print(f"[run:{run_name}] Completed.")
    print(f"[run:{run_name}] Saved metrics (all splits): {summary_all_metrics_path}")
    print(f"[run:{run_name}] Saved metrics (legacy test): {summary_test_legacy_path}")
    print(f"[run:{run_name}] Saved training metrics: {train_metrics_path}")
    print(f"[run:{run_name}] Saved testing metrics: {test_metrics_path}")
    print(f"[run:{run_name}] Saved training scores: {training_scores_path}")
    print(f"[run:{run_name}] Saved training manifest: {train_manifest_path}")
    print(f"[run:{run_name}] Saved training history: {training_history_path}")
    print(f"[run:{run_name}] Saved noise schedule: {noise_schedule_path}")
    print(f"[run:{run_name}] Saved dependence metrics: {dep_returns_path} and {dep_squared_path}")
    print(f"[run:{run_name}] Saved tail metrics: {tail_metrics_path}")
    print(f"[run:{run_name}] Saved path-risk metrics: {path_risk_metrics_path}")
    print(f"[run:{run_name}] Saved plot-input CSVs: {data_dir}")
    print(f"[run:{run_name}] Saved plots: {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diffusion simulator from JSON config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional JSON config filename in ./gan_training_configs (extension .json optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "gan_training_configs")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, config_path, config = load_config(configs_dir, config_name=args.config_name)
    print(f"[run:{run_name}] Loaded config: {config_path}")
    validate_config(config)
    run_simulation(run_name=run_name, config_path=config_path, config=config)


if __name__ == "__main__":
    main()
