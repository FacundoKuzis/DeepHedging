"""Feature construction and windowing utilities for TimeGAN v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_feature_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    allowed = {"returns_only", "returns_plus_abs_return", "returns_plus_rolling_vol"}
    if key not in allowed:
        raise ValueError(f"Unsupported feature_mode '{mode}'. Allowed: {sorted(allowed)}")
    return key


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 0:
        raise ValueError("feature_rolling_vol_window must be > 0.")
    series = pd.Series(arr)
    out = series.rolling(window=window, min_periods=1).std(ddof=0).to_numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def build_return_feature_matrix(
    returns: np.ndarray,
    feature_mode: str = "returns_only",
    feature_rolling_vol_window: int = 5,
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot build feature matrix from empty returns array.")
    if not np.isfinite(arr).all():
        raise ValueError("Returns contain NaN or inf.")

    mode = validate_feature_mode(feature_mode)
    features = [arr.astype(np.float32)]
    names = ["log_return"]

    if mode == "returns_plus_abs_return":
        features.append(np.abs(arr).astype(np.float32))
        names.append("abs_log_return")
    elif mode == "returns_plus_rolling_vol":
        features.append(rolling_std(arr, window=int(feature_rolling_vol_window)))
        names.append(f"rolling_vol_{int(feature_rolling_vol_window)}")

    mat = np.stack(features, axis=1).astype(np.float32)
    return mat, names


def make_sliding_windows(
    feature_matrix: np.ndarray,
    seq_len: int,
    stride: int,
    min_windows: int,
) -> np.ndarray:
    mat = np.asarray(feature_matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D. Got shape {mat.shape}.")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if min_windows <= 0:
        raise ValueError("min_windows must be > 0.")
    if mat.shape[0] < seq_len:
        raise ValueError(
            f"Not enough observations to build windows of length {seq_len}. "
            f"Got {mat.shape[0]} rows."
        )

    windows = []
    for start in range(0, mat.shape[0] - seq_len + 1, stride):
        window = mat[start : start + seq_len]
        if np.isfinite(window).all():
            windows.append(window)

    if len(windows) < min_windows:
        raise ValueError(
            f"Not enough clean windows for TimeGAN training. "
            f"Found {len(windows)} windows, min_windows={min_windows}."
        )

    return np.asarray(windows, dtype=np.float32)


def feature_stats_table(
    raw_windows: np.ndarray,
    transformed_windows: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    raw = np.asarray(raw_windows, dtype=np.float64)
    trn = np.asarray(transformed_windows, dtype=np.float64)
    if raw.shape != trn.shape:
        raise ValueError(
            f"raw_windows and transformed_windows must share shape. Got {raw.shape} vs {trn.shape}."
        )
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D window tensor. Got shape {raw.shape}.")
    if len(feature_names) != raw.shape[2]:
        raise ValueError(
            f"feature_names length mismatch. Expected {raw.shape[2]}, got {len(feature_names)}."
        )

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        raw_col = raw[:, :, idx].reshape(-1)
        trn_col = trn[:, :, idx].reshape(-1)
        rows.append(
            {
                "feature_idx": int(idx),
                "feature_name": str(name),
                "pre_min": float(np.min(raw_col)),
                "pre_max": float(np.max(raw_col)),
                "pre_mean": float(np.mean(raw_col)),
                "pre_std": float(np.std(raw_col)),
                "post_min": float(np.min(trn_col)),
                "post_max": float(np.max(trn_col)),
                "post_mean": float(np.mean(trn_col)),
                "post_std": float(np.std(trn_col)),
            }
        )
    return pd.DataFrame(rows)
