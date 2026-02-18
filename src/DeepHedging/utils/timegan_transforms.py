"""Return transforms used by TimeGAN v2 training pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_SQRT2 = math.sqrt(2.0)


def validate_transform_name(name: str) -> str:
    key = str(name).strip().lower()
    allowed = {"minmax", "gaussian_cdf", "empirical_cdf"}
    if key not in allowed:
        raise ValueError(f"Unsupported return_transform '{name}'. Allowed: {sorted(allowed)}")
    return key


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-12:
        return 1e-12
    return std


def _norm_cdf(values: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + vec_erf(values / _SQRT2))


def _norm_ppf(values: np.ndarray) -> np.ndarray:
    """
    Inverse CDF approximation by Peter J. Acklam.
    Accurate enough for simulation transforms and does not require scipy.
    """
    p = np.asarray(values, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("Normal PPF input must be strictly inside (0, 1).")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1.0 - plow
    q = np.zeros_like(p)

    low = p < plow
    mid = (p >= plow) & (p <= phigh)
    high = p > phigh

    if np.any(low):
        x = np.sqrt(-2.0 * np.log(p[low]))
        q[low] = (
            (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5])
            / ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1.0)
        )

    if np.any(mid):
        x = p[mid] - 0.5
        r = x * x
        q[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * x
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    if np.any(high):
        x = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        q[high] = -(
            (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5])
            / ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1.0)
        )
    return q


def fit_transform_1d(values: np.ndarray, method: str, eps: float = 1e-6) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot fit transform on empty array.")
    if not np.isfinite(arr).all():
        raise ValueError("Transform input contains NaN or inf.")

    key = validate_transform_name(method)
    eps = float(eps)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"transform_eps must satisfy 0 < eps < 0.5. Got {eps}.")

    if key == "minmax":
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        span = mx - mn
        if not np.isfinite(span) or span < 1e-12:
            transformed = np.full_like(arr, 0.5, dtype=np.float64)
        else:
            transformed = (arr - mn) / span
            transformed = np.clip(transformed, 0.0, 1.0)
        state = {"method": key, "min": mn, "max": mx}
        return transformed.astype(np.float32), state

    if key == "gaussian_cdf":
        mean = float(np.mean(arr))
        std = _safe_std(arr)
        z = (arr - mean) / std
        transformed = np.clip(_norm_cdf(z), eps, 1.0 - eps)
        state = {"method": key, "mean": mean, "std": std, "eps": eps}
        return transformed.astype(np.float32), state

    # empirical_cdf
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    transformed = (ranks + 0.5) / float(n)
    transformed = np.clip(transformed, eps, 1.0 - eps)
    state = {"method": key, "sorted_values": np.sort(arr), "eps": eps}
    return transformed.astype(np.float32), state


def inverse_transform_1d(values: np.ndarray, state: dict[str, Any], eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    if state is None or "method" not in state:
        raise ValueError("Transform state is missing required 'method'.")

    method = validate_transform_name(str(state["method"]))
    eps = float(eps)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"transform_eps must satisfy 0 < eps < 0.5. Got {eps}.")

    if method == "minmax":
        clipped = np.clip(arr, 0.0, 1.0)
        mn = float(state["min"])
        mx = float(state["max"])
        span = mx - mn
        if not np.isfinite(span) or span < 1e-12:
            out = np.full_like(clipped, mn, dtype=np.float64)
        else:
            out = clipped * span + mn
        return out.astype(np.float32)

    clipped = np.clip(arr, eps, 1.0 - eps)
    if method == "gaussian_cdf":
        mean = float(state["mean"])
        std = float(state["std"])
        out = mean + std * _norm_ppf(clipped)
        return out.astype(np.float32)

    # empirical_cdf inverse
    sorted_values = np.asarray(state["sorted_values"], dtype=np.float64)
    if sorted_values.size == 0:
        raise ValueError("Empirical transform state has empty support.")
    try:
        out = np.quantile(sorted_values, clipped, method="linear")
    except TypeError:  # pragma: no cover - numpy<1.22 fallback
        out = np.quantile(sorted_values, clipped, interpolation="linear")
    return np.asarray(out, dtype=np.float32)
