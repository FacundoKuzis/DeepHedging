"""Finance-oriented diagnostics for TimeGAN v2 experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_step_log_returns(paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float64)
    return np.log(np.maximum(arr[:, 1:], 1e-8) / np.maximum(arr[:, :-1], 1e-8))


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


def mean_acf(paths: np.ndarray, max_lag: int, squared: bool = False) -> np.ndarray:
    lr = one_step_log_returns(paths)
    if squared:
        lr = lr * lr
    acf_values = np.array([acf_for_lags(row, max_lag=max_lag) for row in lr], dtype=np.float64)
    return np.nanmean(acf_values, axis=0)


def dependence_metrics_df(
    real_paths: np.ndarray,
    synth_paths: np.ndarray,
    split: str,
    max_lag: int,
    squared: bool,
) -> pd.DataFrame:
    real_acf = mean_acf(real_paths, max_lag=max_lag, squared=squared)
    synth_acf = mean_acf(synth_paths, max_lag=max_lag, squared=squared)
    return pd.DataFrame(
        {
            "split": split,
            "series_type": "squared_log_returns" if squared else "log_returns",
            "lag": np.arange(1, max_lag + 1, dtype=np.int64),
            "real_acf": real_acf,
            "synthetic_acf": synth_acf,
            "abs_error": np.abs(real_acf - synth_acf),
        }
    )


def tail_metrics_df(
    real_paths: np.ndarray,
    synth_paths: np.ndarray,
    split: str,
    quantiles: list[float],
    exceedance_thresholds: list[float],
) -> pd.DataFrame:
    real_lr = one_step_log_returns(real_paths).reshape(-1)
    synth_lr = one_step_log_returns(synth_paths).reshape(-1)

    rows: list[dict[str, float | str]] = []
    for q in quantiles:
        qf = float(q)
        real_q = float(np.quantile(real_lr, qf))
        synth_q = float(np.quantile(synth_lr, qf))
        rows.append(
            {
                "split": split,
                "metric": "quantile_value",
                "tail_side": "left" if qf <= 0.5 else "right",
                "level": qf,
                "real": real_q,
                "synthetic": synth_q,
                "abs_error": abs(synth_q - real_q),
            }
        )

    for thr in exceedance_thresholds:
        t = abs(float(thr))
        left_real = float(np.mean(real_lr <= -t))
        left_syn = float(np.mean(synth_lr <= -t))
        right_real = float(np.mean(real_lr >= t))
        right_syn = float(np.mean(synth_lr >= t))
        rows.append(
            {
                "split": split,
                "metric": "exceedance_prob",
                "tail_side": "left",
                "level": t,
                "real": left_real,
                "synthetic": left_syn,
                "abs_error": abs(left_syn - left_real),
            }
        )
        rows.append(
            {
                "split": split,
                "metric": "exceedance_prob",
                "tail_side": "right",
                "level": t,
                "real": right_real,
                "synthetic": right_syn,
                "abs_error": abs(right_syn - right_real),
            }
        )
    return pd.DataFrame(rows)


def cumulative_log_returns(paths: np.ndarray) -> np.ndarray:
    lr = one_step_log_returns(paths)
    return np.sum(lr, axis=1)


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float64)
    running_max = np.maximum.accumulate(arr, axis=1)
    drawdowns = 1.0 - arr / np.maximum(running_max, 1e-8)
    return np.max(drawdowns, axis=1)


def path_risk_metrics_df(real_paths: np.ndarray, synth_paths: np.ndarray, split: str) -> pd.DataFrame:
    real_cum = cumulative_log_returns(real_paths)
    synth_cum = cumulative_log_returns(synth_paths)
    real_mdd = max_drawdown(real_paths)
    synth_mdd = max_drawdown(synth_paths)

    rows = [
        {
            "split": split,
            "metric": "cum_log_return_mean",
            "real": float(np.mean(real_cum)),
            "synthetic": float(np.mean(synth_cum)),
        },
        {
            "split": split,
            "metric": "cum_log_return_std",
            "real": float(np.std(real_cum)),
            "synthetic": float(np.std(synth_cum)),
        },
        {
            "split": split,
            "metric": "max_drawdown_mean",
            "real": float(np.mean(real_mdd)),
            "synthetic": float(np.mean(synth_mdd)),
        },
        {
            "split": split,
            "metric": "max_drawdown_std",
            "real": float(np.std(real_mdd)),
            "synthetic": float(np.std(synth_mdd)),
        },
        {
            "split": split,
            "metric": "max_drawdown_p95",
            "real": float(np.quantile(real_mdd, 0.95)),
            "synthetic": float(np.quantile(synth_mdd, 0.95)),
        },
    ]
    df = pd.DataFrame(rows)
    df["abs_error"] = np.abs(df["synthetic"] - df["real"])
    return df


def distribution_vectors_df(values: np.ndarray, split: str, source: str, metric: str) -> pd.DataFrame:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return pd.DataFrame(
        {
            "split": split,
            "source": source,
            "metric": metric,
            "obs_id": np.arange(arr.size, dtype=np.int64),
            "value": arr,
        }
    )
