"""
Standalone demo script for the TimeGAN-based price simulator.

It trains/loads TimeGANStock, samples synthetic paths, compares against real
rolling windows, and saves diagnostic plots + metrics.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work without installing the package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.HedgingInstruments import TimeGANStock


# ===========================
# Global Config (edit here)
# ===========================
S0 = 100.0
T = 22 / 252
N = 22
R = 0.05

TICKER = "SPY"
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
INTERVAL = "1d"
PRICE_COL = "Close"
TIMEGAN_DIR = "timegan_log_returns"
CSV_PATH = os.path.join(ROOT_DIR, "assets", "csvs", TIMEGAN_DIR, "spy_2019_2024_1d.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "assets", "models", TIMEGAN_DIR, "spy_timegan_returns_N63.pkl")

DOWNLOAD_IF_MISSING = True
RETRAIN = True
STRIDE = 1
MIN_WINDOWS = 200

TRAIN_EPOCHS = 800
BATCH_SIZE = 128
NOISE_DIM = 32
LAYERS_DIM = 128
LATENT_DIM = 24
LEARNING_RATE = 5e-4
GAMMA = 1.0
RANDOM_SEED = 42
TRAINING_TARGET = "log_returns"  # "log_returns" or "price_levels"
MATCH_RETURN_MOMENTS = False
INPUT_RETURN_CLIP_QUANTILES = None
OUTPUT_RETURN_CLIP_QUANTILES = None

N_SYNTH_PATHS = 300
N_REAL_WINDOWS_COMPARE = 300
N_PLOT_PATHS = 20
ROLLING_VOL_WINDOW = 20
ACF_MAX_LAG = 10
HIST_BINS = 50

PLOTS_DIR = os.path.join(ROOT_DIR, "assets", "plots", TIMEGAN_DIR)
METRICS_PATH = os.path.join(ROOT_DIR, "assets", "csvs", "timegan", "summary_metrics.csv")


def normalize_paths(paths: np.ndarray, base: float = 100.0) -> np.ndarray:
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


def histogram_l1_distance(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    lower = min(np.min(a), np.min(b))
    upper = max(np.max(a), np.max(b))
    hist_a, edges = np.histogram(a, bins=bins, range=(lower, upper), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.abs(hist_a - hist_b) * widths))


def plot_overlay_paths(real_paths_norm: np.ndarray, synth_paths_norm: np.ndarray, output_path: str):
    plt.figure(figsize=(10, 6))
    for i in range(min(N_PLOT_PATHS, real_paths_norm.shape[0])):
        plt.plot(real_paths_norm[i], color="tab:blue", alpha=0.35, linewidth=1)
    for i in range(min(N_PLOT_PATHS, synth_paths_norm.shape[0])):
        plt.plot(synth_paths_norm[i], color="tab:orange", alpha=0.35, linewidth=1, linestyle="--")
    plt.title("Normalized Paths Overlay (base=100)")
    plt.xlabel("Time step")
    plt.ylabel("Normalized price")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hist(real: np.ndarray, synth: np.ndarray, title: str, xlabel: str, output_path: str):
    plt.figure(figsize=(10, 6))
    plt.hist(real, bins=HIST_BINS, alpha=0.6, density=True, label="Real")
    plt.hist(synth, bins=HIST_BINS, alpha=0.6, density=True, label="Synthetic")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_acf(real_acf: np.ndarray, synth_acf: np.ndarray, output_path: str):
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


def build_summary_metrics(real_paths: np.ndarray, synth_paths: np.ndarray) -> pd.DataFrame:
    real_r = one_step_returns(real_paths).reshape(-1)
    synth_r = one_step_returns(synth_paths).reshape(-1)
    real_term = terminal_returns(real_paths)
    synth_term = terminal_returns(synth_paths)
    real_acf = mean_acf(real_paths, ACF_MAX_LAG)
    synth_acf = mean_acf(synth_paths, ACF_MAX_LAG)

    rows = [
        {
            "metric": "daily_return_mean",
            "real": float(np.mean(real_r)),
            "synthetic": float(np.mean(synth_r)),
        },
        {
            "metric": "daily_return_std",
            "real": float(np.std(real_r)),
            "synthetic": float(np.std(synth_r)),
        },
        {
            "metric": "daily_return_skew",
            "real": skewness(real_r),
            "synthetic": skewness(synth_r),
        },
        {
            "metric": "daily_return_kurtosis_excess",
            "real": kurtosis_excess(real_r),
            "synthetic": kurtosis_excess(synth_r),
        },
        {
            "metric": "terminal_return_mean",
            "real": float(np.mean(real_term)),
            "synthetic": float(np.mean(synth_term)),
        },
        {
            "metric": "terminal_return_std",
            "real": float(np.std(real_term)),
            "synthetic": float(np.std(synth_term)),
        },
        {
            "metric": "terminal_hist_l1",
            "real": 0.0,
            "synthetic": histogram_l1_distance(real_term, synth_term, bins=HIST_BINS),
        },
        {
            "metric": "one_step_log_return_hist_l1",
            "real": 0.0,
            "synthetic": histogram_l1_distance(
                one_step_log_returns(real_paths).reshape(-1),
                one_step_log_returns(synth_paths).reshape(-1),
                bins=HIST_BINS,
            ),
        },
    ]

    acf_abs_err = np.abs(real_acf - synth_acf)
    rows.append(
        {
            "metric": "acf_abs_error_mean_lag_1_10",
            "real": 0.0,
            "synthetic": float(np.nanmean(acf_abs_err)),
        }
    )

    for lag_idx, lag_err in enumerate(acf_abs_err, start=1):
        rows.append(
            {
                "metric": f"acf_abs_error_lag_{lag_idx}",
                "real": 0.0,
                "synthetic": float(lag_err),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    instrument = TimeGANStock(
        S0=S0,
        T=T,
        N=N,
        r=R,
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        price_col=PRICE_COL,
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        download_if_missing=DOWNLOAD_IF_MISSING,
        retrain=RETRAIN,
        stride=STRIDE,
        random_seed=RANDOM_SEED,
        min_windows=MIN_WINDOWS,
        train_epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        noise_dim=NOISE_DIM,
        layers_dim=LAYERS_DIM,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        training_target=TRAINING_TARGET,
        match_return_moments=MATCH_RETURN_MOMENTS,
        input_return_clip_quantiles=INPUT_RETURN_CLIP_QUANTILES,
        output_return_clip_quantiles=OUTPUT_RETURN_CLIP_QUANTILES,
    )

    synth_paths = instrument.generate_paths(N_SYNTH_PATHS, random_seed=RANDOM_SEED).numpy()
    real_paths = instrument.get_real_windows(N_REAL_WINDOWS_COMPARE, random_seed=RANDOM_SEED)

    real_paths_norm = normalize_paths(real_paths, base=100.0)
    synth_paths_norm = normalize_paths(synth_paths, base=100.0)

    plot_overlay_paths(
        real_paths_norm=real_paths_norm,
        synth_paths_norm=synth_paths_norm,
        output_path=os.path.join(PLOTS_DIR, "overlay_paths_normalized.pdf"),
    )

    real_terminal = terminal_returns(real_paths_norm)
    synth_terminal = terminal_returns(synth_paths_norm)
    plot_hist(
        real_terminal,
        synth_terminal,
        title="Terminal Returns: Real vs Synthetic",
        xlabel="Terminal return",
        output_path=os.path.join(PLOTS_DIR, "terminal_returns_hist.pdf"),
    )

    real_log_r = one_step_log_returns(real_paths_norm).reshape(-1)
    synth_log_r = one_step_log_returns(synth_paths_norm).reshape(-1)
    plot_hist(
        real_log_r,
        synth_log_r,
        title="One-Step Log Returns: Real vs Synthetic",
        xlabel="Log return",
        output_path=os.path.join(PLOTS_DIR, "one_step_log_returns_hist.pdf"),
    )

    real_roll_vol = rolling_vol_distribution(real_paths_norm, window=ROLLING_VOL_WINDOW)
    synth_roll_vol = rolling_vol_distribution(synth_paths_norm, window=ROLLING_VOL_WINDOW)
    plot_hist(
        real_roll_vol,
        synth_roll_vol,
        title=f"Rolling Volatility ({ROLLING_VOL_WINDOW} steps): Real vs Synthetic",
        xlabel="Rolling volatility",
        output_path=os.path.join(PLOTS_DIR, "rolling_volatility_hist.pdf"),
    )

    real_acf = mean_acf(real_paths_norm, ACF_MAX_LAG)
    synth_acf = mean_acf(synth_paths_norm, ACF_MAX_LAG)
    plot_acf(
        real_acf=real_acf,
        synth_acf=synth_acf,
        output_path=os.path.join(PLOTS_DIR, "acf_lags_1_10.pdf"),
    )

    summary = build_summary_metrics(real_paths_norm, synth_paths_norm)
    summary.to_csv(METRICS_PATH, index=False)

    print("[ok] TimeGAN simulator demo completed.")
    print(f"[save] plots: {PLOTS_DIR}")
    print(f"[save] metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
