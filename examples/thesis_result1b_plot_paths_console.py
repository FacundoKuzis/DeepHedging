"""
Simple console script to visualize GBM paths from a thesis_result1b compare config.

It saves:
1) Paths in price levels S
2) Paths in log-moneyness log(S/K)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.thesis_result1_common import THESIS_MODELS_ROOT, load_config_by_name  # noqa: E402
from DeepHedging.HedgingInstruments import GBMStock  # noqa: E402
from DeepHedging.utils.gbm_calibration import calibrate_gbm_from_market_data  # noqa: E402
from DeepHedging.utils.historical_windows import build_historical_windows_from_csv  # noqa: E402
from DeepHedging.utils.market_data import download_ohlcv_to_csv  # noqa: E402


RESULT1B_ROOT = os.path.join(THESIS_MODELS_ROOT, "thesis_result1b")


def _gbm_sigma_kwargs(config: dict, base_sigma: float) -> dict:
    mode = str(config.get("gbm_sigma_per_path_mode", "fixed")).strip().lower()
    kwargs = {
        "sigma": float(base_sigma),
        "sigma_per_path_mode": mode,
    }
    if mode == "uniform":
        kwargs["sigma_uniform_low"] = float(config["gbm_sigma_uniform_low"])
        kwargs["sigma_uniform_high"] = float(config["gbm_sigma_uniform_high"])
    elif mode == "discrete":
        kwargs["sigma_discrete_values"] = [float(v) for v in config["gbm_sigma_discrete_values"]]
        if config.get("gbm_sigma_discrete_probs") is not None:
            kwargs["sigma_discrete_probs"] = [float(p) for p in config["gbm_sigma_discrete_probs"]]
    return kwargs


def _run_plot_dir(run_name: str) -> str:
    out_dir = os.path.join(RESULT1B_ROOT, "compare", run_name, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _market_cache_dir() -> str:
    cache_dir = os.path.join(RESULT1B_ROOT, "market_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _build_paths_from_config(config: dict) -> tuple[np.ndarray, int]:
    n = int(config["n"])
    trading_days = int(config["trading_days_per_year"])
    eval_paths = int(config["eval_paths"])
    eval_seed = int(config["eval_seed"])
    context_length = int(config.get("context_length", 0)) if bool(config.get("use_price_history_context", False)) else 0

    test_mode = str(config.get("test_data_mode", "simulated")).strip().lower()
    if test_mode == "historical_windows":
        cache_dir = _market_cache_dir()
        test_csv = os.path.join(
            cache_dir,
            f"{config['ticker']}_{config['test_start_date']}_{config['test_end_date']}_{config['interval']}.csv",
        )
        if not os.path.isfile(test_csv):
            if not bool(config["download_if_missing"]):
                raise FileNotFoundError(f"Missing test CSV and download_if_missing=false: {test_csv}")
            download_ohlcv_to_csv(
                ticker=str(config["ticker"]),
                start_date=str(config["test_start_date"]),
                end_date=str(config["test_end_date"]),
                interval=str(config["interval"]),
                output_csv=test_csv,
            )
        windows_2d, _ = build_historical_windows_from_csv(
            csv_path=test_csv,
            n_hedging_steps=n,
            start_date=str(config["test_start_date"]),
            end_date=str(config["test_end_date"]),
            price_col=str(config["price_col"]),
            stride=int(config.get("historical_stride", 1)),
            max_windows=eval_paths,
        )
        return windows_2d.astype(np.float32), 0

    cache_dir = _market_cache_dir()
    calib = calibrate_gbm_from_market_data(
        market_cache_dir=cache_dir,
        ticker=str(config["ticker"]),
        train_start_date=str(config["train_start_date"]),
        train_end_date=str(config["train_end_date"]),
        interval=str(config["interval"]),
        price_col=str(config["price_col"]),
        trading_days_per_year=trading_days,
        sigma_source=str(config["sigma_source"]),
        implied_vol_source=str(config["implied_vol_source"]),
        implied_vol_stat=str(config["implied_vol_stat"]),
        fixed_implied_vol=None if config["fixed_implied_vol"] is None else float(config["fixed_implied_vol"]),
        risk_free_source=str(config["risk_free_source"]),
        fixed_risk_free=None if config["fixed_risk_free"] is None else float(config["fixed_risk_free"]),
        download_if_missing=bool(config["download_if_missing"]),
    )

    instrument = GBMStock(
        S0=float(config["s0"]),
        T=float(n / float(trading_days)),
        N=n,
        r=float(calib.r_train),
        **_gbm_sigma_kwargs(config=config, base_sigma=float(calib.sigma_train)),
    )
    if context_length > 0 and hasattr(instrument, "generate_paths_with_context"):
        full = instrument.generate_paths_with_context(
            num_paths=eval_paths,
            n_context_steps=context_length,
            random_seed=eval_seed,
        )
        full_np = tf.convert_to_tensor(full, dtype=tf.float32).numpy()
        return full_np, context_length

    paths = instrument.generate_paths(num_paths=eval_paths, random_seed=eval_seed)
    return tf.convert_to_tensor(paths, dtype=tf.float32).numpy(), 0


def _select_plot_paths(paths_2d: np.ndarray, n_plot_paths: int, seed: int) -> np.ndarray:
    n_total = int(paths_2d.shape[0])
    n_pick = min(int(n_plot_paths), n_total)
    if n_pick <= 0:
        raise ValueError("n_plot_paths must be > 0.")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_total, size=n_pick, replace=False)
    return paths_2d[idx]


def _plot_levels(paths_2d: np.ndarray, out_path: str, context_length: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    t = np.arange(paths_2d.shape[1])
    for row in paths_2d:
        ax.plot(t, row, alpha=0.35, linewidth=1.0)
    if context_length > 0:
        ax.axvline(context_length, color="black", linestyle="--", linewidth=1.5, label="Inicio hedge")
    ax.set_title("Paths simulados en niveles S")
    ax.set_xlabel("Paso temporal")
    ax.set_ylabel("Precio")
    if context_length > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_log_moneyness(paths_2d: np.ndarray, strike: float, out_path: str, context_length: int) -> None:
    k = float(strike)
    if k <= 0.0:
        raise ValueError("strike must be > 0 for log-moneyness plot.")
    safe = np.maximum(paths_2d, 1e-12)
    log_m = np.log(safe / k)
    fig, ax = plt.subplots(figsize=(11, 6))
    t = np.arange(log_m.shape[1])
    for row in log_m:
        ax.plot(t, row, alpha=0.35, linewidth=1.0)
    if context_length > 0:
        ax.axvline(context_length, color="black", linestyle="--", linewidth=1.5, label="Inicio hedge")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Paths simulados en log-moneyness: log(S/K)")
    ax.set_xlabel("Paso temporal")
    ax.set_ylabel("log(S/K)")
    if context_length > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_plot(run_name: str, config: dict, n_plot_paths: int) -> None:
    paths_2d, context_length = _build_paths_from_config(config)
    sample = _select_plot_paths(
        paths_2d=paths_2d,
        n_plot_paths=n_plot_paths,
        seed=int(config["eval_seed"]),
    )

    out_dir = _run_plot_dir(run_name)
    levels_path = os.path.join(out_dir, "gbm_paths_levels.jpg")
    logm_path = os.path.join(out_dir, "gbm_paths_log_moneyness.jpg")
    csv_path = os.path.join(out_dir, "gbm_paths_sample.csv")

    _plot_levels(sample, levels_path, context_length)
    _plot_log_moneyness(sample, float(config["strike"]), logm_path, context_length)

    cols = [f"t_{i}" for i in range(sample.shape[1])]
    pd.DataFrame(sample, columns=cols).to_csv(csv_path, index=False)

    print(f"[run:{run_name}] Saved levels plot: {levels_path}")
    print(f"[run:{run_name}] Saved log-moneyness plot: {logm_path}")
    print(f"[run:{run_name}] Saved sampled paths CSV: {csv_path}")
    if context_length > 0:
        print(f"[run:{run_name}] Context length detected: {context_length} timesteps.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GBM paths from thesis_result1b compare config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="JSON config filename in thesis_result1b_configs/compare (extension optional).",
    )
    parser.add_argument(
        "--n-plot-paths",
        type=int,
        default=80,
        help="Number of sampled paths to draw (default: 80).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1b_configs", "compare")
    run_name, _, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter COMPARE JSON config name from 'thesis_result1b_configs/compare': ",
    )
    print(f"[run:{run_name}] Loaded config and generating paths...")
    run_plot(run_name=run_name, config=cfg, n_plot_paths=int(args.n_plot_paths))


if __name__ == "__main__":
    main()
