"""
Console script to visualize path samples from unified compare configs.

It saves:
1) Paths in price levels S
2) Paths in log-moneyness ln(S/K)
All labels are in Spanish and plots are generated without titles.
"""

from __future__ import annotations

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

from examples.compare_console import _apply_relaxed_defaults_compare  # noqa: E402
from examples.thesis_result1_common import (  # noqa: E402
    THESIS_MODELS_ROOT,
    build_instrument_from_config,
    get_config_relative_stem,
    set_global_determinism,
)
from examples.unified_console_common import (  # noqa: E402
    detect_pipeline,
    load_merged_config,
    resolve_config_path,
    strip_meta_keys,
)
from DeepHedging.utils.gbm_calibration import calibrate_gbm_from_market_data  # noqa: E402
from DeepHedging.utils.historical_windows import build_historical_windows_from_csv  # noqa: E402
from DeepHedging.utils.market_data import download_ohlcv_to_csv  # noqa: E402


def _run_plot_dir(config_path: str) -> str:
    rel_stem = get_config_relative_stem(config_path)
    run_dir = os.path.normpath(os.path.join(THESIS_MODELS_ROOT, rel_stem))
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _market_cache_dir() -> str:
    cache_dir = os.path.join(THESIS_MODELS_ROOT, "market_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _as_paths_2d(paths: tf.Tensor | np.ndarray) -> np.ndarray:
    arr = tf.convert_to_tensor(paths, dtype=tf.float32).numpy()
    if arr.ndim == 3:
        if int(arr.shape[2]) < 1:
            raise ValueError("Generated paths have zero instruments.")
        if int(arr.shape[2]) > 1:
            print(
                "[plot][warning] Detected multiple instruments; plotting only instrument index 0."
            )
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected paths rank-2 after projection. Got shape={arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _build_paths_from_config(
    config: dict, eval_paths_override: int | None = None
) -> tuple[np.ndarray, int, np.ndarray | None]:
    n = int(config["n"])
    trading_days = int(config["trading_days_per_year"])
    eval_paths = int(eval_paths_override) if eval_paths_override is not None else int(config["eval_paths"])
    eval_seed = int(config["eval_seed"])
    set_global_determinism(eval_seed)

    use_context = bool(config.get("use_price_history_context", False))
    context_length = int(config.get("context_length", 0)) if use_context else 0
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
        # Keep parity with compare runner: normalize each window to configured S0.
        raw_s0 = windows_2d[:, 0].astype(np.float64)
        if np.any(~np.isfinite(raw_s0)) or np.any(raw_s0 <= 0.0):
            raise ValueError("Invalid historical window start prices for normalization.")
        target_s0 = float(config["s0"])
        windows_2d = (target_s0 * (windows_2d / raw_s0[:, None])).astype(np.float32)
        if context_length > 0:
            print(
                "[plot][info] test_data_mode='historical_windows': context prefix is not included in paths."
            )
        return windows_2d, 0, None

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

    cfg_for_builders = dict(config)
    cfg_for_builders["r"] = float(calib.r_train)
    cfg_for_builders["sigma"] = float(calib.sigma_train)
    cfg_for_builders.setdefault(
        "instrument_model",
        str(config.get("instrument_model", "gbm")).strip().lower(),
    )
    instrument = build_instrument_from_config(cfg_for_builders)

    if context_length > 0 and hasattr(instrument, "generate_paths_with_context"):
        full = instrument.generate_paths_with_context(
            num_paths=eval_paths,
            n_context_steps=context_length,
            random_seed=eval_seed,
        )
        sigma_vec = None
        if hasattr(instrument, "get_last_sampled_sigmas"):
            sigma_vec = instrument.get_last_sampled_sigmas()
            if sigma_vec is not None:
                sigma_vec = np.asarray(sigma_vec, dtype=np.float32).reshape(-1)
        return _as_paths_2d(full), context_length, sigma_vec

    paths = instrument.generate_paths(num_paths=eval_paths, random_seed=eval_seed)
    sigma_vec = None
    if hasattr(instrument, "get_last_sampled_sigmas"):
        sigma_vec = instrument.get_last_sampled_sigmas()
        if sigma_vec is not None:
            sigma_vec = np.asarray(sigma_vec, dtype=np.float32).reshape(-1)
    return _as_paths_2d(paths), 0, sigma_vec


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
        ax.axvline(context_length, color="black", linestyle="--", linewidth=1.5, label="Inicio de cobertura")
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
        ax.axvline(context_length, color="black", linestyle="--", linewidth=1.5, label="Inicio de cobertura")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Paso temporal")
    ax.set_ylabel("ln(S/K)")
    if context_length > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sigma_histogram(sigmas: np.ndarray, out_path: str) -> None:
    vals = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite sigma values available for histogram.")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(vals, bins=60, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Volatilidad (sigma)")
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_plot(
    run_name: str,
    config: dict,
    config_path: str,
    n_plot_paths: int,
    plot_all_paths: bool = False,
    eval_paths_override: int | None = None,
) -> None:
    paths_2d, context_length, sigma_vec = _build_paths_from_config(
        config, eval_paths_override=eval_paths_override
    )
    if bool(plot_all_paths):
        sample = np.asarray(paths_2d, dtype=np.float32)
    else:
        sample = _select_plot_paths(
            paths_2d=paths_2d,
            n_plot_paths=n_plot_paths,
            seed=int(config["eval_seed"]),
        )

    out_dir = _run_plot_dir(config_path=config_path)
    levels_path = os.path.join(out_dir, "sample_paths_levels.jpg")
    logm_path = os.path.join(out_dir, "sample_paths_log_moneyness.jpg")
    csv_path = os.path.join(out_dir, "sample_paths.csv")
    sigma_hist_path = os.path.join(out_dir, "sampled_sigmas_hist.jpg")
    sigma_csv_path = os.path.join(out_dir, "sampled_sigmas.csv")

    _plot_levels(sample, levels_path, context_length)
    _plot_log_moneyness(sample, float(config["strike"]), logm_path, context_length)

    cols = [f"t_{i}" for i in range(sample.shape[1])]
    pd.DataFrame(sample, columns=cols).to_csv(csv_path, index=False)
    if sigma_vec is not None:
        _plot_sigma_histogram(sigma_vec, sigma_hist_path)
        pd.DataFrame({"sigma": np.asarray(sigma_vec, dtype=np.float32)}).to_csv(sigma_csv_path, index=False)

    print(f"[run:{run_name}] Saved levels plot: {levels_path}")
    print(f"[run:{run_name}] Saved log-moneyness plot: {logm_path}")
    print(f"[run:{run_name}] Saved sampled paths CSV: {csv_path}")
    if sigma_vec is not None:
        print(f"[run:{run_name}] Saved sigma histogram: {sigma_hist_path}")
        print(f"[run:{run_name}] Saved sampled sigmas CSV: {sigma_csv_path}")
    if eval_paths_override is not None:
        print(f"[run:{run_name}] eval_paths override used: {int(eval_paths_override)}")
    if context_length > 0:
        print(f"[run:{run_name}] Context length detected: {context_length} timesteps.")
    if bool(plot_all_paths):
        print(f"[run:{run_name}] Se graficaron todos los paths: n={int(sample.shape[0])}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot simulated/historical paths from a unified compare config."
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config path/name under ./configs (extension optional).",
    )
    parser.add_argument(
        "--n-plot-paths",
        type=int,
        default=80,
        help="Number of sampled paths to draw (default: 80).",
    )
    parser.add_argument(
        "--plot-all-paths",
        action="store_true",
        help="Si se activa, grafica todos los paths (ignora --n-plot-paths).",
    )
    parser.add_argument(
        "--eval-paths",
        type=int,
        default=None,
        help="Optional override for eval_paths used only for plotting.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(
        config_name=args.config_name,
        task="compare",
        prompt_label="Enter COMPARE config name/path from 'configs': ",
    )
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)
    if pipeline != "result1b":
        raise ValueError(
            "This plot script currently supports pipeline='result1b' compare configs only."
        )

    run_name = str(merged.get("run_name") or os.path.splitext(os.path.basename(source_path))[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")

    cfg = strip_meta_keys(merged)
    cfg = _apply_relaxed_defaults_compare(cfg, pipeline=pipeline)

    print(f"[run:{run_name}] Loaded config: {source_path}")
    print(f"[run:{run_name}] Pipeline: {pipeline}")
    run_plot(
        run_name=run_name,
        config=cfg,
        config_path=source_path,
        n_plot_paths=int(args.n_plot_paths),
        plot_all_paths=bool(args.plot_all_paths),
        eval_paths_override=args.eval_paths,
    )


if __name__ == "__main__":
    main()
