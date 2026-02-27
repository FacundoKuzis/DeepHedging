"""
Generate 4 Student-t path histogram variants from a compare config baseline.

Outputs:
- 4 histograms of terminal price S_T
- 4 histograms of terminal log-moneyness ln(S_T / K)
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.compare_console import _apply_relaxed_defaults_compare  # noqa: E402
from examples.thesis_result1_common import THESIS_MODELS_ROOT, get_config_relative_stem  # noqa: E402
from examples.thesis_result1b_plot_paths_console import _build_paths_from_config  # noqa: E402
from examples.unified_console_common import detect_pipeline, load_merged_config, resolve_config_path, strip_meta_keys  # noqa: E402


def _out_dir(config_path: str) -> str:
    rel_stem = get_config_relative_stem(config_path)
    out_dir = os.path.join(THESIS_MODELS_ROOT, rel_stem, "plots", "student_t_histograms")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: str) -> None:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=80, alpha=0.85, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A11 Student-t histogram variants.")
    parser.add_argument(
        "config_name",
        nargs="?",
        default="A11/euro_student_t/stress/wavenet_ctx50_conv/compare/vs_bs_est_sigma_ctx50",
        help="Compare config path/name under ./configs.",
    )
    parser.add_argument("--eval-paths", type=int, default=20000, help="Paths per variant.")
    args = parser.parse_args()

    config_path = resolve_config_path(
        config_name=args.config_name,
        task="compare",
        prompt_label="Enter COMPARE config name/path from 'configs': ",
    )
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)
    if pipeline != "result1b":
        raise ValueError("This script supports only pipeline='result1b'.")
    cfg = _apply_relaxed_defaults_compare(strip_meta_keys(merged), pipeline=pipeline)

    base = dict(cfg)
    base["instrument_model"] = "student_t"
    base["test_data_mode"] = "simulated"

    variants = [
        {
            "name": "v1_df4_sigma02",
            "title": "Variante 1: df=4, sigma=0.2",
            "overrides": {
                "gbm_sigma_per_path_mode": "fixed",
                "sigma": 0.2,
                "student_t_df_per_path_mode": "fixed",
                "student_t_df": 4.0,
            },
        },
        {
            "name": "v2_df8_sigma02",
            "title": "Variante 2: df=8, sigma=0.2",
            "overrides": {
                "gbm_sigma_per_path_mode": "fixed",
                "sigma": 0.2,
                "student_t_df_per_path_mode": "fixed",
                "student_t_df": 8.0,
            },
        },
        {
            "name": "v3_df30_sigma02",
            "title": "Variante 3: df=30, sigma=0.2",
            "overrides": {
                "gbm_sigma_per_path_mode": "fixed",
                "sigma": 0.2,
                "student_t_df_per_path_mode": "fixed",
                "student_t_df": 30.0,
            },
        },
        {
            "name": "v4_dfU3_2_30_sigmaU0_01_0_8",
            "title": "Variante 4: df~U(3.2,30), sigma~U(0.01,0.8)",
            "overrides": {
                "gbm_sigma_per_path_mode": "uniform",
                "gbm_sigma_uniform_low": 0.01,
                "gbm_sigma_uniform_high": 0.8,
                "student_t_df_per_path_mode": "uniform",
                "student_t_df_uniform_low": 3.2,
                "student_t_df_uniform_high": 30.0,
            },
        },
    ]

    out_dir = _out_dir(source_path)
    strike = float(base["strike"])

    for idx, var in enumerate(variants, start=1):
        cfg_i = dict(base)
        cfg_i.update(var["overrides"])
        paths_2d, context_length, _ = _build_paths_from_config(cfg_i, eval_paths_override=int(args.eval_paths))
        terminal_s = np.asarray(paths_2d[:, -1], dtype=np.float64)
        terminal_ln_m = np.log(np.maximum(terminal_s, 1e-12) / max(strike, 1e-12))

        s_path = os.path.join(out_dir, f"{idx:02d}_{var['name']}_hist_terminal_price.jpg")
        ln_path = os.path.join(out_dir, f"{idx:02d}_{var['name']}_hist_terminal_ln_moneyness.jpg")

        _plot_hist(
            terminal_s,
            title=f"{var['title']} - Histograma de S_T",
            xlabel="S_T",
            out_path=s_path,
        )
        _plot_hist(
            terminal_ln_m,
            title=f"{var['title']} - Histograma de ln(S_T/K)",
            xlabel="ln(S_T/K)",
            out_path=ln_path,
        )
        print(
            f"[{idx}/4] Saved:\n"
            f"  - {s_path}\n"
            f"  - {ln_path}\n"
            f"  context_length_detected={context_length}"
        )


if __name__ == "__main__":
    main()
