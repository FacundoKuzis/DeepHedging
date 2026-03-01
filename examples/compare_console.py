"""
Unified compare runner for thesis pipelines.

Supported pipelines:
- result1
- result1b
- result1b_option_market
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.unified_console_common import (  # noqa: E402
    detect_pipeline,
    load_merged_config,
    resolve_config_path,
    strip_meta_keys,
)


def _apply_relaxed_defaults_compare(config: dict[str, Any], pipeline: str) -> dict[str, Any]:
    cfg = dict(config)
    if pipeline == "result1b":
        defaults = {
            "output_root": "assets/thesis_result1b",
            "models_dir": "models",
            "optimizers_dir": "optimizers",
            "price_computation_mode": "pathwise_if_available",
            "sigma_mode": "train_average",
            "historical_sigma_window_days": 252,
            "fixed_implied_vol": None,
            "fixed_risk_free": None,
            "historical_stride": 1,
            "max_test_windows": 15000,
            "bootstrap_method": "iid",
            "moving_block_size": 21,
            "reuse_actions_between_steps": True,
            "eval_agent_batch_size": 2000,
            "terminal_progress_log_every_agent_batches": 2,
            "use_price_history_context": False,
            "context_length": 0,
            "context_feature_mode": "log_returns",
            "context_pre_ttm_mode": "calculated",
            "context_for_path_generation_only": False,
            "history_conv1d_enabled": False,
            "history_conv1d_layers": None,
            "history_conv1d_pooling": "global_max",
            "instrument_model": "gbm",
            "gbm_sigma_per_path_mode": "fixed",
            "garch_alpha": 0.05,
            "garch_beta": 0.9,
            "garch_omega": None,
            "garch_leverage": 0.0,
            "garch_use_student_t": False,
            "garch_student_t_df": 8.0,
            "student_t_df": 8.0,
            "student_t_df_per_path_mode": "fixed",
            "student_t_df_uniform_low": None,
            "student_t_df_uniform_high": None,
            "student_t_df_discrete_values": None,
            "student_t_df_discrete_probs": None,
            "benchmark_lrm_provider": "bs_closed_form",
            "benchmark_lrm_outer_paths": 512,
            "benchmark_lrm_var_epsilon": 1e-10,
            "benchmark_lrm_use_antithetic": True,
            "benchmark_lrm_seed_mode": "shared_crn",
            "benchmark_lrm_mc_inner_paths": 1024,
            "benchmark_lrm_mc_inner_chunk_size": 64,
            "benchmark_lrm_mc_parallel_enabled": False,
            "benchmark_lrm_mc_n_workers": 1,
            "benchmark_lrm_mc_parallel_backend": "thread",
            "benchmark_lrm_mc_parallel_chunk_size": None,
            "benchmark_lrm_lsm_train_paths": 50000,
            "benchmark_lrm_lsm_ridge_alpha": 1e-6,
            "benchmark_lrm_lsm_feature_set": "default",
            "benchmark_lrm_lsm_poly_degree": 2,
            "benchmark_lrm_lsm_use_cache": True,
            "benchmark_lrm_lsm_cache_dir": None,
            "benchmark_lrm_lsm_cache_key": None,
            "benchmark_lrm_lsm_force_rebuild": False,
            "benchmark_lrm_verbose": False,
            "benchmark_lrm_log_every_t": 5,
            "benchmark_lrm_mc_log_every_chunks": 0,
            "benchmark_delta_sigma_mode": "none",
            "benchmark_delta_sigma_context_days": 50,
            "benchmark_delta_sigma_min_obs": 10,
            "benchmark_delta_sigma_garch_alpha": None,
            "benchmark_delta_sigma_garch_beta": None,
            "benchmark_delta_sigma_garch_leverage": None,
            "benchmark_delta_sigma_garch_omega": None,
            "benchmark_delta_sigma_floor": 1e-6,
            "benchmark_delta_sigma_cap": None,
            "benchmark_delta_sigma_default": None,
            "benchmark_delta_student_t_fit_enabled": False,
            "benchmark_delta_student_t_mode": "static",
            "benchmark_delta_student_t_min_obs": 10,
            "benchmark_delta_student_t_df_default": 8.0,
            "benchmark_delta_student_t_df_floor": 2.1,
            "benchmark_delta_student_t_df_cap": 200.0,
            "benchmark_hmm_num_states": 3,
            "benchmark_hmm_transition_smoothing": 1.0,
            "benchmark_hmm_state_multiplier_floor": 0.35,
            "benchmark_hmm_state_multiplier_cap": 3.5,
            "benchmark_hmm_tail_adjustment_enabled": True,
            "benchmark_hmm_tail_multiplier_cap": 1.6,
            "benchmark_agents_to_compare": [],
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)
    return cfg


def _dispatch_compare(
    pipeline: str,
    run_name: str,
    config_path_for_snapshot: str,
    config: dict[str, Any],
) -> None:
    if pipeline == "result1":
        from examples.thesis_result1_compare_console import run_comparison, validate_config
    elif pipeline == "result1b":
        from examples.thesis_result1b_compare_console import run_comparison, validate_config
    elif pipeline == "result1b_option_market":
        from examples.thesis_result1b_option_market_compare_console import run_comparison, validate_config
    else:
        raise ValueError(f"Unsupported pipeline for compare: {pipeline}")

    validate_config(config)
    run_comparison(run_name=run_name, config_path=config_path_for_snapshot, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified compare console.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config path/name under ./configs (extension optional).",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(
        config_name=args.config_name,
        task="compare",
        prompt_label="Enter COMPARE config name/path from 'configs': ",
    )
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)

    run_name = str(merged.get("run_name") or os.path.splitext(os.path.basename(source_path))[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")

    config_clean = strip_meta_keys(merged)
    config_clean = _apply_relaxed_defaults_compare(config_clean, pipeline=pipeline)

    print(f"[run:{run_name}] Loaded config: {source_path}")
    print(f"[run:{run_name}] Pipeline: {pipeline}")
    _dispatch_compare(
        pipeline=pipeline,
        run_name=run_name,
        config_path_for_snapshot=source_path,
        config=config_clean,
    )


if __name__ == "__main__":
    main()
