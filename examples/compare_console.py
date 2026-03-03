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


THESIS_MODELS_ROOT = os.path.normpath(r"G:\Mi unidad\Tesis2026\Models\Organized")


def _normalize_path_separators(path: str) -> str:
    return str(path).replace("\\", "/")


def _get_config_relative_stem(config_path: str) -> str:
    abs_path = os.path.abspath(config_path)
    try:
        rel = os.path.relpath(abs_path, ROOT_DIR)
    except ValueError:
        rel = os.path.basename(abs_path)
    rel = _normalize_path_separators(rel)
    rel_no_ext, _ = os.path.splitext(rel)

    prefixes = [
        "configs/runs/",
        "configs/",
        "thesis_result1b_configs/",
        "thesis_result1_configs/",
    ]
    for p in prefixes:
        if rel_no_ext.startswith(p):
            rel_no_ext = rel_no_ext[len(p):]
            break
    rel_no_ext = rel_no_ext.strip("/").strip()
    if not rel_no_ext:
        rel_no_ext = os.path.splitext(os.path.basename(abs_path))[0]
    if rel_no_ext.startswith("train/") or rel_no_ext.startswith("compare/"):
        rel_no_ext = f"thesis_result1b/{rel_no_ext}"
    elif rel_no_ext.startswith("option_market_compare/"):
        rel_no_ext = f"thesis_result1b/{rel_no_ext}"
    return rel_no_ext


def _apply_relaxed_defaults_compare(config: dict[str, Any], pipeline: str) -> dict[str, Any]:
    cfg = dict(config)
    if pipeline == "result1b":
        defaults = {
            "output_root": "assets/thesis_result1b",
            "models_dir": "models",
            "optimizers_dir": "optimizers",
            "compare_mode": "benchmark_vs_targets",
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
            "benchmark_delta_r_mode": "none",
            "benchmark_delta_r_context_days": 50,
            "benchmark_delta_r_min_obs": 10,
            "benchmark_delta_r_default": None,
            "benchmark_delta_r_floor": None,
            "benchmark_delta_r_cap": None,
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
            "benchmark_actions_reuse_from_run": None,
            "benchmark_actions_reuse_agent_name": None,
            "benchmark_actions_reuse_apply_no_intervention": False,
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


def _dispatch_result1b_paths_plot(
    run_name: str,
    config_path_for_snapshot: str,
    config: dict[str, Any],
) -> None:
    from examples.thesis_result1b_plot_paths_console import run_plot

    print(f"[run:{run_name}] Generating sample path plots.")
    run_plot(
        run_name=run_name,
        config=config,
        config_path=config_path_for_snapshot,
        n_plot_paths=80,
        plot_all_paths=True,
        eval_paths_override=None,
    )


def _result1b_run_dir_from_config_path(config_path: str) -> str:
    rel_stem = _get_config_relative_stem(config_path)
    return os.path.normpath(os.path.join(THESIS_MODELS_ROOT, rel_stem))


def _result1b_compare_missing_core_outputs(run_dir: str) -> list[str]:
    required_rel = [
        "run_metadata.json",
        "raw/raw_payload_manifest.json",
        "raw/terminal_errors_by_agent.npy",
        "tables/pairwise_terminal_stats.csv",
        "tables/point_metrics.csv",
        "tables/empirical_risk_metrics.csv",
        "tables/terminal_errors_long.csv",
        "tables/terminal_errors_wide.csv",
        "tables/calibration_manifest.csv",
    ]
    missing: list[str] = []
    for rel in required_rel:
        full = os.path.join(run_dir, rel.replace("/", os.sep))
        if not os.path.exists(full):
            missing.append(rel)
    return missing


def _result1b_compare_plan(config_path: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    run_dir = _result1b_run_dir_from_config_path(config_path)
    details: dict[str, Any] = {"run_dir": run_dir}
    if not os.path.isdir(run_dir):
        details["reason"] = "run_dir_missing"
        return "full", details

    missing_core = _result1b_compare_missing_core_outputs(run_dir)
    details["missing_core"] = missing_core
    if missing_core:
        details["reason"] = "missing_core_outputs"
        return "full", details

    bootstrap_enabled = bool(config.get("bootstrap_enabled", False))
    boot_csv = os.path.join(run_dir, "tables", "bootstrap_metrics_wide.csv")
    details["bootstrap_enabled"] = bootstrap_enabled
    details["bootstrap_csv"] = boot_csv
    if bootstrap_enabled and not os.path.isfile(boot_csv):
        raw_err = os.path.join(run_dir, "raw", "terminal_errors_by_agent.npy")
        raw_manifest = os.path.join(run_dir, "raw", "raw_payload_manifest.json")
        can_boot_only = os.path.isfile(raw_err) and os.path.isfile(raw_manifest)
        details["reason"] = "bootstrap_missing"
        details["bootstrap_only_possible"] = can_boot_only
        return ("bootstrap_only" if can_boot_only else "full"), details

    details["reason"] = "already_complete"
    return "skip", details


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified compare console.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config path/name under ./configs (extension optional).",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Force full compare run even if output artifacts already exist.",
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

    compare_executed = False
    if not bool(args.force_run) and pipeline == "result1b":
        action, details = _result1b_compare_plan(config_path=source_path, config=config_clean)
        run_dir = str(details.get("run_dir"))
        if action == "skip":
            print(
                f"[run:{run_name}] Compare outputs already complete at '{run_dir}'. "
                "Skipping (use --force-run to recompute)."
            )
        elif action == "bootstrap_only":
            from examples.thesis_result1b_compare_console import run_bootstrap_only_from_saved_payload

            print(
                f"[run:{run_name}] Only bootstrap output is missing at '{run_dir}'. "
                "Running bootstrap-only recovery."
            )
            run_bootstrap_only_from_saved_payload(
                run_name=run_name,
                config_path=source_path,
                config=config_clean,
            )
            compare_executed = True
        else:
            _dispatch_compare(
                pipeline=pipeline,
                run_name=run_name,
                config_path_for_snapshot=source_path,
                config=config_clean,
            )
            compare_executed = True
    else:
        _dispatch_compare(
            pipeline=pipeline,
            run_name=run_name,
            config_path_for_snapshot=source_path,
            config=config_clean,
        )
        compare_executed = True

    if pipeline == "result1b":
        if compare_executed:
            print(f"[run:{run_name}] Compare step finished. Building path plots.")
        _dispatch_result1b_paths_plot(
            run_name=run_name,
            config_path_for_snapshot=source_path,
            config=config_clean,
        )


if __name__ == "__main__":
    main()
