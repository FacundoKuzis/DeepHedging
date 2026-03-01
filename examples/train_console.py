"""
Unified train runner for thesis pipelines.

Supported pipelines:
- result1
- result1b
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
import glob

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


def _apply_relaxed_defaults_train(config: dict[str, Any], pipeline: str) -> dict[str, Any]:
    cfg = dict(config)
    if pipeline == "result1b":
        defaults = {
            "output_root": "assets/thesis_result1b",
            "models_dir": "models",
            "optimizers_dir": "optimizers",
            "load_if_exists": False,
            "save_after_train": True,
            "checkpoint_enabled": True,
            "checkpoint_every_epochs": 5,
            "checkpoint_save_best": True,
            "checkpoint_metric": "auto",
            "checkpoint_save_optimizer": True,
            "checkpoint_resume_if_available": True,
            "download_if_missing": True,
            "learning_rate_strategy": "exponential_decay",
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
            "fixed_implied_vol": None,
            "fixed_risk_free": None,
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)
    return cfg


def _dispatch_train(pipeline: str, run_name: str, config_path_for_snapshot: str, config: dict[str, Any]) -> None:
    if pipeline == "result1":
        from examples.thesis_result1_train_console import run_training, validate_config
    elif pipeline == "result1b":
        from examples.thesis_result1b_train_console import run_training, validate_config
    else:
        raise ValueError(f"Unsupported pipeline for train: {pipeline}")

    validate_config(config)
    run_training(run_name=run_name, config_path=config_path_for_snapshot, config=config)


def _result1b_train_run_dir_from_config_path(config_path: str) -> str:
    rel_stem = _get_config_relative_stem(config_path)
    return os.path.normpath(os.path.join(THESIS_MODELS_ROOT, rel_stem))


def _result1b_train_missing_outputs(config_path: str, config: dict[str, Any]) -> tuple[str, list[str]]:
    run_dir = _result1b_train_run_dir_from_config_path(config_path)
    missing: list[str] = []
    if not os.path.isdir(run_dir):
        missing.append("run_dir")
        return run_dir, missing

    required_rel = [
        "run_metadata.json",
        "tables/calibration_manifest.csv",
        "tables/training_history.csv",
    ]
    for rel in required_rel:
        full = os.path.join(run_dir, rel.replace("/", os.sep))
        if not os.path.exists(full):
            missing.append(rel)

    if bool(config.get("save_after_train", True)):
        model_name = str(config.get("model_name", "")).strip()
        if model_name:
            model_glob = os.path.join(run_dir, "models", "**", f"{model_name}.keras")
            model_hits = glob.glob(model_glob, recursive=True)
            if len(model_hits) == 0:
                missing.append(f"models/**/{model_name}.keras")

            optimizer_glob = os.path.join(run_dir, "optimizers", "**", model_name)
            opt_hits = [p for p in glob.glob(optimizer_glob, recursive=True) if os.path.isdir(p)]
            if len(opt_hits) == 0:
                missing.append(f"optimizers/**/{model_name}")
        else:
            missing.append("model_name")

    return run_dir, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified train console.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config path/name under ./configs (extension optional).",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Force full training run even if output artifacts already exist.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(
        config_name=args.config_name,
        task="train",
        prompt_label="Enter TRAIN config name/path from 'configs': ",
    )
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)

    run_name = str(merged.get("run_name") or os.path.splitext(os.path.basename(source_path))[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")

    config_clean = strip_meta_keys(merged)
    config_clean = _apply_relaxed_defaults_train(config_clean, pipeline=pipeline)

    print(f"[run:{run_name}] Loaded config: {source_path}")
    print(f"[run:{run_name}] Pipeline: {pipeline}")

    if not bool(args.force_run) and pipeline == "result1b":
        run_dir, missing = _result1b_train_missing_outputs(
            config_path=source_path,
            config=config_clean,
        )
        if len(missing) == 0:
            print(
                f"[run:{run_name}] Training outputs already complete at '{run_dir}'. "
                "Skipping (use --force-run to retrain)."
            )
            return

    _dispatch_train(
        pipeline=pipeline,
        run_name=run_name,
        config_path_for_snapshot=source_path,
        config=config_clean,
    )


if __name__ == "__main__":
    main()
