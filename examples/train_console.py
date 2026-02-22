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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.unified_console_common import (  # noqa: E402
    detect_pipeline,
    load_merged_config,
    resolve_config_path,
    strip_meta_keys,
)


def _apply_relaxed_defaults_train(config: dict[str, Any], pipeline: str) -> dict[str, Any]:
    cfg = dict(config)
    if pipeline == "result1b":
        defaults = {
            "output_root": "assets/thesis_result1b",
            "models_dir": "models",
            "optimizers_dir": "optimizers",
            "load_if_exists": False,
            "save_after_train": True,
            "download_if_missing": True,
            "learning_rate_strategy": "exponential_decay",
            "use_price_history_context": False,
            "context_length": 0,
            "context_feature_mode": "log_returns",
            "context_pre_ttm_mode": "calculated",
            "context_for_path_generation_only": False,
            "gbm_sigma_per_path_mode": "fixed",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified train console.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config path/name under ./configs (extension optional).",
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
    _dispatch_train(
        pipeline=pipeline,
        run_name=run_name,
        config_path_for_snapshot=source_path,
        config=config_clean,
    )


if __name__ == "__main__":
    main()
