"""
Run all train/compare scenarios inside a config folder, sequentially.

Execution order:
1) All train configs
2) All compare configs

For compare runs, it can optionally generate full path plots using the same
configs (reusing thesis_result1b_plot_paths_console).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.compare_console import (  # noqa: E402
    _apply_relaxed_defaults_compare,
    _dispatch_compare,
)
from examples.train_console import (  # noqa: E402
    _apply_relaxed_defaults_train,
    _dispatch_train,
)
from examples.thesis_result1b_plot_paths_console import run_plot  # noqa: E402
from examples.unified_console_common import (  # noqa: E402
    CONFIGS_ROOT,
    detect_pipeline,
    load_merged_config,
    strip_meta_keys,
)


class BatchLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [batch] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()


def _resolve_folder(folder_ref: str) -> str:
    ref = str(folder_ref).strip()
    if not ref:
        raise ValueError("folder_ref cannot be empty.")
    candidates = []
    if os.path.isabs(ref):
        candidates.append(ref)
    else:
        candidates.append(os.path.normpath(os.path.join(ROOT_DIR, ref)))
        candidates.append(os.path.normpath(os.path.join(CONFIGS_ROOT, ref)))
        candidates.append(os.path.normpath(os.path.join(CONFIGS_ROOT, "runs", ref)))
    for cand in dict.fromkeys(candidates):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "Scenario folder not found. Tried:\n" + "\n".join(f"- {c}" for c in candidates)
    )


def _classify_task(config_path: str) -> str:
    path_norm = config_path.replace("\\", "/").lower()
    if "/train/" in path_norm:
        return "train"
    if "/compare/" in path_norm:
        return "compare"
    if "/calibration/" in path_norm:
        # Calibration configs are compare-like runs (e.g., NI-band sweeps).
        return "compare"
    # Fallback by raw JSON keys.
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "benchmark_agent_name" in raw:
        return "compare"
    if isinstance(raw, dict) and "agent_name" in raw and "train_paths" in raw:
        return "train"
    raise ValueError(f"Cannot classify config as train/compare: {config_path}")


def _discover_configs(folder_path: str) -> tuple[list[str], list[str]]:
    train_cfgs: list[str] = []
    compare_cfgs: list[str] = []
    for root, _, files in os.walk(folder_path):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            cfg = os.path.normpath(os.path.join(root, fname))
            task = _classify_task(cfg)
            if task == "train":
                train_cfgs.append(cfg)
            else:
                compare_cfgs.append(cfg)
    train_cfgs = sorted(set(train_cfgs))
    compare_cfgs = sorted(set(compare_cfgs))
    return train_cfgs, compare_cfgs


def _run_train_config(
    config_path: str,
    checkpoint_every_epochs: int,
    logger: BatchLogger,
) -> tuple[bool, float]:
    t0 = time.perf_counter()
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)
    run_name = str(merged.get("run_name") or os.path.splitext(os.path.basename(source_path))[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")

    cfg = strip_meta_keys(merged)
    cfg = _apply_relaxed_defaults_train(cfg, pipeline=pipeline)
    if pipeline == "result1b":
        cfg["checkpoint_enabled"] = bool(cfg.get("checkpoint_enabled", True))
        cfg["checkpoint_every_epochs"] = int(cfg.get("checkpoint_every_epochs", int(checkpoint_every_epochs)))
        if "checkpoint_every_epochs" not in merged:
            cfg["checkpoint_every_epochs"] = int(checkpoint_every_epochs)

    logger.log(
        f"TRAIN start | run={run_name} | pipeline={pipeline} | config={source_path} | "
        f"checkpoint_every_epochs={cfg.get('checkpoint_every_epochs')}"
    )
    _dispatch_train(
        pipeline=pipeline,
        run_name=run_name,
        config_path_for_snapshot=source_path,
        config=cfg,
    )
    elapsed = time.perf_counter() - t0
    logger.log(f"TRAIN done  | run={run_name} | elapsed={elapsed:.2f}s")
    return True, elapsed


def _run_compare_config(
    config_path: str,
    plot_all_paths: bool,
    logger: BatchLogger,
) -> tuple[bool, float]:
    t0 = time.perf_counter()
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)
    run_name = str(merged.get("run_name") or os.path.splitext(os.path.basename(source_path))[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")

    cfg = strip_meta_keys(merged)
    cfg = _apply_relaxed_defaults_compare(cfg, pipeline=pipeline)

    logger.log(f"COMPARE start | run={run_name} | pipeline={pipeline} | config={source_path}")
    _dispatch_compare(
        pipeline=pipeline,
        run_name=run_name,
        config_path_for_snapshot=source_path,
        config=cfg,
    )
    logger.log(f"COMPARE done  | run={run_name} | elapsed={time.perf_counter() - t0:.2f}s")

    if pipeline == "result1b" and bool(plot_all_paths):
        logger.log(f"PLOTS start   | run={run_name} | all_paths=True")
        run_plot(
            run_name=run_name,
            config=cfg,
            config_path=source_path,
            n_plot_paths=int(cfg.get("eval_paths", 80)),
            plot_all_paths=True,
            eval_paths_override=None,
        )
        logger.log(f"PLOTS done    | run={run_name}")

    elapsed = time.perf_counter() - t0
    return True, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all train/compare scenarios inside a config folder."
    )
    parser.add_argument(
        "folder_ref",
        help="Folder under ./configs (or absolute path) containing train/compare JSONs.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=5,
        help="Default checkpoint cadence for training runs when config omits it (default: 5).",
    )
    parser.add_argument(
        "--skip-compare-path-plots",
        action="store_true",
        help="No generar plots de paths luego de cada comparación.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next scenario if one run fails.",
    )
    args = parser.parse_args()

    if int(args.checkpoint_every_epochs) <= 0:
        raise ValueError("--checkpoint-every-epochs must be > 0.")

    folder_path = _resolve_folder(args.folder_ref)
    train_cfgs, compare_cfgs = _discover_configs(folder_path)
    if len(train_cfgs) == 0 and len(compare_cfgs) == 0:
        raise ValueError(f"No JSON configs found under folder: {folder_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_tag = os.path.basename(os.path.normpath(folder_path))
    log_path = os.path.join(ROOT_DIR, "logs", "batch_runs", f"{stamp}_{folder_tag}.log")
    logger = BatchLogger(log_path=log_path)
    logger.log(f"Scenario folder resolved: {folder_path}")
    logger.log(f"Train configs found: {len(train_cfgs)}")
    logger.log(f"Compare configs found: {len(compare_cfgs)}")
    logger.log(f"Batch log file: {log_path}")

    failed: list[tuple[str, str, str]] = []
    success_count = 0
    total_start = time.perf_counter()
    try:
        for idx, cfg_path in enumerate(train_cfgs, start=1):
            logger.log(f"[{idx}/{len(train_cfgs)}] Running TRAIN config: {cfg_path}")
            try:
                _run_train_config(
                    config_path=cfg_path,
                    checkpoint_every_epochs=int(args.checkpoint_every_epochs),
                    logger=logger,
                )
                success_count += 1
            except Exception as exc:
                err = traceback.format_exc()
                logger.log(f"TRAIN failed | config={cfg_path} | error={exc}")
                logger.log(err)
                failed.append(("train", cfg_path, str(exc)))
                if not bool(args.continue_on_error):
                    raise

        for idx, cfg_path in enumerate(compare_cfgs, start=1):
            logger.log(f"[{idx}/{len(compare_cfgs)}] Running COMPARE config: {cfg_path}")
            try:
                _run_compare_config(
                    config_path=cfg_path,
                    plot_all_paths=not bool(args.skip_compare_path_plots),
                    logger=logger,
                )
                success_count += 1
            except Exception as exc:
                err = traceback.format_exc()
                logger.log(f"COMPARE failed | config={cfg_path} | error={exc}")
                logger.log(err)
                failed.append(("compare", cfg_path, str(exc)))
                if not bool(args.continue_on_error):
                    raise
    finally:
        elapsed = time.perf_counter() - total_start
        logger.log(f"Batch finished | elapsed={elapsed:.2f}s | success={success_count} | failed={len(failed)}")
        if failed:
            logger.log("Failed runs summary:")
            for task, path, err in failed:
                logger.log(f"- task={task} | config={path} | error={err}")
        logger.close()


if __name__ == "__main__":
    main()
