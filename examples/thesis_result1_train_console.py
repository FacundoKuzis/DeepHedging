"""
Console runner for thesis Result 1 training jobs.

Behavior:
1. Reads only JSON file name (CLI arg optional, otherwise prompt).
2. Loads config from ./thesis_result1_configs/train.
3. Validates strict schema (no missing, no extra keys).
4. Trains exactly one trainable agent run and saves model + optimizer.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# Determinism flags should be set before TensorFlow import.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.thesis_result1_common import (
    THESIS_MODELS_ROOT,
    TRAINABLE_AGENT_NAMES,
    build_agent_from_config,
    build_claim_from_config,
    build_environment,
    build_instrument_from_config,
    build_risk_measure_from_config,
    copy_config_snapshot,
    ensure_run_dirs,
    load_config_by_name,
    model_and_optimizer_paths,
    set_global_determinism,
    strict_validate_keys,
    validate_common_market_fields,
    validate_agent_name,
)


def required_keys() -> set[str]:
    return {
        "schema_version",
        "model_family",
        "run_name",
        "description",
        "output_root",
        "agent_name",
        "model_name",
        "s0",
        "n",
        "trading_days_per_year",
        "r",
        "sigma",
        "strike",
        "contingent_claim",
        "claim_underlying_index",
        "fixing_indices",
        "proportional_cost",
        "risk_measure_name",
        "cvar_alpha",
        "train_paths",
        "val_paths",
        "n_epochs",
        "batch_size",
        "initial_learning_rate",
        "decay_steps",
        "decay_rate",
        "resample_each_epoch",
        "global_random_seed",
        "models_dir",
        "optimizers_dir",
        "load_if_exists",
        "save_after_train",
    }


def optional_keys() -> set[str]:
    return {
        "no_intervention_bound",
        "no_intervention_mode",
        "benchmark_no_intervention_mode",
        "checkpoint_every_epochs",
        "checkpoint_compare_every_epochs",
        "checkpoint_compare_agent_name",
        "checkpoint_compare_paths",
        "checkpoint_compare_seed",
        "checkpoint_compare_plot_min_x",
        "checkpoint_compare_plot_max_x",
        "learning_rate_strategy",
        "reduce_on_plateau_factor",
        "reduce_on_plateau_patience",
        "reduce_on_plateau_min_delta",
        "reduce_on_plateau_min_learning_rate",
        "reduce_on_plateau_cooldown",
        "reduce_on_plateau_monitor",
        "early_stopping_enabled",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_monitor",
        "extend_training_enabled",
        "extend_training_patience",
        "extend_training_min_delta",
        "extend_training_monitor",
        "extend_training_epochs_increment",
        "extend_training_max_total_epochs",
    }


def _optional_positive_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or int(value) <= 0:
        raise ValueError(f"{field_name} must be null or an integer > 0.")
    return int(value)


def _optional_non_negative_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or int(value) < 0:
        raise ValueError(f"{field_name} must be null or an integer >= 0.")
    return int(value)


def validate_config(config: dict[str, Any]) -> None:
    strict_validate_keys(config, required_keys(), optional_keys())
    validate_common_market_fields(config)

    validate_agent_name(str(config["agent_name"]))
    if str(config["agent_name"]) not in TRAINABLE_AGENT_NAMES:
        raise ValueError(
            f"agent_name must be trainable. Allowed: {sorted(TRAINABLE_AGENT_NAMES)}."
        )

    if float(config["proportional_cost"]) < 0.0:
        raise ValueError("proportional_cost must be >= 0.")
    if int(config["train_paths"]) <= 0:
        raise ValueError("train_paths must be > 0.")
    if int(config["val_paths"]) < 0:
        raise ValueError("val_paths must be >= 0.")
    if int(config["n_epochs"]) <= 0:
        raise ValueError("n_epochs must be > 0.")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0.")
    if float(config["initial_learning_rate"]) <= 0.0:
        raise ValueError("initial_learning_rate must be > 0.")
    if int(config["decay_steps"]) <= 0:
        raise ValueError("decay_steps must be > 0.")
    if not (0.0 < float(config["decay_rate"]) <= 1.0):
        raise ValueError("decay_rate must satisfy 0 < decay_rate <= 1.")

    for key in ["output_root", "run_name", "model_name", "models_dir", "optimizers_dir"]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    for key in ["load_if_exists", "save_after_train", "resample_each_epoch"]:
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean.")

    if "no_intervention_bound" in config:
        val = config["no_intervention_bound"]
        if val is not None and (not isinstance(val, (int, float)) or float(val) < 0.0):
            raise ValueError("no_intervention_bound must be null or a number >= 0.")
    for mode_key in ["no_intervention_mode", "benchmark_no_intervention_mode"]:
        if mode_key in config:
            mode_val = str(config[mode_key]).strip().lower()
            if mode_val not in {"absolute", "percentage"}:
                raise ValueError(f"{mode_key} must be 'absolute' or 'percentage'.")
    effective_mode = str(
        config.get(
            "benchmark_no_intervention_mode",
            config.get("no_intervention_mode", "absolute"),
        )
    ).strip().lower()
    effective_bound = config.get("no_intervention_bound", 0.0)
    if effective_mode == "percentage" and effective_bound is not None:
        bound_float = float(effective_bound)
        if not (0.0 <= bound_float <= 1.0):
            raise ValueError(
                "For percentage mode, no_intervention_bound must be in [0, 1]. "
                "Example: 0.05 = 5%."
            )

    lr_strategy = str(config.get("learning_rate_strategy", "exponential_decay")).strip().lower()
    if lr_strategy not in {"exponential_decay", "constant", "reduce_on_plateau"}:
        raise ValueError("learning_rate_strategy must be one of: exponential_decay, constant, reduce_on_plateau.")

    monitor_allowed = {"train_loss", "val_loss"}
    for monitor_key in [
        "reduce_on_plateau_monitor",
        "early_stopping_monitor",
        "extend_training_monitor",
    ]:
        monitor_val = str(config.get(monitor_key, "val_loss")).strip().lower()
        if monitor_val not in monitor_allowed:
            raise ValueError(f"{monitor_key} must be one of {sorted(monitor_allowed)}.")

    if "reduce_on_plateau_factor" in config:
        val = float(config["reduce_on_plateau_factor"])
        if not (0.0 < val < 1.0):
            raise ValueError("reduce_on_plateau_factor must satisfy 0 < factor < 1.")
    _optional_non_negative_int(config.get("reduce_on_plateau_patience"), "reduce_on_plateau_patience")
    if "reduce_on_plateau_min_delta" in config and float(config["reduce_on_plateau_min_delta"]) < 0.0:
        raise ValueError("reduce_on_plateau_min_delta must be >= 0.")
    if "reduce_on_plateau_min_learning_rate" in config and float(config["reduce_on_plateau_min_learning_rate"]) <= 0.0:
        raise ValueError("reduce_on_plateau_min_learning_rate must be > 0.")
    _optional_non_negative_int(config.get("reduce_on_plateau_cooldown"), "reduce_on_plateau_cooldown")

    if "early_stopping_enabled" in config and not isinstance(config["early_stopping_enabled"], bool):
        raise ValueError("early_stopping_enabled must be bool when provided.")
    _optional_non_negative_int(config.get("early_stopping_patience"), "early_stopping_patience")
    if "early_stopping_min_delta" in config and float(config["early_stopping_min_delta"]) < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0.")

    if "extend_training_enabled" in config and not isinstance(config["extend_training_enabled"], bool):
        raise ValueError("extend_training_enabled must be bool when provided.")
    _optional_non_negative_int(config.get("extend_training_patience"), "extend_training_patience")
    if "extend_training_min_delta" in config and float(config["extend_training_min_delta"]) < 0.0:
        raise ValueError("extend_training_min_delta must be >= 0.")
    _optional_positive_int(
        config.get("extend_training_epochs_increment"),
        "extend_training_epochs_increment",
    )
    _optional_positive_int(
        config.get("extend_training_max_total_epochs"),
        "extend_training_max_total_epochs",
    )
    if "extend_training_max_total_epochs" in config and config["extend_training_max_total_epochs"] is not None:
        if int(config["extend_training_max_total_epochs"]) < int(config["n_epochs"]):
            raise ValueError("extend_training_max_total_epochs must be >= n_epochs.")

    checkpoint_every = _optional_positive_int(
        config.get("checkpoint_every_epochs"), "checkpoint_every_epochs"
    )
    compare_every = _optional_positive_int(
        config.get("checkpoint_compare_every_epochs"), "checkpoint_compare_every_epochs"
    )
    if compare_every is not None:
        compare_agent_name = config.get("checkpoint_compare_agent_name")
        if not isinstance(compare_agent_name, str) or not compare_agent_name.strip():
            raise ValueError(
                "checkpoint_compare_agent_name must be a non-empty string when checkpoint_compare_every_epochs is set."
            )
        validate_agent_name(compare_agent_name)
        if compare_agent_name in TRAINABLE_AGENT_NAMES:
            raise ValueError("checkpoint_compare_agent_name must be a benchmark (non-trainable) agent.")

        compare_paths = config.get("checkpoint_compare_paths")
        if not isinstance(compare_paths, int) or int(compare_paths) <= 0:
            raise ValueError(
                "checkpoint_compare_paths must be an integer > 0 when checkpoint_compare_every_epochs is set."
            )

    if "checkpoint_compare_seed" in config and config["checkpoint_compare_seed"] is not None:
        if not isinstance(config["checkpoint_compare_seed"], int):
            raise ValueError("checkpoint_compare_seed must be null or integer.")
    if "checkpoint_compare_plot_min_x" in config and config["checkpoint_compare_plot_min_x"] is not None:
        if not isinstance(config["checkpoint_compare_plot_min_x"], (int, float)):
            raise ValueError("checkpoint_compare_plot_min_x must be null or number.")
    if "checkpoint_compare_plot_max_x" in config and config["checkpoint_compare_plot_max_x"] is not None:
        if not isinstance(config["checkpoint_compare_plot_max_x"], (int, float)):
            raise ValueError("checkpoint_compare_plot_max_x must be null or number.")
    if (
        config.get("checkpoint_compare_plot_min_x") is not None
        and config.get("checkpoint_compare_plot_max_x") is not None
        and float(config["checkpoint_compare_plot_min_x"]) >= float(config["checkpoint_compare_plot_max_x"])
    ):
        raise ValueError("checkpoint_compare_plot_min_x must be < checkpoint_compare_plot_max_x.")

    _ = checkpoint_every  # explicit for readability in validation path


def run_training(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    run_dirs = ensure_run_dirs(
        output_root=str(config["output_root"]),
        run_type="train",
        run_name=run_name,
    )
    copied_cfg = copy_config_snapshot(config_path, run_dirs["run_dir"])
    print(f"[run:{run_name}] Config validated and copied to: {copied_cfg}")
    print(f"[run:{run_name}] Storage root: {THESIS_MODELS_ROOT}")

    seed = int(config["global_random_seed"])
    set_global_determinism(seed)

    instrument = build_instrument_from_config(config)
    claim = build_claim_from_config(config)
    agent = build_agent_from_config(
        agent_name=str(config["agent_name"]),
        instrument=instrument,
        claim=claim,
        config=config,
    )
    risk_measure = build_risk_measure_from_config(config)

    checkpoint_every = config.get("checkpoint_every_epochs")
    checkpoint_every = int(checkpoint_every) if checkpoint_every is not None else None

    compare_every = config.get("checkpoint_compare_every_epochs")
    compare_every = int(compare_every) if compare_every is not None else None
    checkpoint_compare_agent = None
    checkpoint_compare_paths = None
    checkpoint_compare_seed = int(config["global_random_seed"])
    checkpoint_compare_plot_min_x = -1.0
    checkpoint_compare_plot_max_x = 1.0

    if compare_every is not None:
        checkpoint_compare_agent = build_agent_from_config(
            agent_name=str(config["checkpoint_compare_agent_name"]),
            instrument=instrument,
            claim=claim,
            config=config,
        )
        checkpoint_compare_paths = int(config["checkpoint_compare_paths"])
        if config.get("checkpoint_compare_seed") is not None:
            checkpoint_compare_seed = int(config["checkpoint_compare_seed"])
        if config.get("checkpoint_compare_plot_min_x") is not None:
            checkpoint_compare_plot_min_x = float(config["checkpoint_compare_plot_min_x"])
        if config.get("checkpoint_compare_plot_max_x") is not None:
            checkpoint_compare_plot_max_x = float(config["checkpoint_compare_plot_max_x"])
        print(
            f"[run:{run_name}] Enabled checkpoint comparisons every {compare_every} epochs "
            f"against {config['checkpoint_compare_agent_name']}."
        )

    lr_strategy = str(config.get("learning_rate_strategy", "exponential_decay")).strip().lower()
    if lr_strategy == "exponential_decay":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(config["initial_learning_rate"]),
            decay_steps=int(config["decay_steps"]),
            decay_rate=float(config["decay_rate"]),
            staircase=True,
        )
    elif lr_strategy in {"constant", "reduce_on_plateau"}:
        lr_schedule = float(config["initial_learning_rate"])
    else:
        raise ValueError(f"Unsupported learning_rate_strategy: {lr_strategy}")

    # Build environment with resolved LR strategy.
    env = build_environment(
        agent=agent,
        instrument=instrument,
        claim=claim,
        proportional_cost=float(config["proportional_cost"]),
        risk_measure=risk_measure,
        n_epochs=int(config["n_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate_schedule=lr_schedule,
        optimizer_cls=tf.keras.optimizers.Adam,
        resample_each_epoch=bool(config["resample_each_epoch"]),
        train_seed=seed,
    )

    model_path, optimizer_path = model_and_optimizer_paths(
        agent=agent,
        model_name=str(config["model_name"]),
        models_dir=str(config["models_dir"]),
        optimizers_dir=str(config["optimizers_dir"]),
    )

    if bool(config["load_if_exists"]):
        if os.path.isfile(model_path):
            agent.load_model(model_path)
            print(f"[run:{run_name}] Loaded existing model: {model_path}")
        if os.path.isdir(optimizer_path):
            env.load_optimizer(optimizer_path, only_weights=True)
            print(f"[run:{run_name}] Loaded existing optimizer: {optimizer_path}")

    reduce_on_plateau_enabled = lr_strategy == "reduce_on_plateau"
    reduce_cfg = {
        "factor": float(config.get("reduce_on_plateau_factor", 0.5)),
        "patience": int(config.get("reduce_on_plateau_patience", 5)),
        "min_delta": float(config.get("reduce_on_plateau_min_delta", 1e-4)),
        "min_lr": float(config.get("reduce_on_plateau_min_learning_rate", 1e-6)),
        "cooldown": int(config.get("reduce_on_plateau_cooldown", 0)),
        "monitor": str(config.get("reduce_on_plateau_monitor", "val_loss")).strip().lower(),
    }
    reduce_state = {
        "best": None,
        "bad_epochs": 0,
        "cooldown_counter": 0,
    }

    early_cfg = {
        "enabled": bool(config.get("early_stopping_enabled", False)),
        "patience": int(config.get("early_stopping_patience", 10)),
        "min_delta": float(config.get("early_stopping_min_delta", 1e-4)),
        "monitor": str(config.get("early_stopping_monitor", "val_loss")).strip().lower(),
    }
    early_state = {
        "best": None,
        "bad_epochs": 0,
    }

    extend_cfg = {
        "enabled": bool(config.get("extend_training_enabled", False)),
        "patience": int(config.get("extend_training_patience", 2)),
        "min_delta": float(config.get("extend_training_min_delta", 1e-4)),
        "monitor": str(config.get("extend_training_monitor", "val_loss")).strip().lower(),
        "increment": int(config.get("extend_training_epochs_increment", 20)),
        "max_total_epochs": int(
            config.get("extend_training_max_total_epochs", int(config["n_epochs"]))
        ),
    }
    extend_state = {
        "best": None,
        "bad_epochs": 0,
    }
    if extend_cfg["enabled"]:
        print(
            f"[run:{run_name}] Extend training enabled: increment={extend_cfg['increment']}, "
            f"max_total_epochs={extend_cfg['max_total_epochs']}."
        )
    if early_cfg["enabled"]:
        print(
            f"[run:{run_name}] Early stopping enabled: patience={early_cfg['patience']}, "
            f"monitor={early_cfg['monitor']}."
        )
    if reduce_on_plateau_enabled:
        print(
            f"[run:{run_name}] ReduceLROnPlateau enabled: factor={reduce_cfg['factor']}, "
            f"patience={reduce_cfg['patience']}, monitor={reduce_cfg['monitor']}."
        )

    def _metric_value(epoch_info: dict[str, Any], monitor_name: str) -> float:
        if monitor_name == "val_loss":
            val = epoch_info.get("val_loss")
            if val is not None:
                return float(val)
        return float(epoch_info["train_loss"])

    def _update_improvement_state(state: dict[str, Any], metric: float, min_delta: float) -> bool:
        best = state["best"]
        if best is None or metric < (float(best) - float(min_delta)):
            state["best"] = float(metric)
            state["bad_epochs"] = 0
            return True
        state["bad_epochs"] += 1
        return False

    def _epoch_end_callback(epoch_info: dict[str, Any]) -> dict[str, Any]:
        epoch = int(epoch_info["epoch"])
        actions: dict[str, Any] = {}

        if checkpoint_every is not None and (epoch % checkpoint_every == 0):
            checkpoint_dir = os.path.join(
                run_dirs["run_dir"],
                "checkpoints",
                f"epoch_{epoch:04d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_model_path = os.path.join(checkpoint_dir, "model.keras")
            checkpoint_optimizer_path = os.path.join(checkpoint_dir, "optimizer")
            agent.save_model(checkpoint_model_path)
            env.save_optimizer(checkpoint_optimizer_path)
            print(f"[run:{run_name}] Saved checkpoint model: {checkpoint_model_path}")
            print(f"[run:{run_name}] Saved checkpoint optimizer: {checkpoint_optimizer_path}")

        if compare_every is not None and (epoch % compare_every == 0):
            compare_plot_dir = os.path.join(run_dirs["plots_dir"], "epoch_comparisons")
            compare_table_dir = os.path.join(run_dirs["tables_dir"], "epoch_comparisons")
            os.makedirs(compare_plot_dir, exist_ok=True)
            os.makedirs(compare_table_dir, exist_ok=True)

            compare_plot_path = os.path.join(compare_plot_dir, f"epoch_{epoch:04d}.jpg")
            compare_stats_path = os.path.join(compare_table_dir, f"epoch_{epoch:04d}.xlsx")
            env.terminal_hedging_error_multiple_agents(
                agents=[checkpoint_compare_agent, agent],
                n_paths=int(checkpoint_compare_paths),
                random_seed=int(checkpoint_compare_seed + epoch),
                plot_error=True,
                plot_title="Error de Cobertura Terminal",
                save_plot_path=compare_plot_path,
                save_stats_path=compare_stats_path,
                loss_functions=None,
                min_x=float(checkpoint_compare_plot_min_x),
                max_x=float(checkpoint_compare_plot_max_x),
                language="es",
                pricing_method="fixed",
            )
            print(f"[run:{run_name}] Saved checkpoint comparison plot: {compare_plot_path}")
            print(f"[run:{run_name}] Saved checkpoint comparison stats: {compare_stats_path}")

        if reduce_on_plateau_enabled:
            metric = _metric_value(epoch_info, reduce_cfg["monitor"])
            improved = _update_improvement_state(
                state=reduce_state,
                metric=metric,
                min_delta=reduce_cfg["min_delta"],
            )
            if improved:
                reduce_state["cooldown_counter"] = max(0, int(reduce_state["cooldown_counter"]))
            else:
                if int(reduce_state["cooldown_counter"]) > 0:
                    reduce_state["cooldown_counter"] -= 1
                elif int(reduce_state["bad_epochs"]) >= int(reduce_cfg["patience"]):
                    current_lr = float(epoch_info["learning_rate"])
                    new_lr = max(float(reduce_cfg["min_lr"]), current_lr * float(reduce_cfg["factor"]))
                    if new_lr < (current_lr - 1e-12):
                        actions["set_learning_rate"] = float(new_lr)
                        print(
                            f"[run:{run_name}] Epoch {epoch}: ReduceLROnPlateau "
                            f"{current_lr:.6g} -> {new_lr:.6g}"
                        )
                    reduce_state["bad_epochs"] = 0
                    reduce_state["cooldown_counter"] = int(reduce_cfg["cooldown"])

        if early_cfg["enabled"]:
            metric = _metric_value(epoch_info, early_cfg["monitor"])
            _update_improvement_state(
                state=early_state,
                metric=metric,
                min_delta=early_cfg["min_delta"],
            )
            if int(early_state["bad_epochs"]) >= int(early_cfg["patience"]):
                actions["stop_training"] = True
                print(
                    f"[run:{run_name}] Early stopping at epoch {epoch} "
                    f"(monitor={early_cfg['monitor']}, bad_epochs={early_state['bad_epochs']})."
                )

        if extend_cfg["enabled"]:
            metric = _metric_value(epoch_info, extend_cfg["monitor"])
            _update_improvement_state(
                state=extend_state,
                metric=metric,
                min_delta=extend_cfg["min_delta"],
            )
            planned_epochs = int(epoch_info["planned_epochs"])
            max_total = int(extend_cfg["max_total_epochs"])
            if (
                epoch >= planned_epochs
                and planned_epochs < max_total
                and int(extend_state["bad_epochs"]) < int(extend_cfg["patience"])
            ):
                remaining = max_total - planned_epochs
                extend_by = min(int(extend_cfg["increment"]), int(remaining))
                if extend_by > 0:
                    actions["extend_n_epochs_by"] = int(extend_by)
                    print(
                        f"[run:{run_name}] Extending training by {extend_by} epochs "
                        f"(new planned total: {planned_epochs + extend_by})."
                    )

        return actions

    print(f"[run:{run_name}] Training agent={config['agent_name']} claim={config['contingent_claim']}...")
    t0 = time.perf_counter()
    out = env.train(
        train_paths=int(config["train_paths"]),
        val_paths=int(config["val_paths"]),
        random_seed=seed,
        epoch_end_callback=_epoch_end_callback,
    )
    elapsed = time.perf_counter() - t0
    print(f"[run:{run_name}] Training finished in {elapsed:.2f}s")

    if int(config["val_paths"]) > 0:
        train_losses, val_losses = out
        hist_df = pd.DataFrame(
            {
                "epoch": list(range(1, len(train_losses) + 1)),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }
        )
    else:
        train_losses = out
        hist_df = pd.DataFrame(
            {"epoch": list(range(1, len(train_losses) + 1)), "train_loss": train_losses}
        )
    actual_epochs = int(len(train_losses))

    hist_path = os.path.join(run_dirs["tables_dir"], "training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[run:{run_name}] Saved training history: {hist_path}")

    meta = {
        "run_name": run_name,
        "agent_name": str(config["agent_name"]),
        "agent_internal_name": agent.name,
        "claim": str(config["contingent_claim"]),
        "proportional_cost": float(config["proportional_cost"]),
        "train_paths": int(config["train_paths"]),
        "val_paths": int(config["val_paths"]),
        "n_epochs": int(config["n_epochs"]),
        "actual_epochs_trained": actual_epochs,
        "batch_size": int(config["batch_size"]),
        "seed": seed,
        "elapsed_seconds": elapsed,
        "model_path": model_path,
        "optimizer_path": optimizer_path,
        "storage_root": THESIS_MODELS_ROOT,
        "checkpoint_every_epochs": checkpoint_every,
        "checkpoint_compare_every_epochs": compare_every,
        "learning_rate_strategy": lr_strategy,
        "early_stopping_enabled": bool(early_cfg["enabled"]),
        "extend_training_enabled": bool(extend_cfg["enabled"]),
    }
    meta_path = os.path.join(run_dirs["run_dir"], "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[run:{run_name}] Saved run metadata: {meta_path}")

    if bool(config["save_after_train"]):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
        agent.save_model(model_path)
        env.save_optimizer(optimizer_path)
        print(f"[run:{run_name}] Saved model: {model_path}")
        print(f"[run:{run_name}] Saved optimizer: {optimizer_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis Result 1 model from JSON config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional config filename in ./thesis_result1_configs/train (extension optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1_configs", "train")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, cfg_path, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter TRAIN JSON config name from 'thesis_result1_configs/train' (example: european_no_cost_lstm.json): ",
    )
    print(f"[run:{run_name}] Loaded config: {cfg_path}")
    validate_config(cfg)
    run_training(run_name=run_name, config_path=cfg_path, config=cfg)


if __name__ == "__main__":
    main()
