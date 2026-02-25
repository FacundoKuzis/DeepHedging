"""
Console runner for thesis Result 1b training jobs.

Bloque 1b:
- Calibra GBM en train market window (e.g. 2005-2019)
- Entrena agente deep hedging en ese mundo simple
- Guarda calibracion + artefactos con trazabilidad
"""

import argparse
import json
import os
import sys
import time
from typing import Any

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

from examples.thesis_result1_common import (  # noqa: E402
    THESIS_MODELS_ROOT,
    TRAINABLE_AGENT_NAMES,
    build_agent_from_config,
    build_claim_from_config,
    build_environment,
    build_instrument_from_config,
    build_risk_measure_from_config,
    copy_config_snapshot,
    load_config_by_name,
    set_global_determinism,
    strict_validate_keys,
    get_config_relative_stem,
    validate_agent_name,
)
from DeepHedging.utils.gbm_calibration import calibrate_gbm_from_market_data  # noqa: E402


RESULT1B_ROOT = THESIS_MODELS_ROOT


def _run_dirs(run_name: str, config_path: str) -> dict[str, str]:
    _ = run_name
    rel_stem = get_config_relative_stem(config_path)
    run_dir = os.path.normpath(os.path.join(RESULT1B_ROOT, rel_stem))
    logs_dir = os.path.join(run_dir, "logs")
    plots_dir = os.path.join(run_dir, "plots")
    tables_dir = os.path.join(run_dir, "tables")
    models_root = os.path.join(run_dir, "models")
    optimizers_root = os.path.join(run_dir, "optimizers")
    market_cache_dir = os.path.join(RESULT1B_ROOT, "market_data_cache")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(optimizers_root, exist_ok=True)
    os.makedirs(market_cache_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "tables_dir": tables_dir,
        "models_root": models_root,
        "optimizers_root": optimizers_root,
        "market_cache_dir": market_cache_dir,
    }



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
        "ticker",
        "train_start_date",
        "train_end_date",
        "interval",
        "price_col",
        "download_if_missing",
        "sigma_source",
        "implied_vol_source",
        "implied_vol_stat",
        "fixed_implied_vol",
        "risk_free_source",
        "fixed_risk_free",
        "learning_rate_strategy",
    }



def optional_keys() -> set[str]:
    return {
        "use_price_history_context",
        "context_length",
        "context_feature_mode",
        "context_pre_ttm_mode",
        "context_for_path_generation_only",
        "history_conv1d_enabled",
        "history_conv1d_layers",
        "history_conv1d_pooling",
        "instrument_model",
        "path_transformation_type",
        "include_log_strike_feature",
        "gbm_sigma_per_path_mode",
        "gbm_sigma_uniform_low",
        "gbm_sigma_uniform_high",
        "gbm_sigma_discrete_values",
        "gbm_sigma_discrete_probs",
        "garch_alpha",
        "garch_beta",
        "garch_omega",
        "garch_leverage",
        "garch_use_student_t",
        "garch_student_t_df",
        "reduce_on_plateau_factor",
        "reduce_on_plateau_patience",
        "reduce_on_plateau_min_delta",
        "reduce_on_plateau_cooldown",
        "reduce_on_plateau_min_lr",
        "early_stopping_enabled",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }



def validate_config(config: dict[str, Any]) -> None:
    strict_validate_keys(config, required_keys(), optional_keys())

    if int(config["schema_version"]) != 1:
        raise ValueError("schema_version must be 1.")
    if str(config["model_family"]).strip().lower() != "deep_hedging_result_1b":
        raise ValueError("model_family must be 'deep_hedging_result_1b'.")

    validate_agent_name(str(config["agent_name"]))
    if str(config["agent_name"]) not in TRAINABLE_AGENT_NAMES:
        raise ValueError(f"agent_name must be trainable. Allowed: {sorted(TRAINABLE_AGENT_NAMES)}")

    if int(config["n"]) <= 0:
        raise ValueError("n must be > 0")
    if int(config["trading_days_per_year"]) <= 0:
        raise ValueError("trading_days_per_year must be > 0")
    if float(config["s0"]) <= 0.0:
        raise ValueError("s0 must be > 0")
    if float(config["strike"]) <= 0.0:
        raise ValueError("strike must be > 0")
    if float(config["proportional_cost"]) < 0.0:
        raise ValueError("proportional_cost must be >= 0")

    if int(config["train_paths"]) <= 0:
        raise ValueError("train_paths must be > 0")
    if int(config["val_paths"]) < 0:
        raise ValueError("val_paths must be >= 0")
    if int(config["n_epochs"]) <= 0:
        raise ValueError("n_epochs must be > 0")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0")

    if float(config["initial_learning_rate"]) <= 0.0:
        raise ValueError("initial_learning_rate must be > 0")
    if int(config["decay_steps"]) <= 0:
        raise ValueError("decay_steps must be > 0")
    if not (0.0 < float(config["decay_rate"]) <= 1.0):
        raise ValueError("decay_rate must satisfy 0 < decay_rate <= 1")

    lr_strategy = str(config["learning_rate_strategy"]).strip().lower()
    if lr_strategy not in {"exponential_decay", "constant", "reduce_on_plateau"}:
        raise ValueError(
            "learning_rate_strategy must be 'exponential_decay', 'constant' or 'reduce_on_plateau'."
        )
    if lr_strategy == "reduce_on_plateau":
        factor = float(config.get("reduce_on_plateau_factor", 0.5))
        if not (0.0 < factor < 1.0):
            raise ValueError("reduce_on_plateau_factor must satisfy 0 < factor < 1.")
        patience = int(config.get("reduce_on_plateau_patience", 5))
        if patience < 0:
            raise ValueError("reduce_on_plateau_patience must be >= 0.")
        min_delta = float(config.get("reduce_on_plateau_min_delta", 1e-4))
        if min_delta < 0.0:
            raise ValueError("reduce_on_plateau_min_delta must be >= 0.")
        cooldown = int(config.get("reduce_on_plateau_cooldown", 0))
        if cooldown < 0:
            raise ValueError("reduce_on_plateau_cooldown must be >= 0.")
        min_lr = float(config.get("reduce_on_plateau_min_lr", 1e-6))
        if min_lr <= 0.0:
            raise ValueError("reduce_on_plateau_min_lr must be > 0.")
        if min_lr > float(config["initial_learning_rate"]):
            raise ValueError("reduce_on_plateau_min_lr cannot exceed initial_learning_rate.")

    if "early_stopping_enabled" in config and not isinstance(config["early_stopping_enabled"], bool):
        raise ValueError("early_stopping_enabled must be bool when provided.")
    if bool(config.get("early_stopping_enabled", False)):
        es_patience = int(config.get("early_stopping_patience", 20))
        if es_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0.")
        es_min_delta = float(config.get("early_stopping_min_delta", 1e-4))
        if es_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0.")

    use_context = bool(config.get("use_price_history_context", False))
    if "context_for_path_generation_only" in config and not isinstance(config["context_for_path_generation_only"], bool):
        raise ValueError("context_for_path_generation_only must be bool when provided.")
    context_pre_ttm_mode = str(config.get("context_pre_ttm_mode", "calculated")).strip().lower()
    if context_pre_ttm_mode == "extended":
        context_pre_ttm_mode = "calculated"
    if context_pre_ttm_mode not in {"calculated", "zero"}:
        raise ValueError("context_pre_ttm_mode must be 'calculated' or 'zero'.")
    if use_context:
        if "context_length" not in config:
            raise ValueError("context_length is required when use_price_history_context=true.")
        if int(config["context_length"]) <= 0:
            raise ValueError("context_length must be > 0 when use_price_history_context=true.")
        mode = str(config.get("context_feature_mode", "")).strip().lower()
        if mode not in {"log_returns", "log_moneyness"}:
            raise ValueError("context_feature_mode must be 'log_returns' or 'log_moneyness'.")
    else:
        if "context_length" in config and int(config["context_length"]) < 0:
            raise ValueError("context_length must be >= 0.")

    if "history_conv1d_enabled" in config and not isinstance(config["history_conv1d_enabled"], bool):
        raise ValueError("history_conv1d_enabled must be bool when provided.")
    history_conv1d_enabled = bool(config.get("history_conv1d_enabled", False))
    if history_conv1d_enabled:
        if not use_context:
            raise ValueError("history_conv1d_enabled=true requires use_price_history_context=true.")
        if bool(config.get("context_for_path_generation_only", False)):
            raise ValueError(
                "history_conv1d_enabled=true is incompatible with context_for_path_generation_only=true."
            )
        layers = config.get("history_conv1d_layers")
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError(
                "history_conv1d_layers must be a non-empty list when history_conv1d_enabled=true."
            )
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"history_conv1d_layers[{i}] must be an object.")
            if "filters" not in layer or "kernel_size" not in layer:
                raise ValueError(
                    f"history_conv1d_layers[{i}] must include 'filters' and 'kernel_size'."
                )
            if int(layer["filters"]) <= 0:
                raise ValueError(f"history_conv1d_layers[{i}].filters must be > 0.")
            if int(layer["kernel_size"]) <= 0:
                raise ValueError(f"history_conv1d_layers[{i}].kernel_size must be > 0.")
            if "dilation_rate" in layer and int(layer["dilation_rate"]) <= 0:
                raise ValueError(f"history_conv1d_layers[{i}].dilation_rate must be > 0 when provided.")
            if "dropout" in layer:
                d = float(layer["dropout"])
                if d < 0.0 or d >= 1.0:
                    raise ValueError(f"history_conv1d_layers[{i}].dropout must satisfy 0 <= dropout < 1.")
            if "activation" in layer and not str(layer["activation"]).strip():
                raise ValueError(f"history_conv1d_layers[{i}].activation cannot be empty when provided.")
        pooling = str(config.get("history_conv1d_pooling", "global_max")).strip().lower()
        if pooling not in {"global_max", "global_avg"}:
            raise ValueError(
                "history_conv1d_pooling must be 'global_max' or 'global_avg' when history_conv1d_enabled=true."
            )

    if "include_log_strike_feature" in config and not isinstance(config["include_log_strike_feature"], bool):
        raise ValueError("include_log_strike_feature must be bool when provided.")
    path_t = str(config.get("path_transformation_type", "log_moneyness")).strip().lower()
    if path_t not in {"none", "log", "log_moneyness"}:
        raise ValueError("path_transformation_type must be one of {'none','log','log_moneyness'}.")
    instrument_model = str(config.get("instrument_model", "gbm")).strip().lower()
    if instrument_model not in {"gbm", "garch"}:
        raise ValueError("instrument_model must be one of {'gbm','garch'}.")

    sigma_path_mode = str(config.get("gbm_sigma_per_path_mode", "fixed")).strip().lower()
    if sigma_path_mode not in {"fixed", "uniform", "discrete"}:
        raise ValueError("gbm_sigma_per_path_mode must be one of {'fixed','uniform','discrete'}.")
    if sigma_path_mode == "uniform":
        if "gbm_sigma_uniform_low" not in config or "gbm_sigma_uniform_high" not in config:
            raise ValueError(
                "gbm_sigma_uniform_low and gbm_sigma_uniform_high are required when gbm_sigma_per_path_mode='uniform'."
            )
        lo = float(config["gbm_sigma_uniform_low"])
        hi = float(config["gbm_sigma_uniform_high"])
        if lo <= 0.0 or hi <= 0.0 or lo >= hi:
            raise ValueError("Require 0 < gbm_sigma_uniform_low < gbm_sigma_uniform_high.")
    if sigma_path_mode == "discrete":
        values = config.get("gbm_sigma_discrete_values")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                "gbm_sigma_discrete_values must be a non-empty list when gbm_sigma_per_path_mode='discrete'."
            )
        for v in values:
            if float(v) <= 0.0:
                raise ValueError("gbm_sigma_discrete_values must contain values > 0.")
        probs = config.get("gbm_sigma_discrete_probs")
        if probs is not None:
            if not isinstance(probs, list) or len(probs) != len(values):
                raise ValueError(
                    "gbm_sigma_discrete_probs must be a list with same length as gbm_sigma_discrete_values."
                )
            if any(float(p) < 0.0 for p in probs):
                raise ValueError("gbm_sigma_discrete_probs must be >= 0.")
            if sum(float(p) for p in probs) <= 0.0:
                raise ValueError("gbm_sigma_discrete_probs must sum to > 0.")

    if instrument_model == "garch":
        alpha = float(config.get("garch_alpha", 0.05))
        beta = float(config.get("garch_beta", 0.9))
        if alpha < 0.0:
            raise ValueError("garch_alpha must be >= 0.")
        if beta < 0.0:
            raise ValueError("garch_beta must be >= 0.")
        if alpha + beta >= 1.0:
            raise ValueError("Require garch_alpha + garch_beta < 1 for stability.")
        if "garch_omega" in config and config["garch_omega"] is not None:
            if float(config["garch_omega"]) <= 0.0:
                raise ValueError("garch_omega must be > 0 when provided.")
        if "garch_leverage" in config and not isinstance(config["garch_leverage"], (int, float)):
            raise ValueError("garch_leverage must be numeric when provided.")
        leverage = float(config.get("garch_leverage", 0.0))
        stability_lhs = alpha + beta + 2.0 * leverage
        if stability_lhs >= 1.0:
            raise ValueError(
                "Require garch_alpha + garch_beta + 2*garch_leverage < 1 for stationarity. "
                f"Got {stability_lhs:.6f}."
            )
        if "garch_use_student_t" in config and not isinstance(config["garch_use_student_t"], bool):
            raise ValueError("garch_use_student_t must be bool when provided.")
        if bool(config.get("garch_use_student_t", False)):
            if float(config.get("garch_student_t_df", 8.0)) <= 2.0:
                raise ValueError("garch_student_t_df must be > 2 when garch_use_student_t=true.")

    for key in [
        "run_name",
        "description",
        "model_name",
        "ticker",
        "train_start_date",
        "train_end_date",
        "interval",
        "price_col",
        "models_dir",
        "optimizers_dir",
    ]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    for key in ["download_if_missing", "load_if_exists", "save_after_train", "resample_each_epoch"]:
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be bool")

    sigma_source = str(config["sigma_source"]).strip().lower()
    if sigma_source not in {"historical", "implied"}:
        raise ValueError("sigma_source must be 'historical' or 'implied'.")

    implied_vol_source = str(config["implied_vol_source"]).strip().lower()
    if implied_vol_source not in {"vix", "fixed", "option_market"}:
        raise ValueError("implied_vol_source must be 'vix', 'fixed' or 'option_market'.")

    implied_vol_stat = str(config["implied_vol_stat"]).strip().lower()
    if implied_vol_stat not in {"mean", "median"}:
        raise ValueError("implied_vol_stat must be 'mean' or 'median'.")

    risk_free_source = str(config["risk_free_source"]).strip().lower()
    if risk_free_source not in {"irx", "fixed"}:
        raise ValueError("risk_free_source must be 'irx' or 'fixed'.")

    if implied_vol_source == "fixed":
        if config["fixed_implied_vol"] is None or float(config["fixed_implied_vol"]) <= 0.0:
            raise ValueError("fixed_implied_vol must be > 0 when implied_vol_source='fixed'.")

    if risk_free_source == "fixed":
        if config["fixed_risk_free"] is None:
            raise ValueError("fixed_risk_free must be provided when risk_free_source='fixed'.")



def _model_optimizer_paths(agent, model_name: str, dirs: dict[str, str]) -> tuple[str, str]:
    model_path = os.path.join(dirs["models_root"], agent.name, f"{model_name}.keras")
    optimizer_path = os.path.join(dirs["optimizers_root"], agent.name, model_name)
    return model_path, optimizer_path


def run_training(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    dirs = _run_dirs(run_name, config_path=config_path)
    cfg_snapshot = copy_config_snapshot(config_path, dirs["run_dir"])
    resolved_cfg_path = os.path.join(dirs["run_dir"], "resolved_config.json")
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[run:{run_name}] Config validated and copied to: {cfg_snapshot}")
    print(f"[run:{run_name}] Storage root: {RESULT1B_ROOT}")

    seed = int(config["global_random_seed"])
    set_global_determinism(seed)

    calib = calibrate_gbm_from_market_data(
        market_cache_dir=dirs["market_cache_dir"],
        ticker=str(config["ticker"]),
        train_start_date=str(config["train_start_date"]),
        train_end_date=str(config["train_end_date"]),
        interval=str(config["interval"]),
        price_col=str(config["price_col"]),
        trading_days_per_year=int(config["trading_days_per_year"]),
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
    cfg_for_builders.setdefault("instrument_model", str(config.get("instrument_model", "gbm")).strip().lower())
    instrument = build_instrument_from_config(cfg_for_builders)

    claim = build_claim_from_config(cfg_for_builders)
    agent = build_agent_from_config(
        agent_name=str(config["agent_name"]),
        instrument=instrument,
        claim=claim,
        config=cfg_for_builders,
    )
    risk_measure = build_risk_measure_from_config(cfg_for_builders)

    lr_strategy = str(config["learning_rate_strategy"]).strip().lower()
    if lr_strategy == "exponential_decay":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(config["initial_learning_rate"]),
            decay_steps=int(config["decay_steps"]),
            decay_rate=float(config["decay_rate"]),
            staircase=True,
        )
    else:
        lr_schedule = float(config["initial_learning_rate"])

    plateau_callback = None
    if lr_strategy == "reduce_on_plateau":
        reduce_factor = float(config.get("reduce_on_plateau_factor", 0.5))
        reduce_patience = int(config.get("reduce_on_plateau_patience", 5))
        reduce_min_delta = float(config.get("reduce_on_plateau_min_delta", 1e-4))
        reduce_cooldown = int(config.get("reduce_on_plateau_cooldown", 0))
        reduce_min_lr = float(config.get("reduce_on_plateau_min_lr", 1e-6))
        plateau_state = {
            "best": None,
            "bad_epochs": 0,
            "cooldown": 0,
        }

        def _plateau_epoch_callback(epoch_info: dict[str, Any]) -> dict[str, Any]:
            val_loss = epoch_info.get("val_loss")
            monitored = float(val_loss) if val_loss is not None else float(epoch_info["train_loss"])
            current_lr = float(epoch_info["learning_rate"])
            best = plateau_state["best"]
            improved = False
            if best is None or monitored < float(best) - reduce_min_delta:
                improved = True
                plateau_state["best"] = monitored
                plateau_state["bad_epochs"] = 0
            else:
                if plateau_state["cooldown"] > 0:
                    plateau_state["cooldown"] -= 1
                else:
                    plateau_state["bad_epochs"] += 1

            if improved:
                return {}
            if plateau_state["cooldown"] > 0:
                return {}
            if plateau_state["bad_epochs"] < reduce_patience:
                return {}

            new_lr = max(current_lr * reduce_factor, reduce_min_lr)
            plateau_state["bad_epochs"] = 0
            plateau_state["cooldown"] = reduce_cooldown
            if new_lr >= current_lr - 1e-15:
                return {}
            print(
                f"[run:{run_name}] ReduceOnPlateau: monitored={monitored:.6f}, "
                f"lr {current_lr:.6g} -> {new_lr:.6g}"
            )
            return {"set_learning_rate": float(new_lr)}

        plateau_callback = _plateau_epoch_callback

    early_stop_enabled = bool(config.get("early_stopping_enabled", False))
    early_patience = int(config.get("early_stopping_patience", 20))
    early_min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    early_state = {
        "best": None,
        "bad_epochs": 0,
    }

    def _combined_epoch_callback(epoch_info: dict[str, Any]) -> dict[str, Any]:
        callback_out: dict[str, Any] = {}

        if plateau_callback is not None:
            plateau_out = plateau_callback(epoch_info)
            if isinstance(plateau_out, dict):
                callback_out.update(plateau_out)

        if early_stop_enabled:
            val_loss = epoch_info.get("val_loss")
            monitored = float(val_loss) if val_loss is not None else float(epoch_info["train_loss"])
            best = early_state["best"]
            improved = (best is None) or (monitored < float(best) - early_min_delta)
            if improved:
                early_state["best"] = monitored
                early_state["bad_epochs"] = 0
            else:
                early_state["bad_epochs"] += 1
                if early_state["bad_epochs"] >= early_patience:
                    print(
                        f"[run:{run_name}] EarlyStopping: no improvement for "
                        f"{early_patience} epochs (best={float(early_state['best']):.6f}, "
                        f"current={monitored:.6f}). Stopping."
                    )
                    callback_out["stop_training"] = True

        return callback_out

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
        use_price_history_context=bool(config.get("use_price_history_context", False)),
        context_length=int(config.get("context_length", 0)),
        context_feature_mode=str(config.get("context_feature_mode", "log_returns")),
        context_visible_to_agent=not bool(config.get("context_for_path_generation_only", False)),
    )

    model_path, optimizer_path = _model_optimizer_paths(
        agent=agent,
        model_name=str(config["model_name"]),
        dirs=dirs,
    )

    if bool(config["load_if_exists"]):
        if os.path.isfile(model_path):
            agent.load_model(model_path)
            print(f"[run:{run_name}] Loaded existing model: {model_path}")
        if os.path.isdir(optimizer_path):
            env.load_optimizer(optimizer_path, only_weights=True)
            print(f"[run:{run_name}] Loaded existing optimizer: {optimizer_path}")

    print(
        f"[run:{run_name}] Training agent={config['agent_name']} claim={config['contingent_claim']} "
        f"instrument={str(config.get('instrument_model', 'gbm')).strip().lower()} "
        f"with calibrated r={calib.r_train:.6f}, sigma={calib.sigma_train:.6f}"
    )
    t0 = time.perf_counter()
    out = env.train(
        train_paths=int(config["train_paths"]),
        val_paths=int(config["val_paths"]),
        random_seed=seed,
        epoch_end_callback=_combined_epoch_callback,
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
            {
                "epoch": list(range(1, len(train_losses) + 1)),
                "train_loss": train_losses,
            }
        )

    hist_path = os.path.join(dirs["tables_dir"], "training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[run:{run_name}] Saved training history: {hist_path}")

    calibration_manifest = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "ticker": str(config["ticker"]),
                "train_start_date": str(config["train_start_date"]),
                "train_end_date": str(config["train_end_date"]),
                "sigma_source": str(config["sigma_source"]),
                "implied_vol_source": str(config["implied_vol_source"]),
                "risk_free_source": str(config["risk_free_source"]),
                "use_price_history_context": bool(config.get("use_price_history_context", False)),
                "context_for_path_generation_only": bool(config.get("context_for_path_generation_only", False)),
                "context_length": int(config.get("context_length", 0)),
                "context_feature_mode": str(config.get("context_feature_mode", "log_returns")),
                "context_pre_ttm_mode": str(config.get("context_pre_ttm_mode", "calculated")),
                "history_conv1d_enabled": bool(config.get("history_conv1d_enabled", False)),
                "history_conv1d_layers": json.dumps(config.get("history_conv1d_layers")),
                "history_conv1d_pooling": str(config.get("history_conv1d_pooling", "global_max")),
                "instrument_model": str(config.get("instrument_model", "gbm")),
                "gbm_sigma_per_path_mode": str(config.get("gbm_sigma_per_path_mode", "fixed")),
                "gbm_sigma_uniform_low": config.get("gbm_sigma_uniform_low"),
                "gbm_sigma_uniform_high": config.get("gbm_sigma_uniform_high"),
                "gbm_sigma_discrete_values": json.dumps(config.get("gbm_sigma_discrete_values")),
                "gbm_sigma_discrete_probs": json.dumps(config.get("gbm_sigma_discrete_probs")),
                "garch_alpha": config.get("garch_alpha"),
                "garch_beta": config.get("garch_beta"),
                "garch_omega": config.get("garch_omega"),
                "garch_leverage": config.get("garch_leverage"),
                "garch_use_student_t": config.get("garch_use_student_t"),
                "garch_student_t_df": config.get("garch_student_t_df"),
                "learning_rate_strategy": str(config.get("learning_rate_strategy", "constant")),
                "early_stopping_enabled": bool(config.get("early_stopping_enabled", False)),
                "early_stopping_patience": int(config.get("early_stopping_patience", 20)),
                "early_stopping_min_delta": float(config.get("early_stopping_min_delta", 1e-4)),
                "sigma_train": float(calib.sigma_train),
                "r_train": float(calib.r_train),
                "mu_train": float(calib.mu_train),
                "train_rows": int(calib.train_rows),
                "close_csv_path": str(calib.close_csv_path),
                "implied_csv_path": "" if calib.implied_csv_path is None else str(calib.implied_csv_path),
                "risk_free_csv_path": "" if calib.risk_free_csv_path is None else str(calib.risk_free_csv_path),
            }
        ]
    )
    calib_path = os.path.join(dirs["tables_dir"], "calibration_manifest.csv")
    calibration_manifest.to_csv(calib_path, index=False)
    print(f"[run:{run_name}] Saved calibration manifest: {calib_path}")

    if bool(config["save_after_train"]):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
        agent.save_model(model_path)
        env.save_optimizer(optimizer_path)
        print(f"[run:{run_name}] Saved model: {model_path}")
        print(f"[run:{run_name}] Saved optimizer: {optimizer_path}")

    metadata = {
        "run_name": run_name,
        "agent_name": str(config["agent_name"]),
        "model_name": str(config["model_name"]),
        "contingent_claim": str(config["contingent_claim"]),
        "seed": seed,
        "elapsed_seconds": float(elapsed),
        "storage_root": RESULT1B_ROOT,
        "calibrated_r": float(calib.r_train),
        "calibrated_sigma": float(calib.sigma_train),
        "model_path": model_path,
        "optimizer_path": optimizer_path,
    }
    metadata_path = os.path.join(dirs["run_dir"], "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[run:{run_name}] Saved run metadata: {metadata_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis Result 1b model from JSON config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional config filename in ./thesis_result1b_configs/train (extension optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1b_configs", "train")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, cfg_path, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter TRAIN JSON config name from 'thesis_result1b_configs/train': ",
    )
    print(f"[run:{run_name}] Loaded config: {cfg_path}")
    validate_config(cfg)
    run_training(run_name=run_name, config_path=cfg_path, config=cfg)


if __name__ == "__main__":
    main()
