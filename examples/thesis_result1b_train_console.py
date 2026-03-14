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
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import numpy as np
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
        "wavenet_num_filters",
        "wavenet_num_residual_blocks",
        "wavenet_block_configs",
        "wavenet_activation",
        "wavenet_use_skip_connections",
        "wavenet_output_hidden_filters",
        "sequence_output_mode",
        "position_activation",
        "gbm_sigma_per_path_mode",
        "gbm_sigma_uniform_low",
        "gbm_sigma_uniform_high",
        "gbm_sigma_discrete_values",
        "gbm_sigma_discrete_probs",
        "r_per_path_mode",
        "r_uniform_low",
        "r_uniform_high",
        "r_discrete_values",
        "r_discrete_probs",
        "garch_alpha",
        "garch_beta",
        "garch_omega",
        "garch_leverage",
        "garch_use_student_t",
        "garch_student_t_df",
        "garch_alpha_per_path_mode",
        "garch_alpha_uniform_low",
        "garch_alpha_uniform_high",
        "garch_alpha_discrete_values",
        "garch_alpha_discrete_probs",
        "garch_beta_per_path_mode",
        "garch_beta_uniform_low",
        "garch_beta_uniform_high",
        "garch_beta_discrete_values",
        "garch_beta_discrete_probs",
        "garch_leverage_per_path_mode",
        "garch_leverage_uniform_low",
        "garch_leverage_uniform_high",
        "garch_leverage_discrete_values",
        "garch_leverage_discrete_probs",
        "garch_student_t_df_per_path_mode",
        "garch_student_t_df_uniform_low",
        "garch_student_t_df_uniform_high",
        "garch_student_t_df_discrete_values",
        "garch_student_t_df_discrete_probs",
        "hmm_transition_matrix",
        "hmm_initial_distribution",
        "hmm_vol_multipliers",
        "hmm_r_multipliers",
        "hmm_params_per_path_mode",
        "hmm_num_states",
        "hmm_transition_uniform_low",
        "hmm_transition_uniform_high",
        "hmm_initial_uniform_low",
        "hmm_initial_uniform_high",
        "hmm_vol_multipliers_uniform_low",
        "hmm_vol_multipliers_uniform_high",
        "hmm_vol_multipliers_sort",
        "hmm_r_multipliers_uniform_low",
        "hmm_r_multipliers_uniform_high",
        "student_t_df",
        "student_t_df_per_path_mode",
        "student_t_df_uniform_low",
        "student_t_df_uniform_high",
        "student_t_df_discrete_values",
        "student_t_df_discrete_probs",
        "tail_shock_enabled",
        "tail_shock_magnitude_low",
        "tail_shock_magnitude_high",
        "tail_shock_gap_low",
        "tail_shock_gap_high",
        "reduce_on_plateau_factor",
        "reduce_on_plateau_patience",
        "reduce_on_plateau_min_delta",
        "reduce_on_plateau_cooldown",
        "reduce_on_plateau_min_lr",
        "early_stopping_enabled",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "max_epochs_extension_enabled",
        "max_epochs_extension_window_epochs",
        "max_epochs_extension_by",
        "max_added_epochs",
        "checkpoint_enabled",
        "checkpoint_every_epochs",
        "checkpoint_save_best",
        "checkpoint_metric",
        "checkpoint_save_optimizer",
        "checkpoint_resume_if_available",
        "dense_units",
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

    if "max_epochs_extension_enabled" in config and not isinstance(config["max_epochs_extension_enabled"], bool):
        raise ValueError("max_epochs_extension_enabled must be bool when provided.")
    if bool(config.get("max_epochs_extension_enabled", False)):
        extend_window = int(config.get("max_epochs_extension_window_epochs", 10))
        if extend_window <= 0:
            raise ValueError("max_epochs_extension_window_epochs must be > 0.")
        extend_by = int(config.get("max_epochs_extension_by", 10))
        if extend_by <= 0:
            raise ValueError("max_epochs_extension_by must be > 0.")
        max_added_epochs = int(config.get("max_added_epochs", 100))
        if max_added_epochs <= 0:
            raise ValueError("max_added_epochs must be > 0 when max_epochs_extension_enabled=true.")

    if "checkpoint_enabled" in config and not isinstance(config["checkpoint_enabled"], bool):
        raise ValueError("checkpoint_enabled must be bool when provided.")
    if bool(config.get("checkpoint_enabled", False)):
        if int(config.get("checkpoint_every_epochs", 1)) <= 0:
            raise ValueError("checkpoint_every_epochs must be > 0 when checkpoint_enabled=true.")
        if "checkpoint_save_best" in config and not isinstance(config["checkpoint_save_best"], bool):
            raise ValueError("checkpoint_save_best must be bool when provided.")
        metric_name = str(config.get("checkpoint_metric", "auto")).strip().lower()
        if metric_name not in {"auto", "val_loss", "train_loss"}:
            raise ValueError("checkpoint_metric must be one of {'auto','val_loss','train_loss'}.")
        if "checkpoint_save_optimizer" in config and not isinstance(config["checkpoint_save_optimizer"], bool):
            raise ValueError("checkpoint_save_optimizer must be bool when provided.")
        if "checkpoint_resume_if_available" in config and not isinstance(
            config["checkpoint_resume_if_available"], bool
        ):
            raise ValueError("checkpoint_resume_if_available must be bool when provided.")

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

    if "sequence_output_mode" in config:
        seq_out = str(config["sequence_output_mode"]).strip().lower()
        if seq_out not in {"trade", "position"}:
            raise ValueError("sequence_output_mode must be 'trade' or 'position'.")
    if "position_activation" in config:
        pos_act = str(config["position_activation"]).strip().lower()
        if pos_act not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("position_activation must be one of {'linear','sigmoid','tanh'}.")
    if "wavenet_num_filters" in config:
        if int(config["wavenet_num_filters"]) <= 0:
            raise ValueError("wavenet_num_filters must be > 0.")
    if "wavenet_num_residual_blocks" in config:
        if int(config["wavenet_num_residual_blocks"]) <= 0:
            raise ValueError("wavenet_num_residual_blocks must be > 0.")
    if "wavenet_activation" in config and not str(config["wavenet_activation"]).strip():
        raise ValueError("wavenet_activation cannot be empty when provided.")
    if "wavenet_use_skip_connections" in config and not isinstance(config["wavenet_use_skip_connections"], bool):
        raise ValueError("wavenet_use_skip_connections must be bool when provided.")
    if "wavenet_output_hidden_filters" in config:
        if int(config["wavenet_output_hidden_filters"]) < 0:
            raise ValueError("wavenet_output_hidden_filters must be >= 0.")
    if "wavenet_block_configs" in config:
        blocks = config["wavenet_block_configs"]
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise ValueError("wavenet_block_configs must be a non-empty list when provided.")
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError(f"wavenet_block_configs[{i}] must be an object.")
            if "filters" not in block or "kernel_size" not in block:
                raise ValueError(
                    f"wavenet_block_configs[{i}] must include 'filters' and 'kernel_size'."
                )
            if int(block["filters"]) <= 0:
                raise ValueError(f"wavenet_block_configs[{i}].filters must be > 0.")
            if int(block["kernel_size"]) <= 0:
                raise ValueError(f"wavenet_block_configs[{i}].kernel_size must be > 0.")
            if "dilation_rate" in block and int(block["dilation_rate"]) <= 0:
                raise ValueError(f"wavenet_block_configs[{i}].dilation_rate must be > 0.")
            if "dropout" in block:
                d = float(block["dropout"])
                if d < 0.0 or d >= 1.0:
                    raise ValueError(
                        f"wavenet_block_configs[{i}].dropout must satisfy 0 <= dropout < 1."
                    )
            if "activation" in block and not str(block["activation"]).strip():
                raise ValueError(
                    f"wavenet_block_configs[{i}].activation cannot be empty when provided."
                )

    if "include_log_strike_feature" in config and not isinstance(config["include_log_strike_feature"], bool):
        raise ValueError("include_log_strike_feature must be bool when provided.")
    path_t = str(config.get("path_transformation_type", "log_moneyness")).strip().lower()
    if path_t not in {"none", "log", "log_moneyness"}:
        raise ValueError("path_transformation_type must be one of {'none','log','log_moneyness'}.")
    instrument_model = str(config.get("instrument_model", "gbm")).strip().lower()
    if instrument_model not in {"gbm", "garch", "hmm_garch", "student_t"}:
        raise ValueError("instrument_model must be one of {'gbm','garch','hmm_garch','student_t'}.")

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

    def _validate_mode_for_param(
        param_name: str,
        default_value: float,
        min_inclusive: float | None = None,
        min_exclusive: float | None = None,
    ):
        mode = str(config.get(f"{param_name}_per_path_mode", "fixed")).strip().lower()
        if mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError(
                f"{param_name}_per_path_mode must be one of {'fixed','uniform','discrete'}."
            )
        values = []
        if mode == "fixed":
            values = [float(config.get(param_name, default_value))]
        elif mode == "uniform":
            lo_key = f"{param_name}_uniform_low"
            hi_key = f"{param_name}_uniform_high"
            if lo_key not in config or hi_key not in config:
                raise ValueError(
                    f"{lo_key} and {hi_key} are required when {param_name}_per_path_mode='uniform'."
                )
            lo = float(config[lo_key])
            hi = float(config[hi_key])
            if lo >= hi:
                raise ValueError(f"Require {lo_key} < {hi_key}.")
            values = [lo, hi]
        else:
            vals_key = f"{param_name}_discrete_values"
            probs_key = f"{param_name}_discrete_probs"
            vals = config.get(vals_key)
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(
                    f"{vals_key} must be a non-empty list when {param_name}_per_path_mode='discrete'."
                )
            values = [float(v) for v in vals]
            probs = config.get(probs_key)
            if probs is not None:
                if not isinstance(probs, list) or len(probs) != len(vals):
                    raise ValueError(
                        f"{probs_key} must be a list with same length as {vals_key}."
                    )
                if any(float(p) < 0.0 for p in probs):
                    raise ValueError(f"{probs_key} must be >= 0.")
                if sum(float(p) for p in probs) <= 0.0:
                    raise ValueError(f"{probs_key} must sum to > 0.")
        if min_inclusive is not None:
            if any(float(v) < float(min_inclusive) for v in values):
                raise ValueError(f"{param_name} values must be >= {float(min_inclusive)}.")
        if min_exclusive is not None:
            if any(float(v) <= float(min_exclusive) for v in values):
                raise ValueError(f"{param_name} values must be > {float(min_exclusive)}.")
        return mode, values

    if instrument_model in {"garch", "hmm_garch"}:
        _, alpha_vals = _validate_mode_for_param("garch_alpha", 0.05, min_inclusive=0.0)
        _, beta_vals = _validate_mode_for_param("garch_beta", 0.9, min_inclusive=0.0)
        _, lev_vals = _validate_mode_for_param("garch_leverage", 0.0, min_inclusive=0.0)
        _, r_vals = _validate_mode_for_param("r", float(config.get("fixed_risk_free", 0.0)))
        if max(alpha_vals) + max(beta_vals) >= 1.0:
            raise ValueError("Require max(garch_alpha) + max(garch_beta) < 1 for stability.")
        worst_stationary_lhs = max(alpha_vals) + max(beta_vals) + 2.0 * max(lev_vals)
        if worst_stationary_lhs >= 1.0:
            raise ValueError(
                "Require max(garch_alpha) + max(garch_beta) + 2*max(garch_leverage) < 1 "
                f"for stationarity. Got {worst_stationary_lhs:.6f}."
            )
        if "garch_omega" in config and config["garch_omega"] is not None:
            if float(config["garch_omega"]) <= 0.0:
                raise ValueError("garch_omega must be > 0 when provided.")
        if "garch_use_student_t" in config and not isinstance(config["garch_use_student_t"], bool):
            raise ValueError("garch_use_student_t must be bool when provided.")
        if bool(config.get("garch_use_student_t", False)):
            _validate_mode_for_param("garch_student_t_df", 8.0, min_exclusive=2.0)
        if "tail_shock_enabled" in config and not isinstance(config["tail_shock_enabled"], bool):
            raise ValueError("tail_shock_enabled must be bool when provided.")
        if bool(config.get("tail_shock_enabled", False)):
            ts_lo = float(config.get("tail_shock_magnitude_low", 0.02))
            ts_hi = float(config.get("tail_shock_magnitude_high", 0.10))
            if not (0.0 < ts_lo < ts_hi < 1.0):
                raise ValueError(
                    "Require 0 < tail_shock_magnitude_low < tail_shock_magnitude_high < 1."
                )
            gap_lo = int(config.get("tail_shock_gap_low", 10))
            gap_hi = int(config.get("tail_shock_gap_high", 30))
            if gap_lo <= 0 or gap_hi < gap_lo:
                raise ValueError("Require 0 < tail_shock_gap_low <= tail_shock_gap_high.")
        if any(not np.isfinite(float(v)) for v in r_vals):
            raise ValueError("r values must be finite.")

        if instrument_model == "hmm_garch":
            legacy_hmm_keys = [
                k
                for k in config.keys()
                if k.startswith("hmm_p_") or k.startswith("hmm_vol_multiplier_")
            ]
            if legacy_hmm_keys:
                raise ValueError(
                    "Legacy HMM scalar keys are no longer supported. "
                    f"Remove: {sorted(legacy_hmm_keys)}"
                )
            hmm_mode = str(config.get("hmm_params_per_path_mode", "fixed")).strip().lower()
            if hmm_mode not in {"fixed", "uniform_random"}:
                raise ValueError("hmm_params_per_path_mode must be one of {'fixed','uniform_random'}.")

            if hmm_mode == "fixed":
                tm = config.get("hmm_transition_matrix")
                pi = config.get("hmm_initial_distribution")
                mult = config.get("hmm_vol_multipliers")
                r_mult = config.get("hmm_r_multipliers")
                if tm is None or pi is None or mult is None or r_mult is None:
                    raise ValueError(
                        "instrument_model='hmm_garch' with hmm_params_per_path_mode='fixed' "
                        "requires: hmm_transition_matrix, hmm_initial_distribution, "
                        "hmm_vol_multipliers, hmm_r_multipliers."
                    )
                tm_arr = np.asarray(tm, dtype=np.float64)
                pi_arr = np.asarray(pi, dtype=np.float64).reshape(-1)
                mult_arr = np.asarray(mult, dtype=np.float64).reshape(-1)
                r_mult_arr = np.asarray(r_mult, dtype=np.float64).reshape(-1)
                if tm_arr.ndim != 2 or tm_arr.shape[0] != tm_arr.shape[1]:
                    raise ValueError("hmm_transition_matrix must be a square matrix (KxK).")
                k = int(tm_arr.shape[0])
                if k < 2:
                    raise ValueError("hmm_transition_matrix must have K >= 2.")
                if pi_arr.shape[0] != k:
                    raise ValueError("hmm_initial_distribution length must match hmm_transition_matrix size.")
                if mult_arr.shape[0] != k:
                    raise ValueError("hmm_vol_multipliers length must match hmm_transition_matrix size.")
                if r_mult_arr.shape[0] != k:
                    raise ValueError("hmm_r_multipliers length must match hmm_transition_matrix size.")
                if np.any(~np.isfinite(tm_arr)) or np.any(tm_arr < 0.0):
                    raise ValueError("hmm_transition_matrix must contain finite entries >= 0.")
                row_sums = np.sum(tm_arr, axis=1)
                if np.any(row_sums <= 0.0):
                    raise ValueError("Each row of hmm_transition_matrix must sum to > 0.")
                if np.any(~np.isfinite(pi_arr)) or np.any(pi_arr < 0.0):
                    raise ValueError("hmm_initial_distribution must contain finite entries >= 0.")
                pi_sum = float(np.sum(pi_arr))
                if pi_sum <= 0.0:
                    raise ValueError("hmm_initial_distribution must sum to > 0.")
                if np.any(~np.isfinite(mult_arr)) or np.any(mult_arr <= 0.0):
                    raise ValueError("hmm_vol_multipliers must contain finite entries > 0.")
                if np.any(~np.isfinite(r_mult_arr)) or np.any(r_mult_arr <= 0.0):
                    raise ValueError("hmm_r_multipliers must contain finite entries > 0.")
            else:
                if "hmm_num_states" not in config:
                    raise ValueError("hmm_num_states is required when hmm_params_per_path_mode='uniform_random'.")
                k = int(config["hmm_num_states"])
                if k < 2:
                    raise ValueError("hmm_num_states must be >= 2.")
                t_lo = float(config.get("hmm_transition_uniform_low", 0.0))
                t_hi = float(config.get("hmm_transition_uniform_high", 1.0))
                if t_lo < 0.0 or t_hi <= t_lo:
                    raise ValueError("Require 0 <= hmm_transition_uniform_low < hmm_transition_uniform_high.")
                i_lo = float(config.get("hmm_initial_uniform_low", 0.0))
                i_hi = float(config.get("hmm_initial_uniform_high", 1.0))
                if i_lo < 0.0 or i_hi <= i_lo:
                    raise ValueError("Require 0 <= hmm_initial_uniform_low < hmm_initial_uniform_high.")
                m_lo = float(config.get("hmm_vol_multipliers_uniform_low", 0.5))
                m_hi = float(config.get("hmm_vol_multipliers_uniform_high", 2.0))
                if m_lo <= 0.0 or m_hi <= m_lo:
                    raise ValueError(
                        "Require 0 < hmm_vol_multipliers_uniform_low < hmm_vol_multipliers_uniform_high."
                    )
                r_lo = float(config.get("hmm_r_multipliers_uniform_low", 0.5))
                r_hi = float(config.get("hmm_r_multipliers_uniform_high", 1.5))
                if r_lo <= 0.0 or r_hi <= r_lo:
                    raise ValueError(
                        "Require 0 < hmm_r_multipliers_uniform_low < hmm_r_multipliers_uniform_high."
                    )
                if "hmm_vol_multipliers_sort" in config and not isinstance(
                    config["hmm_vol_multipliers_sort"], bool
                ):
                    raise ValueError("hmm_vol_multipliers_sort must be bool when provided.")

    if instrument_model == "student_t":
        df_mode = str(config.get("student_t_df_per_path_mode", "fixed")).strip().lower()
        if df_mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError(
                "student_t_df_per_path_mode must be one of {'fixed','uniform','discrete'}."
            )
        if df_mode == "fixed":
            if float(config.get("student_t_df", 8.0)) <= 2.0:
                raise ValueError("student_t_df must be > 2 when student_t_df_per_path_mode='fixed'.")
        elif df_mode == "uniform":
            if "student_t_df_uniform_low" not in config or "student_t_df_uniform_high" not in config:
                raise ValueError(
                    "student_t_df_uniform_low and student_t_df_uniform_high are required "
                    "when student_t_df_per_path_mode='uniform'."
                )
            lo = float(config["student_t_df_uniform_low"])
            hi = float(config["student_t_df_uniform_high"])
            if lo <= 2.0 or hi <= 2.0 or lo >= hi:
                raise ValueError("Require 2 < student_t_df_uniform_low < student_t_df_uniform_high.")
        else:
            vals = config.get("student_t_df_discrete_values")
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(
                    "student_t_df_discrete_values must be a non-empty list when "
                    "student_t_df_per_path_mode='discrete'."
                )
            for v in vals:
                if float(v) <= 2.0:
                    raise ValueError("student_t_df_discrete_values must contain values > 2.")
            probs = config.get("student_t_df_discrete_probs")
            if probs is not None:
                if not isinstance(probs, list) or len(probs) != len(vals):
                    raise ValueError(
                        "student_t_df_discrete_probs must be a list with same length as "
                        "student_t_df_discrete_values."
                    )
                if any(float(p) < 0.0 for p in probs):
                    raise ValueError("student_t_df_discrete_probs must be >= 0.")
                if sum(float(p) for p in probs) <= 0.0:
                    raise ValueError("student_t_df_discrete_probs must sum to > 0.")

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
    max_epochs_extension_enabled = bool(config.get("max_epochs_extension_enabled", False))
    max_epochs_extension_window_epochs = int(config.get("max_epochs_extension_window_epochs", 10))
    max_epochs_extension_by = int(config.get("max_epochs_extension_by", 10))
    max_added_epochs = int(config.get("max_added_epochs", 100))
    early_state = {
        "best": None,
        "bad_epochs": 0,
    }
    max_epochs_extension_state = {
        "best_metric": None,
        "best_epoch": None,
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

    checkpoint_enabled = bool(config.get("checkpoint_enabled", False))
    checkpoint_every_epochs = int(config.get("checkpoint_every_epochs", 1))
    checkpoint_save_best = bool(config.get("checkpoint_save_best", True))
    checkpoint_metric = str(config.get("checkpoint_metric", "auto")).strip().lower()
    checkpoint_save_optimizer = bool(config.get("checkpoint_save_optimizer", True))
    checkpoint_resume_if_available = bool(config.get("checkpoint_resume_if_available", True))
    checkpoint_state: dict[str, Any] = {
        "best_metric": None,
        "best_epoch": None,
    }
    base_total_epochs = int(config["n_epochs"])
    max_total_epochs = int(base_total_epochs + max_added_epochs)
    planned_total_epochs = int(base_total_epochs)
    resume_start_epoch = 0
    skip_training = False
    _warned_missing_val_once = False

    def _remaining_extension_budget() -> int:
        return max(0, int(max_total_epochs) - int(planned_total_epochs))

    def _apply_dynamic_extension(best_epoch: int, reason: str) -> int:
        nonlocal planned_total_epochs
        remaining = _remaining_extension_budget()
        if remaining <= 0:
            print(
                f"[run:{run_name}] Dynamic extension skipped ({reason}): "
                f"max_added_epochs reached ({max_added_epochs})."
            )
            return 0
        extend_now = min(int(max_epochs_extension_by), int(remaining))
        if extend_now <= 0:
            return 0
        prev_total = int(planned_total_epochs)
        planned_total_epochs = int(planned_total_epochs) + int(extend_now)
        print(
            f"[run:{run_name}] Dynamic extension ({reason}): planned_epochs {prev_total} -> "
            f"{planned_total_epochs} (best_epoch={int(best_epoch)} in last "
            f"{int(max_epochs_extension_window_epochs)} epochs, added={int(extend_now)}, "
            f"remaining_budget={_remaining_extension_budget()})."
        )
        return int(extend_now)

    def _maybe_extend_plan_from_recent_best(resume_epoch: int) -> None:
        """
        If training has already reached the planned limit but the best epoch is
        still inside the most recent extension window, extend the plan so a
        resumed run keeps training.
        """
        nonlocal planned_total_epochs
        if not max_epochs_extension_enabled:
            return
        best_epoch = checkpoint_state.get("best_epoch")
        if best_epoch is None:
            return
        while resume_epoch >= int(planned_total_epochs):
            in_recent_window = int(best_epoch) > (
                int(planned_total_epochs) - int(max_epochs_extension_window_epochs)
            )
            if not in_recent_window:
                break
            added = _apply_dynamic_extension(int(best_epoch), reason="resume")
            if added <= 0:
                break

    checkpoints_root = os.path.join(dirs["run_dir"], "checkpoints")
    latest_ckpt_model_path = os.path.join(checkpoints_root, "latest", os.path.basename(model_path))
    latest_ckpt_optimizer_path = os.path.join(checkpoints_root, "latest", "optimizer")
    latest_ckpt_meta_path = os.path.join(checkpoints_root, "latest", "metadata.json")
    best_ckpt_model_path = os.path.join(checkpoints_root, "best", os.path.basename(model_path))
    best_ckpt_optimizer_path = os.path.join(checkpoints_root, "best", "optimizer")
    best_ckpt_meta_path = os.path.join(checkpoints_root, "best", "metadata.json")

    def _safe_remove_path(path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)

    def _atomic_replace_path(src_path: str, dst_path: str) -> None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        _safe_remove_path(dst_path)
        os.replace(src_path, dst_path)

    def _load_checkpoint_metadata(meta_path: str) -> dict[str, Any] | None:
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _save_checkpoint(
        target_model_path: str,
        target_optimizer_path: str,
        target_meta_path: str,
        *,
        kind: str,
        epoch_abs: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        model_root, model_ext = os.path.splitext(target_model_path)
        if not model_ext:
            model_ext = ".keras"
        tmp_model_path = f"{model_root}.tmp_epoch{int(epoch_abs):04d}{model_ext}"
        tmp_optimizer_path = f"{target_optimizer_path}.tmp_epoch{int(epoch_abs):04d}"
        tmp_meta_path = f"{target_meta_path}.tmp_epoch{int(epoch_abs):04d}"

        _safe_remove_path(tmp_model_path)
        _safe_remove_path(f"{tmp_model_path}.history_conv.pkl")
        _safe_remove_path(tmp_optimizer_path)
        _safe_remove_path(tmp_meta_path)

        agent.save_model(tmp_model_path)
        if checkpoint_save_optimizer:
            env.save_optimizer(tmp_optimizer_path)

        tmp_sidecar_path = f"{tmp_model_path}.history_conv.pkl"
        target_sidecar_path = f"{target_model_path}.history_conv.pkl"
        _atomic_replace_path(tmp_model_path, target_model_path)
        if os.path.exists(tmp_sidecar_path):
            _atomic_replace_path(tmp_sidecar_path, target_sidecar_path)
        elif os.path.exists(target_sidecar_path):
            # If no sidecar was produced for this checkpoint, remove stale one.
            _safe_remove_path(target_sidecar_path)

        if checkpoint_save_optimizer:
            _atomic_replace_path(tmp_optimizer_path, target_optimizer_path)

        meta_payload = {
            "kind": str(kind),
            "run_name": str(run_name),
            "epoch": int(epoch_abs),
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
            "model_path": str(target_model_path),
            "optimizer_path": str(target_optimizer_path) if checkpoint_save_optimizer else None,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        os.makedirs(os.path.dirname(tmp_meta_path), exist_ok=True)
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)
        _atomic_replace_path(tmp_meta_path, target_meta_path)

    if bool(config["load_if_exists"]):
        if os.path.isfile(model_path):
            agent.load_model(model_path)
            print(f"[run:{run_name}] Loaded existing model: {model_path}")
        if os.path.isdir(optimizer_path):
            env.load_optimizer(optimizer_path, only_weights=True)
            print(f"[run:{run_name}] Loaded existing optimizer: {optimizer_path}")
        if (
            checkpoint_enabled
            and checkpoint_resume_if_available
            and (not os.path.isfile(model_path))
            and os.path.isfile(latest_ckpt_model_path)
        ):
            agent.load_model(latest_ckpt_model_path)
            if checkpoint_save_optimizer and os.path.isdir(latest_ckpt_optimizer_path):
                env.load_optimizer(latest_ckpt_optimizer_path, only_weights=True)
            latest_meta = _load_checkpoint_metadata(latest_ckpt_meta_path)
            if latest_meta is not None:
                resume_start_epoch = max(0, int(latest_meta.get("epoch", 0)))
            if checkpoint_save_best:
                best_meta = _load_checkpoint_metadata(best_ckpt_meta_path)
                if best_meta is not None:
                    checkpoint_state["best_metric"] = float(best_meta.get("metric_value"))
                    checkpoint_state["best_epoch"] = int(best_meta.get("epoch"))
            _maybe_extend_plan_from_recent_best(resume_start_epoch)
            remaining_epochs = max(0, planned_total_epochs - resume_start_epoch)
            if remaining_epochs <= 0:
                skip_training = True
                print(
                    f"[run:{run_name}] Resume checkpoint already reached epoch {resume_start_epoch}/{planned_total_epochs}. "
                    "Skipping training."
                )
            else:
                env.n_epochs = int(remaining_epochs)
                print(
                    f"[run:{run_name}] Resuming from latest checkpoint: epoch={resume_start_epoch}, "
                    f"remaining_epochs={remaining_epochs}."
                )
    elif checkpoint_enabled and checkpoint_resume_if_available and os.path.isfile(latest_ckpt_model_path):
        agent.load_model(latest_ckpt_model_path)
        if checkpoint_save_optimizer and os.path.isdir(latest_ckpt_optimizer_path):
            env.load_optimizer(latest_ckpt_optimizer_path, only_weights=True)
        latest_meta = _load_checkpoint_metadata(latest_ckpt_meta_path)
        if latest_meta is not None:
            resume_start_epoch = max(0, int(latest_meta.get("epoch", 0)))
        if checkpoint_save_best:
            best_meta = _load_checkpoint_metadata(best_ckpt_meta_path)
            if best_meta is not None:
                checkpoint_state["best_metric"] = float(best_meta.get("metric_value"))
                checkpoint_state["best_epoch"] = int(best_meta.get("epoch"))
        _maybe_extend_plan_from_recent_best(resume_start_epoch)
        remaining_epochs = max(0, planned_total_epochs - resume_start_epoch)
        if remaining_epochs <= 0:
            skip_training = True
            print(
                f"[run:{run_name}] Resume checkpoint already reached epoch {resume_start_epoch}/{planned_total_epochs}. "
                "Skipping training."
            )
        else:
            env.n_epochs = int(remaining_epochs)
            print(
                f"[run:{run_name}] Resuming from latest checkpoint: epoch={resume_start_epoch}, "
                f"remaining_epochs={remaining_epochs}."
            )

    def _checkpointed_epoch_callback(epoch_info: dict[str, Any]) -> dict[str, Any]:
        nonlocal _warned_missing_val_once, planned_total_epochs
        local_epoch = int(epoch_info.get("epoch", 0))
        abs_epoch = int(resume_start_epoch + local_epoch)
        epoch_info_with_abs = dict(epoch_info)
        epoch_info_with_abs["epoch_abs"] = int(abs_epoch)
        callback_out = _combined_epoch_callback(epoch_info_with_abs)

        # Independent max-epochs extension monitor (best epoch wrt val_loss/train_loss).
        ext_val_loss = epoch_info.get("val_loss")
        ext_metric = float(ext_val_loss) if ext_val_loss is not None else float(epoch_info["train_loss"])
        ext_best = max_epochs_extension_state["best_metric"]
        if (ext_best is None) or (ext_metric < float(ext_best)):
            max_epochs_extension_state["best_metric"] = float(ext_metric)
            max_epochs_extension_state["best_epoch"] = int(abs_epoch)

        if not checkpoint_enabled:
            if (
                max_epochs_extension_enabled
                and abs_epoch >= int(planned_total_epochs)
            ):
                best_epoch = max_epochs_extension_state.get("best_epoch")
                if best_epoch is not None:
                    in_recent_window = int(best_epoch) > (
                        int(planned_total_epochs) - int(max_epochs_extension_window_epochs)
                    )
                    if in_recent_window:
                        added = _apply_dynamic_extension(int(best_epoch), reason="runtime")
                        if added > 0:
                            callback_out["extend_n_epochs_by"] = int(
                                callback_out.get("extend_n_epochs_by", 0)
                            ) + int(added)
                            if bool(callback_out.get("stop_training", False)):
                                callback_out["stop_training"] = False
            return callback_out

        metric_key = checkpoint_metric
        if metric_key == "auto":
            metric_key = "val_loss" if epoch_info.get("val_loss") is not None else "train_loss"
        if metric_key == "val_loss" and epoch_info.get("val_loss") is None:
            if not _warned_missing_val_once:
                print(
                    f"[run:{run_name}] checkpoint_metric='val_loss' but val_loss is unavailable; "
                    "falling back to train_loss."
                )
                _warned_missing_val_once = True
            metric_key = "train_loss"
        metric_value = float(epoch_info.get(metric_key, epoch_info["train_loss"]))

        should_save_latest = (abs_epoch % max(1, checkpoint_every_epochs) == 0) or bool(
            callback_out.get("stop_training", False)
        )
        if should_save_latest:
            _save_checkpoint(
                latest_ckpt_model_path,
                latest_ckpt_optimizer_path,
                latest_ckpt_meta_path,
                kind="latest",
                epoch_abs=abs_epoch,
                metric_name=metric_key,
                metric_value=metric_value,
            )
            print(
                f"[run:{run_name}] Checkpoint saved (latest): epoch={abs_epoch}, "
                f"{metric_key}={metric_value:.6f}"
            )

        if checkpoint_save_best:
            best_metric = checkpoint_state["best_metric"]
            improved = (best_metric is None) or (metric_value < float(best_metric))
            if improved:
                checkpoint_state["best_metric"] = float(metric_value)
                checkpoint_state["best_epoch"] = int(abs_epoch)
                _save_checkpoint(
                    best_ckpt_model_path,
                    best_ckpt_optimizer_path,
                    best_ckpt_meta_path,
                    kind="best",
                    epoch_abs=abs_epoch,
                    metric_name=metric_key,
                    metric_value=metric_value,
                )
                print(
                    f"[run:{run_name}] Checkpoint saved (best): epoch={abs_epoch}, "
                    f"{metric_key}={metric_value:.6f}"
                )

        if (
            max_epochs_extension_enabled
            and abs_epoch >= int(planned_total_epochs)
        ):
            best_epoch = max_epochs_extension_state.get("best_epoch")
            if best_epoch is None:
                best_epoch = checkpoint_state.get("best_epoch")
            if best_epoch is not None:
                in_recent_window = int(best_epoch) > (
                    int(planned_total_epochs) - int(max_epochs_extension_window_epochs)
                )
                if in_recent_window:
                    added = _apply_dynamic_extension(int(best_epoch), reason="runtime")
                    if added > 0:
                        callback_out["extend_n_epochs_by"] = int(
                            callback_out.get("extend_n_epochs_by", 0)
                        ) + int(added)
                        if bool(callback_out.get("stop_training", False)):
                            callback_out["stop_training"] = False

        return callback_out

    def _sampling_desc(
        name: str,
        mode_key: str,
        fixed_key: str,
        uniform_low_key: str,
        uniform_high_key: str,
        discrete_values_key: str,
        default_fixed,
    ) -> str:
        mode = str(config.get(mode_key, "fixed")).strip().lower()
        if mode == "uniform":
            lo = config.get(uniform_low_key)
            hi = config.get(uniform_high_key)
            return f"{name}=uniform[{lo},{hi}]"
        if mode == "discrete":
            values = config.get(discrete_values_key)
            n_vals = 0 if values is None else len(values)
            return f"{name}=discrete(n={n_vals})"
        fixed_val = config.get(fixed_key, default_fixed)
        return f"{name}=fixed({fixed_val})"

    instrument_model = str(config.get("instrument_model", "gbm")).strip().lower()
    sampling_bits = [
        _sampling_desc(
            name="sigma",
            mode_key="gbm_sigma_per_path_mode",
            fixed_key="sigma",
            uniform_low_key="gbm_sigma_uniform_low",
            uniform_high_key="gbm_sigma_uniform_high",
            discrete_values_key="gbm_sigma_discrete_values",
            default_fixed=config.get("sigma", calib.sigma_train),
        )
    ]
    if instrument_model in {"garch", "hmm_garch"}:
        sampling_bits.append(
            _sampling_desc(
                name="r",
                mode_key="r_per_path_mode",
                fixed_key="r",
                uniform_low_key="r_uniform_low",
                uniform_high_key="r_uniform_high",
                discrete_values_key="r_discrete_values",
                default_fixed=config.get("r", calib.r_train),
            )
        )
        sampling_bits.append(
            _sampling_desc(
                name="alpha",
                mode_key="garch_alpha_per_path_mode",
                fixed_key="garch_alpha",
                uniform_low_key="garch_alpha_uniform_low",
                uniform_high_key="garch_alpha_uniform_high",
                discrete_values_key="garch_alpha_discrete_values",
                default_fixed=config.get("garch_alpha", 0.05),
            )
        )
        sampling_bits.append(
            _sampling_desc(
                name="beta",
                mode_key="garch_beta_per_path_mode",
                fixed_key="garch_beta",
                uniform_low_key="garch_beta_uniform_low",
                uniform_high_key="garch_beta_uniform_high",
                discrete_values_key="garch_beta_discrete_values",
                default_fixed=config.get("garch_beta", 0.9),
            )
        )
        sampling_bits.append(
            _sampling_desc(
                name="leverage",
                mode_key="garch_leverage_per_path_mode",
                fixed_key="garch_leverage",
                uniform_low_key="garch_leverage_uniform_low",
                uniform_high_key="garch_leverage_uniform_high",
                discrete_values_key="garch_leverage_discrete_values",
                default_fixed=config.get("garch_leverage", 0.0),
            )
        )
        if bool(config.get("garch_use_student_t", False)):
            sampling_bits.append(
                _sampling_desc(
                    name="t_df",
                    mode_key="garch_student_t_df_per_path_mode",
                    fixed_key="garch_student_t_df",
                    uniform_low_key="garch_student_t_df_uniform_low",
                    uniform_high_key="garch_student_t_df_uniform_high",
                    discrete_values_key="garch_student_t_df_discrete_values",
                    default_fixed=config.get("garch_student_t_df", 8.0),
                )
            )
        if instrument_model == "hmm_garch":
            hmm_mode = str(config.get("hmm_params_per_path_mode", "fixed")).strip().lower()
            sampling_bits.append(f"hmm_params_mode={hmm_mode}")
            if hmm_mode == "uniform_random":
                sampling_bits.append(f"hmm_states={int(config.get('hmm_num_states', 0))}")
                sampling_bits.append(
                    f"hmm_transition_u=[{float(config.get('hmm_transition_uniform_low', 0.0)):.4g},"
                    f"{float(config.get('hmm_transition_uniform_high', 1.0)):.4g}]"
                )
                sampling_bits.append(
                    f"hmm_initial_u=[{float(config.get('hmm_initial_uniform_low', 0.0)):.4g},"
                    f"{float(config.get('hmm_initial_uniform_high', 1.0)):.4g}]"
                )
                sampling_bits.append(
                    f"hmm_mult_u=[{float(config.get('hmm_vol_multipliers_uniform_low', 0.5)):.4g},"
                    f"{float(config.get('hmm_vol_multipliers_uniform_high', 2.0)):.4g}]"
                )
                sampling_bits.append(
                    f"hmm_mult_sort={bool(config.get('hmm_vol_multipliers_sort', True))}"
                )
            else:
                tm = np.asarray(config.get("hmm_transition_matrix"), dtype=np.float64)
                mult = np.asarray(config.get("hmm_vol_multipliers"), dtype=np.float64).reshape(-1)
                sampling_bits.append(f"hmm_matrix_states={int(tm.shape[0])}")
                sampling_bits.append(
                    f"hmm_mult_range=[{float(np.min(mult)):.4g},{float(np.max(mult)):.4g}]"
                )

    print(
        f"[run:{run_name}] Training agent={config['agent_name']} claim={config['contingent_claim']} "
        f"instrument={instrument_model} with calibrated r={calib.r_train:.6f}, sigma={calib.sigma_train:.6f}. "
        f"Pathwise sampling: {'; '.join(sampling_bits)}"
    )
    t0 = time.perf_counter()
    if skip_training:
        out = ([], []) if int(config["val_paths"]) > 0 else []
    else:
        out = env.train(
            train_paths=int(config["train_paths"]),
            val_paths=int(config["val_paths"]),
            random_seed=seed,
            epoch_end_callback=_checkpointed_epoch_callback,
        )
    elapsed = time.perf_counter() - t0
    print(f"[run:{run_name}] Training finished in {elapsed:.2f}s")

    if int(config["val_paths"]) > 0:
        train_losses, val_losses = out
        epoch_start = int(resume_start_epoch) + 1
        hist_df = pd.DataFrame(
            {
                "epoch": list(range(epoch_start, epoch_start + len(train_losses))),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }
        )
    else:
        train_losses = out
        epoch_start = int(resume_start_epoch) + 1
        hist_df = pd.DataFrame(
            {
                "epoch": list(range(epoch_start, epoch_start + len(train_losses))),
                "train_loss": train_losses,
            }
        )

    hist_path = os.path.join(dirs["tables_dir"], "training_history.csv")
    if hist_df.empty:
        if os.path.isfile(hist_path):
            print(
                f"[run:{run_name}] No new epochs were executed; keeping existing training history: {hist_path}"
            )
        else:
            hist_df.to_csv(hist_path, index=False)
            print(f"[run:{run_name}] Saved empty training history header: {hist_path}")
    else:
        if os.path.isfile(hist_path):
            try:
                prev_hist_df = pd.read_csv(hist_path)
            except Exception:
                prev_hist_df = pd.DataFrame()
            merged_hist_df = pd.concat([prev_hist_df, hist_df], ignore_index=True, sort=False)
            if "epoch" in merged_hist_df.columns:
                merged_hist_df = (
                    merged_hist_df.drop_duplicates(subset=["epoch"], keep="last")
                    .sort_values("epoch")
                    .reset_index(drop=True)
                )
            merged_hist_df.to_csv(hist_path, index=False)
        else:
            hist_df.to_csv(hist_path, index=False)
        print(f"[run:{run_name}] Saved training history (append-aware): {hist_path}")

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
                "r_per_path_mode": str(config.get("r_per_path_mode", "fixed")),
                "r_uniform_low": config.get("r_uniform_low"),
                "r_uniform_high": config.get("r_uniform_high"),
                "r_discrete_values": json.dumps(config.get("r_discrete_values")),
                "r_discrete_probs": json.dumps(config.get("r_discrete_probs")),
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
                "garch_alpha_per_path_mode": str(config.get("garch_alpha_per_path_mode", "fixed")),
                "garch_alpha_uniform_low": config.get("garch_alpha_uniform_low"),
                "garch_alpha_uniform_high": config.get("garch_alpha_uniform_high"),
                "garch_alpha_discrete_values": json.dumps(config.get("garch_alpha_discrete_values")),
                "garch_alpha_discrete_probs": json.dumps(config.get("garch_alpha_discrete_probs")),
                "garch_beta_per_path_mode": str(config.get("garch_beta_per_path_mode", "fixed")),
                "garch_beta_uniform_low": config.get("garch_beta_uniform_low"),
                "garch_beta_uniform_high": config.get("garch_beta_uniform_high"),
                "garch_beta_discrete_values": json.dumps(config.get("garch_beta_discrete_values")),
                "garch_beta_discrete_probs": json.dumps(config.get("garch_beta_discrete_probs")),
                "garch_leverage_per_path_mode": str(config.get("garch_leverage_per_path_mode", "fixed")),
                "garch_leverage_uniform_low": config.get("garch_leverage_uniform_low"),
                "garch_leverage_uniform_high": config.get("garch_leverage_uniform_high"),
                "garch_leverage_discrete_values": json.dumps(config.get("garch_leverage_discrete_values")),
                "garch_leverage_discrete_probs": json.dumps(config.get("garch_leverage_discrete_probs")),
                "garch_student_t_df_per_path_mode": str(config.get("garch_student_t_df_per_path_mode", "fixed")),
                "garch_student_t_df_uniform_low": config.get("garch_student_t_df_uniform_low"),
                "garch_student_t_df_uniform_high": config.get("garch_student_t_df_uniform_high"),
                "garch_student_t_df_discrete_values": json.dumps(config.get("garch_student_t_df_discrete_values")),
                "garch_student_t_df_discrete_probs": json.dumps(config.get("garch_student_t_df_discrete_probs")),
                "hmm_transition_matrix": json.dumps(config.get("hmm_transition_matrix")),
                "hmm_initial_distribution": json.dumps(config.get("hmm_initial_distribution")),
                "hmm_vol_multipliers": json.dumps(config.get("hmm_vol_multipliers")),
                "hmm_params_per_path_mode": str(config.get("hmm_params_per_path_mode", "fixed")),
                "hmm_num_states": config.get("hmm_num_states"),
                "hmm_transition_uniform_low": config.get("hmm_transition_uniform_low"),
                "hmm_transition_uniform_high": config.get("hmm_transition_uniform_high"),
                "hmm_initial_uniform_low": config.get("hmm_initial_uniform_low"),
                "hmm_initial_uniform_high": config.get("hmm_initial_uniform_high"),
                "hmm_vol_multipliers_uniform_low": config.get("hmm_vol_multipliers_uniform_low"),
                "hmm_vol_multipliers_uniform_high": config.get("hmm_vol_multipliers_uniform_high"),
                "hmm_vol_multipliers_sort": config.get("hmm_vol_multipliers_sort"),
                "student_t_df": config.get("student_t_df"),
                "student_t_df_per_path_mode": str(config.get("student_t_df_per_path_mode", "fixed")),
                "student_t_df_uniform_low": config.get("student_t_df_uniform_low"),
                "student_t_df_uniform_high": config.get("student_t_df_uniform_high"),
                "student_t_df_discrete_values": json.dumps(config.get("student_t_df_discrete_values")),
                "student_t_df_discrete_probs": json.dumps(config.get("student_t_df_discrete_probs")),
                "learning_rate_strategy": str(config.get("learning_rate_strategy", "constant")),
                "early_stopping_enabled": bool(config.get("early_stopping_enabled", False)),
                "early_stopping_patience": int(config.get("early_stopping_patience", 20)),
                "early_stopping_min_delta": float(config.get("early_stopping_min_delta", 1e-4)),
                "max_epochs_extension_enabled": bool(
                    config.get("max_epochs_extension_enabled", False)
                ),
                "max_epochs_extension_window_epochs": int(
                    config.get("max_epochs_extension_window_epochs", 10)
                ),
                "max_epochs_extension_by": int(
                    config.get("max_epochs_extension_by", 10)
                ),
                "max_added_epochs": int(
                    config.get("max_added_epochs", 100)
                ),
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
        "max_epochs_extension_enabled": bool(config.get("max_epochs_extension_enabled", False)),
        "max_epochs_extension_window_epochs": int(config.get("max_epochs_extension_window_epochs", 10)),
        "max_epochs_extension_by": int(config.get("max_epochs_extension_by", 10)),
        "max_added_epochs": int(config.get("max_added_epochs", 100)),
        "planned_total_epochs_final": int(planned_total_epochs),
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
