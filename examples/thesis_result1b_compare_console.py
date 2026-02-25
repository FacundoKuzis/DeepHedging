"""
Console runner for thesis Result 1b comparisons.

Bloque 1b:
- Benchmark + agentes entrenados en GBM calibrado
- Evaluacion sobre datos reales (ventanas historicas) o simulados
- Metricas de cola y bootstrap defendibles
"""

import argparse
import json
import os
import hashlib
import re
import sys
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.thesis_result1_common import (  # noqa: E402
    THESIS_MODELS_ROOT,
    TRAINABLE_AGENT_NAMES,
    bootstrap_statistics_list,
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
from DeepHedging.RiskMeasures import CVaR, MAE, WorstCase  # noqa: E402
from DeepHedging.utils.gbm_calibration import (  # noqa: E402
    IRX_TICKER,
    VIX_TICKER,
    calibrate_gbm_from_market_data,
    load_external_series,
    map_option_market_implied_vol_to_window_start,
    map_historical_sigma_to_window_start,
    map_series_to_window_start,
)
from DeepHedging.utils.historical_windows import (  # noqa: E402
    build_historical_windows_from_csv,
    to_environment_paths,
)
from DeepHedging.utils.garch_context_sigma import (  # noqa: E402
    estimate_pathwise_garch_sigma_from_context,
)


RESULT1B_ROOT = THESIS_MODELS_ROOT
ACTIONS_CACHE_SCHEMA_VERSION = 3



def _run_dirs(run_name: str, config_path: str) -> dict[str, str]:
    _ = run_name
    rel_stem = get_config_relative_stem(config_path)
    run_dir = os.path.normpath(os.path.join(RESULT1B_ROOT, rel_stem))
    plots_dir = os.path.join(run_dir, "plots")
    tables_dir = os.path.join(run_dir, "tables")
    logs_dir = os.path.join(run_dir, "logs")
    market_cache_dir = os.path.join(RESULT1B_ROOT, "market_data_cache")
    models_root = RESULT1B_ROOT
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(market_cache_dir, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    return {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "tables_dir": tables_dir,
        "logs_dir": logs_dir,
        "market_cache_dir": market_cache_dir,
        "models_root": models_root,
    }



def required_keys() -> set[str]:
    return {
        "schema_version",
        "model_family",
        "run_name",
        "description",
        "output_root",
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
        "benchmark_agent_name",
        "benchmark_bump_size",
        "benchmark_num_simulations",
        "benchmark_seed",
        "trained_agents",
        "models_dir",
        "optimizers_dir",
        "eval_paths",
        "eval_seed",
        "pricing_method",
        "plot_min_x",
        "plot_max_x",
        "language",
        "bootstrap_enabled",
        "bootstrap_n_bootstraps",
        "bootstrap_confidence_level",
        "bootstrap_batch_size",
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "interval",
        "price_col",
        "download_if_missing",
        "sigma_source",
        "implied_vol_source",
        "implied_vol_stat",
        "fixed_implied_vol",
        "risk_free_source",
        "fixed_risk_free",
        "risk_free_mode",
        "test_data_mode",
        "historical_stride",
        "max_test_windows",
        "bootstrap_method",
        "moving_block_size",
    }



def optional_keys() -> set[str]:
    return {
        "instrument_model",
        "price_computation_mode",
        "path_transformation_type",
        "include_log_strike_feature",
        "sequence_output_mode",
        "position_activation",
        "garch_alpha",
        "garch_beta",
        "garch_omega",
        "garch_leverage",
        "garch_use_student_t",
        "garch_student_t_df",
        "history_conv1d_enabled",
        "history_conv1d_layers",
        "history_conv1d_pooling",
        "gbm_sigma_per_path_mode",
        "gbm_sigma_uniform_low",
        "gbm_sigma_uniform_high",
        "gbm_sigma_discrete_values",
        "gbm_sigma_discrete_probs",
        "benchmark_no_trade_band",
        "no_intervention_bound",
        "benchmark_no_intervention_bound",
        "no_intervention_mode",
        "benchmark_no_intervention_mode",
        "eval_agent_batch_size",
        "terminal_progress_log_every_agent_batches",
        "benchmark_mc_state_chunk_size",
        "benchmark_mc_seed_mode",
        "benchmark_mc_parallel_enabled",
        "benchmark_mc_n_workers",
        "benchmark_mc_parallel_backend",
        "benchmark_mc_parallel_chunk_size",
        "benchmark_mc_parallel_min_states",
        "reuse_actions_between_steps",
        "sigma_mode",
        "historical_sigma_window_days",
        "use_price_history_context",
        "context_length",
        "context_feature_mode",
        "context_pre_ttm_mode",
        "context_for_path_generation_only",
        "option_quotes_csv",
        "option_quote_date_col",
        "option_expiry_col",
        "option_strike_col",
        "option_type_col",
        "option_bid_col",
        "option_ask_col",
        "option_last_col",
        "implied_option_type",
        "calendar_days_per_year",
        "option_max_quote_lag_days",
        "option_max_expiry_diff_days",
        "benchmark_lrm_provider",
        "benchmark_lrm_outer_paths",
        "benchmark_lrm_var_epsilon",
        "benchmark_lrm_use_antithetic",
        "benchmark_lrm_seed_mode",
        "benchmark_lrm_mc_inner_paths",
        "benchmark_lrm_mc_inner_chunk_size",
        "benchmark_lrm_mc_parallel_enabled",
        "benchmark_lrm_mc_n_workers",
        "benchmark_lrm_mc_parallel_backend",
        "benchmark_lrm_mc_parallel_chunk_size",
        "benchmark_lrm_lsm_train_paths",
        "benchmark_lrm_lsm_ridge_alpha",
        "benchmark_lrm_lsm_feature_set",
        "benchmark_lrm_lsm_poly_degree",
        "benchmark_lrm_lsm_use_cache",
        "benchmark_lrm_lsm_cache_dir",
        "benchmark_lrm_lsm_cache_key",
        "benchmark_lrm_lsm_force_rebuild",
        "benchmark_lrm_verbose",
        "benchmark_lrm_log_every_t",
        "benchmark_lrm_mc_log_every_chunks",
        "benchmark_delta_sigma_mode",
        "benchmark_delta_sigma_context_days",
        "benchmark_delta_sigma_min_obs",
        "benchmark_delta_sigma_garch_alpha",
        "benchmark_delta_sigma_garch_beta",
        "benchmark_delta_sigma_garch_leverage",
        "benchmark_delta_sigma_garch_omega",
        "benchmark_delta_sigma_floor",
        "benchmark_delta_sigma_cap",
        "benchmark_delta_sigma_default",
        "benchmark_agents_to_compare",
    }



def validate_config(config: dict[str, Any]) -> None:
    strict_validate_keys(config, required_keys(), optional_keys())

    if int(config["schema_version"]) != 1:
        raise ValueError("schema_version must be 1.")
    if str(config["model_family"]).strip().lower() != "deep_hedging_result_1b":
        raise ValueError("model_family must be 'deep_hedging_result_1b'.")

    validate_agent_name(str(config["benchmark_agent_name"]))
    if str(config["benchmark_agent_name"]) in TRAINABLE_AGENT_NAMES:
        raise ValueError("benchmark_agent_name must be non-trainable benchmark.")

    for key in ["run_name", "description", "ticker", "train_start_date", "train_end_date", "test_start_date", "test_end_date", "interval", "price_col", "models_dir", "optimizers_dir"]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be non-empty string")

    if int(config["n"]) <= 0:
        raise ValueError("n must be > 0")
    if int(config["trading_days_per_year"]) <= 0:
        raise ValueError("trading_days_per_year must be > 0")
    if float(config["s0"]) <= 0.0:
        raise ValueError("s0 must be > 0")
    if float(config["strike"]) <= 0.0:
        raise ValueError("strike must be > 0")

    instrument_model = str(config.get("instrument_model", "gbm")).strip().lower()
    if instrument_model not in {"gbm", "garch"}:
        raise ValueError("instrument_model must be one of {'gbm','garch'}.")

    if int(config["eval_paths"]) <= 0:
        raise ValueError("eval_paths must be > 0")
    if int(config["eval_seed"]) < 0:
        raise ValueError("eval_seed must be >= 0")

    if float(config["plot_min_x"]) >= float(config["plot_max_x"]):
        raise ValueError("plot_min_x must be < plot_max_x")

    if str(config["pricing_method"]).strip().lower() not in {"fixed", "individual"}:
        raise ValueError("pricing_method must be 'fixed' or 'individual'.")
    price_mode = str(config.get("price_computation_mode", "pathwise_if_available")).strip().lower()
    if price_mode == "pathwise":
        price_mode = "pathwise_if_available"
    if price_mode not in {"pathwise_if_available", "scalar"}:
        raise ValueError("price_computation_mode must be 'pathwise_if_available' or 'scalar'.")

    if str(config["language"]).strip().lower() not in {"es", "en"}:
        raise ValueError("language must be 'es' or 'en'.")

    if "include_log_strike_feature" in config and not isinstance(config["include_log_strike_feature"], bool):
        raise ValueError("include_log_strike_feature must be bool when provided.")
    path_t = str(config.get("path_transformation_type", "log_moneyness")).strip().lower()
    if path_t not in {"none", "log", "log_moneyness"}:
        raise ValueError("path_transformation_type must be one of {'none','log','log_moneyness'}.")
    if "sequence_output_mode" in config:
        seq_out = str(config["sequence_output_mode"]).strip().lower()
        if seq_out not in {"trade", "position"}:
            raise ValueError("sequence_output_mode must be 'trade' or 'position'.")
    if "position_activation" in config:
        pos_act = str(config["position_activation"]).strip().lower()
        if pos_act not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("position_activation must be one of {'linear','sigmoid','tanh'}.")

    if not isinstance(config["bootstrap_enabled"], bool):
        raise ValueError("bootstrap_enabled must be bool.")
    if int(config["bootstrap_n_bootstraps"]) <= 0:
        raise ValueError("bootstrap_n_bootstraps must be > 0.")
    if not (0.0 < float(config["bootstrap_confidence_level"]) < 1.0):
        raise ValueError("bootstrap_confidence_level must satisfy 0<c<1")
    if int(config["bootstrap_batch_size"]) <= 0:
        raise ValueError("bootstrap_batch_size must be > 0")

    if str(config["sigma_source"]).strip().lower() not in {"historical", "implied"}:
        raise ValueError("sigma_source must be 'historical' or 'implied'.")
    if str(config["implied_vol_source"]).strip().lower() not in {"vix", "fixed", "option_market"}:
        raise ValueError("implied_vol_source must be 'vix', 'fixed' or 'option_market'.")
    if str(config["implied_vol_stat"]).strip().lower() not in {"mean", "median"}:
        raise ValueError("implied_vol_stat must be 'mean' or 'median'.")
    if str(config["risk_free_source"]).strip().lower() not in {"irx", "fixed"}:
        raise ValueError("risk_free_source must be 'irx' or 'fixed'.")
    if str(config["risk_free_mode"]).strip().lower() not in {"train_average", "per_window_start"}:
        raise ValueError("risk_free_mode must be 'train_average' or 'per_window_start'.")

    sigma_mode = str(config.get("sigma_mode", "train_average")).strip().lower()
    if sigma_mode not in {"train_average", "per_window_start", "rolling_pre_window"}:
        raise ValueError(
            "sigma_mode must be one of {'train_average','per_window_start','rolling_pre_window'}."
        )
    sigma_source = str(config["sigma_source"]).strip().lower()
    if sigma_mode == "rolling_pre_window" and sigma_source != "historical":
        raise ValueError("sigma_mode='rolling_pre_window' is only valid with sigma_source='historical'.")
    if sigma_mode == "per_window_start" and sigma_source == "historical":
        raise ValueError(
            "sigma_mode='per_window_start' with sigma_source='historical' is not supported. "
            "Use sigma_mode='rolling_pre_window' or 'train_average'."
        )
    if sigma_mode == "per_window_start" and str(config["sigma_source"]).strip().lower() == "implied":
        implied_source = str(config["implied_vol_source"]).strip().lower()
        if implied_source not in {"vix", "fixed", "option_market"}:
            raise ValueError("implied_vol_source must be 'vix', 'fixed' or 'option_market' for implied per-window sigma.")
        if implied_source == "option_market":
            required_option_fields = [
                "option_quotes_csv",
                "option_quote_date_col",
                "option_expiry_col",
                "option_strike_col",
                "option_type_col",
                "option_bid_col",
                "option_ask_col",
                "option_last_col",
                "implied_option_type",
                "calendar_days_per_year",
                "option_max_quote_lag_days",
                "option_max_expiry_diff_days",
            ]
            missing = [k for k in required_option_fields if k not in config or config[k] is None]
            if missing:
                raise ValueError(
                    "implied_vol_source='option_market' requires fields: "
                    f"{missing}"
                )
            option_csv = os.path.abspath(os.path.expanduser(str(config["option_quotes_csv"])))
            if not os.path.isfile(option_csv):
                raise FileNotFoundError(f"option_quotes_csv not found: {option_csv}")
            if str(config["implied_option_type"]).strip().lower() not in {"call", "put"}:
                raise ValueError("implied_option_type must be 'call' or 'put'.")
            if float(config["calendar_days_per_year"]) <= 0.0:
                raise ValueError("calendar_days_per_year must be > 0.")
            if int(config["option_max_quote_lag_days"]) < 0:
                raise ValueError("option_max_quote_lag_days must be >= 0.")
            if int(config["option_max_expiry_diff_days"]) < 0:
                raise ValueError("option_max_expiry_diff_days must be >= 0.")
            if str(config["test_data_mode"]).strip().lower() != "historical_windows":
                raise ValueError(
                    "implied_vol_source='option_market' currently requires test_data_mode='historical_windows'."
                )

    delta_sigma_mode = str(config.get("benchmark_delta_sigma_mode", "none")).strip().lower()
    if delta_sigma_mode not in {"none", "garch_context_static", "garch_context_stepwise"}:
        raise ValueError(
            "benchmark_delta_sigma_mode must be one of {'none','garch_context_static','garch_context_stepwise'}."
        )
    if delta_sigma_mode != "none":
        if str(config["benchmark_agent_name"]).strip() != "DeltaHedgingAgent":
            raise ValueError(
                "benchmark_delta_sigma_mode != 'none' requires benchmark_agent_name='DeltaHedgingAgent'."
            )
        context_days = int(config.get("benchmark_delta_sigma_context_days", 50))
        if context_days <= 0:
            raise ValueError("benchmark_delta_sigma_context_days must be > 0.")
        min_obs = int(config.get("benchmark_delta_sigma_min_obs", 10))
        if min_obs < 1:
            raise ValueError("benchmark_delta_sigma_min_obs must be >= 1.")
        alpha = _resolve_float_override(config, "benchmark_delta_sigma_garch_alpha", config.get("garch_alpha", 0.05))
        beta = _resolve_float_override(config, "benchmark_delta_sigma_garch_beta", config.get("garch_beta", 0.9))
        leverage = _resolve_float_override(config, "benchmark_delta_sigma_garch_leverage", config.get("garch_leverage", 0.0))
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("benchmark_delta_sigma_garch_alpha/beta must be >= 0.")
        stability_lhs = alpha + beta + 2.0 * leverage
        if stability_lhs >= 1.0:
            raise ValueError(
                "Require benchmark_delta_sigma_garch_alpha + benchmark_delta_sigma_garch_beta + "
                "2*benchmark_delta_sigma_garch_leverage < 1 for stationarity. "
                f"Got {stability_lhs:.6f}."
            )
        if "benchmark_delta_sigma_garch_omega" in config and config["benchmark_delta_sigma_garch_omega"] is not None:
            if float(config["benchmark_delta_sigma_garch_omega"]) <= 0.0:
                raise ValueError("benchmark_delta_sigma_garch_omega must be > 0 when provided.")
        sigma_floor = float(config.get("benchmark_delta_sigma_floor", 1e-6))
        if sigma_floor <= 0.0:
            raise ValueError("benchmark_delta_sigma_floor must be > 0.")
        if "benchmark_delta_sigma_cap" in config and config["benchmark_delta_sigma_cap"] is not None:
            sigma_cap = float(config["benchmark_delta_sigma_cap"])
            if sigma_cap <= sigma_floor:
                raise ValueError("benchmark_delta_sigma_cap must be > benchmark_delta_sigma_floor.")
        if "benchmark_delta_sigma_default" in config and config["benchmark_delta_sigma_default"] is not None:
            if float(config["benchmark_delta_sigma_default"]) <= 0.0:
                raise ValueError("benchmark_delta_sigma_default must be > 0 when provided.")

    if "historical_sigma_window_days" in config and config["historical_sigma_window_days"] is not None:
        if int(config["historical_sigma_window_days"]) < 2:
            raise ValueError("historical_sigma_window_days must be >= 2 when provided.")

    if str(config["test_data_mode"]).strip().lower() not in {"historical_windows", "simulated"}:
        raise ValueError("test_data_mode must be 'historical_windows' or 'simulated'.")

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
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("garch_alpha and garch_beta must be >= 0.")
        if alpha + beta >= 1.0:
            raise ValueError("Require garch_alpha + garch_beta < 1 for stability.")
        leverage = float(config.get("garch_leverage", 0.0))
        stability_lhs = alpha + beta + 2.0 * leverage
        if stability_lhs >= 1.0:
            raise ValueError(
                "Require garch_alpha + garch_beta + 2*garch_leverage < 1 for stationarity. "
                f"Got {stability_lhs:.6f}."
            )
        if config.get("garch_omega", None) is not None and float(config.get("garch_omega")) <= 0.0:
            raise ValueError("garch_omega must be > 0 when provided.")
        if bool(config.get("garch_use_student_t", False)):
            df = float(config.get("garch_student_t_df", 8.0))
            if df <= 2.0:
                raise ValueError("garch_student_t_df must be > 2 when garch_use_student_t=true.")

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

    if int(config["historical_stride"]) <= 0:
        raise ValueError("historical_stride must be > 0")
    if config["max_test_windows"] is not None and int(config["max_test_windows"]) <= 0:
        raise ValueError("max_test_windows must be null or > 0")

    if str(config["bootstrap_method"]).strip().lower() not in {"iid", "moving_block"}:
        raise ValueError("bootstrap_method must be 'iid' or 'moving_block'.")
    if config["moving_block_size"] is not None and int(config["moving_block_size"]) <= 0:
        raise ValueError("moving_block_size must be null or > 0")

    if not isinstance(config["download_if_missing"], bool):
        raise ValueError("download_if_missing must be bool")

    if "benchmark_lrm_provider" in config:
        provider = str(config["benchmark_lrm_provider"]).strip().lower()
        if provider not in {"bs_closed_form", "monte_carlo", "lsm", "lsmc", "asian_monte_carlo", "asian_lsmc"}:
            raise ValueError(
                "benchmark_lrm_provider must be one of "
                "{'bs_closed_form','monte_carlo','lsm','lsmc','asian_monte_carlo','asian_lsmc'}."
            )
        claim_name = str(config["contingent_claim"]).strip()
        if provider == "bs_closed_form" and claim_name not in {"EuropeanCall", "EuropeanPut"}:
            raise ValueError("benchmark_lrm_provider='bs_closed_form' supports only EuropeanCall/EuropeanPut.")
        if provider in {"asian_monte_carlo", "asian_lsmc"} and not claim_name.startswith("Asian"):
            raise ValueError("benchmark_lrm_provider for Asian requires an Asian contingent_claim.")
    if "benchmark_lrm_outer_paths" in config and int(config["benchmark_lrm_outer_paths"]) <= 1:
        raise ValueError("benchmark_lrm_outer_paths must be > 1.")
    if "benchmark_lrm_var_epsilon" in config and float(config["benchmark_lrm_var_epsilon"]) <= 0.0:
        raise ValueError("benchmark_lrm_var_epsilon must be > 0.")
    if "benchmark_lrm_use_antithetic" in config and not isinstance(config["benchmark_lrm_use_antithetic"], bool):
        raise ValueError("benchmark_lrm_use_antithetic must be bool.")
    if "benchmark_lrm_seed_mode" in config:
        mode = str(config["benchmark_lrm_seed_mode"]).strip().lower()
        if mode not in {"shared_crn", "per_state"}:
            raise ValueError("benchmark_lrm_seed_mode must be 'shared_crn' or 'per_state'.")
    if "benchmark_lrm_mc_inner_paths" in config and int(config["benchmark_lrm_mc_inner_paths"]) <= 1:
        raise ValueError("benchmark_lrm_mc_inner_paths must be > 1.")
    if "benchmark_lrm_mc_inner_chunk_size" in config and int(config["benchmark_lrm_mc_inner_chunk_size"]) <= 0:
        raise ValueError("benchmark_lrm_mc_inner_chunk_size must be > 0.")
    if "benchmark_lrm_mc_parallel_enabled" in config and not isinstance(config["benchmark_lrm_mc_parallel_enabled"], bool):
        raise ValueError("benchmark_lrm_mc_parallel_enabled must be bool.")
    if "benchmark_lrm_mc_n_workers" in config and int(config["benchmark_lrm_mc_n_workers"]) <= 0:
        raise ValueError("benchmark_lrm_mc_n_workers must be > 0.")
    if "benchmark_lrm_mc_parallel_backend" in config:
        backend = str(config["benchmark_lrm_mc_parallel_backend"]).strip().lower()
        if backend not in {"thread", "process"}:
            raise ValueError("benchmark_lrm_mc_parallel_backend must be 'thread' or 'process'.")
    if "benchmark_lrm_mc_parallel_chunk_size" in config and config["benchmark_lrm_mc_parallel_chunk_size"] is not None:
        if int(config["benchmark_lrm_mc_parallel_chunk_size"]) <= 0:
            raise ValueError("benchmark_lrm_mc_parallel_chunk_size must be > 0 when provided.")
    if "benchmark_lrm_lsm_train_paths" in config and int(config["benchmark_lrm_lsm_train_paths"]) <= 10:
        raise ValueError("benchmark_lrm_lsm_train_paths must be > 10.")
    if "benchmark_lrm_lsm_ridge_alpha" in config and float(config["benchmark_lrm_lsm_ridge_alpha"]) <= 0.0:
        raise ValueError("benchmark_lrm_lsm_ridge_alpha must be > 0.")
    if "benchmark_lrm_lsm_feature_set" in config:
        feature_set = str(config["benchmark_lrm_lsm_feature_set"]).strip().lower()
        if feature_set not in {"minimal", "default"}:
            raise ValueError("benchmark_lrm_lsm_feature_set must be 'minimal' or 'default'.")
    if "benchmark_lrm_lsm_poly_degree" in config and int(config["benchmark_lrm_lsm_poly_degree"]) <= 0:
        raise ValueError("benchmark_lrm_lsm_poly_degree must be > 0.")
    if "benchmark_lrm_lsm_use_cache" in config and not isinstance(config["benchmark_lrm_lsm_use_cache"], bool):
        raise ValueError("benchmark_lrm_lsm_use_cache must be bool.")
    if "benchmark_lrm_lsm_force_rebuild" in config and not isinstance(config["benchmark_lrm_lsm_force_rebuild"], bool):
        raise ValueError("benchmark_lrm_lsm_force_rebuild must be bool.")
    if "benchmark_lrm_lsm_cache_dir" in config and config["benchmark_lrm_lsm_cache_dir"] is not None:
        if not isinstance(config["benchmark_lrm_lsm_cache_dir"], str) or not str(config["benchmark_lrm_lsm_cache_dir"]).strip():
            raise ValueError("benchmark_lrm_lsm_cache_dir must be null or non-empty string.")
    if "benchmark_lrm_lsm_cache_key" in config and config["benchmark_lrm_lsm_cache_key"] is not None:
        if not isinstance(config["benchmark_lrm_lsm_cache_key"], str) or not str(config["benchmark_lrm_lsm_cache_key"]).strip():
            raise ValueError("benchmark_lrm_lsm_cache_key must be null or non-empty string.")
    if "benchmark_lrm_verbose" in config and not isinstance(config["benchmark_lrm_verbose"], bool):
        raise ValueError("benchmark_lrm_verbose must be bool.")
    if "benchmark_lrm_log_every_t" in config and int(config["benchmark_lrm_log_every_t"]) <= 0:
        raise ValueError("benchmark_lrm_log_every_t must be > 0.")
    if "benchmark_lrm_mc_log_every_chunks" in config and int(config["benchmark_lrm_mc_log_every_chunks"]) < 0:
        raise ValueError("benchmark_lrm_mc_log_every_chunks must be >= 0.")

    if not isinstance(config["trained_agents"], list):
        raise ValueError("trained_agents must be a list")
    for i, item in enumerate(config["trained_agents"]):
        if not isinstance(item, dict) or set(item.keys()) != {"agent_name", "model_name"}:
            raise ValueError(f"trained_agents[{i}] must contain exactly keys ['agent_name','model_name']")
        name = str(item["agent_name"])
        validate_agent_name(name)
        if name not in TRAINABLE_AGENT_NAMES:
            raise ValueError(f"trained_agents[{i}].agent_name must be trainable")
        if not isinstance(item["model_name"], str) or not item["model_name"].strip():
            raise ValueError(f"trained_agents[{i}].model_name must be non-empty string")

    benchmark_agents_to_compare = config.get("benchmark_agents_to_compare", [])
    if benchmark_agents_to_compare is None:
        benchmark_agents_to_compare = []
    if not isinstance(benchmark_agents_to_compare, list):
        raise ValueError("benchmark_agents_to_compare must be a list when provided.")
    for i, item in enumerate(benchmark_agents_to_compare):
        if not isinstance(item, dict):
            raise ValueError(f"benchmark_agents_to_compare[{i}] must be an object.")
        allowed = {"agent_name", "label", "config_overrides"}
        extra = set(item.keys()) - allowed
        if extra:
            raise ValueError(
                f"benchmark_agents_to_compare[{i}] has unknown keys: {sorted(extra)}. "
                f"Allowed keys: {sorted(allowed)}"
            )
        if "agent_name" not in item:
            raise ValueError(f"benchmark_agents_to_compare[{i}] must include 'agent_name'.")
        name = str(item["agent_name"]).strip()
        validate_agent_name(name)
        if name in TRAINABLE_AGENT_NAMES:
            raise ValueError(
                f"benchmark_agents_to_compare[{i}].agent_name must be non-trainable benchmark. Got {name}."
            )
        if "label" in item and (not isinstance(item["label"], str) or not item["label"].strip()):
            raise ValueError(f"benchmark_agents_to_compare[{i}].label must be a non-empty string when provided.")
        if "config_overrides" in item and not isinstance(item["config_overrides"], dict):
            raise ValueError(f"benchmark_agents_to_compare[{i}].config_overrides must be an object when provided.")

    if len(config["trained_agents"]) == 0 and len(benchmark_agents_to_compare) == 0:
        raise ValueError("Provide at least one comparison target: trained_agents and/or benchmark_agents_to_compare.")



def _display_name(agent, language: str) -> str:
    plot_name = getattr(agent, "plot_name", None)
    if isinstance(plot_name, dict):
        return str(plot_name.get(language, agent.name))
    if isinstance(plot_name, str):
        return plot_name
    return str(getattr(agent, "name", "unknown_agent"))



def _slugify_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "agent"


def _model_path(agent, model_name: str) -> str:
    target_name = f"{model_name}.keras"
    matches: list[str] = []
    for root, _, files in os.walk(RESULT1B_ROOT):
        if target_name not in files:
            continue
        full = os.path.join(root, target_name)
        parent = os.path.basename(os.path.dirname(full))
        if parent == str(agent.name):
            matches.append(full)
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Model not found for trained agent {agent.name} and model_name='{model_name}' under {RESULT1B_ROOT}."
        )
    rels = [os.path.relpath(p, RESULT1B_ROOT) for p in matches]
    raise FileNotFoundError(
        f"Multiple model matches found for agent={agent.name}, model_name='{model_name}'. "
        f"Use unique model_name. Matches: {rels}"
    )



def _compute_empirical_error_metrics(errors: np.ndarray) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot compute metrics on empty error array.")
    out = {
        "mean_error": float(np.mean(arr)),
        "std_error": float(np.std(arr, ddof=0)),
        "mae_error": float(np.mean(np.abs(arr))),
        "mse_error": float(np.mean(arr**2)),
        "worst_case": float(np.min(arr)),
        "var_95": float(np.quantile(arr, 0.05)),
        "var_99": float(np.quantile(arr, 0.01)),
        "q_1pct": float(np.quantile(arr, 0.01)),
        "q_0_5pct": float(np.quantile(arr, 0.005)),
        "q_0_1pct": float(np.quantile(arr, 0.001)),
    }
    tail95 = arr[arr <= out["var_95"]]
    tail99 = arr[arr <= out["var_99"]]
    out["es_95"] = float(np.mean(tail95)) if tail95.size > 0 else out["var_95"]
    out["es_99"] = float(np.mean(tail99)) if tail99.size > 0 else out["var_99"]
    return out


def _paths_signature(paths_3d: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(paths_3d, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"paths_3d must be rank-3. Got shape={arr.shape}")
    n = int(arr.shape[0])
    sample_n = min(32, n)
    sample = arr[:sample_n, :, :]
    digest = hashlib.sha256(sample.tobytes()).hexdigest()
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        "sample_n": int(sample_n),
        "sample_sha256": digest,
        "sample_mean": float(np.mean(sample)),
        "sample_std": float(np.std(sample)),
    }


def _vector_signature(values_1d: np.ndarray | None) -> dict[str, Any] | None:
    if values_1d is None:
        return None
    arr = np.asarray(values_1d, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"length": 0, "sample_sha256": None, "sample_mean": None, "sample_std": None}
    sample_n = min(256, int(arr.size))
    sample = arr[:sample_n]
    digest = hashlib.sha256(sample.tobytes()).hexdigest()
    return {
        "length": int(arr.size),
        "sample_n": int(sample_n),
        "sample_sha256": digest,
        "sample_mean": float(np.mean(sample)),
        "sample_std": float(np.std(sample)),
    }


def _ensure_actions_cache_consistency(
    actions_cache_dir: str,
    cache_manifest: dict[str, Any],
    run_name: str,
) -> None:
    os.makedirs(actions_cache_dir, exist_ok=True)
    manifest_path = os.path.join(actions_cache_dir, "cache_manifest.json")
    current_blob = json.dumps(cache_manifest, sort_keys=True)
    existing_blob = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_blob = json.dumps(existing, sort_keys=True)
        except Exception:
            existing_blob = None

    actions_files = [fname for fname in os.listdir(actions_cache_dir) if fname.endswith("_actions.npy")]
    needs_invalidation = False
    reason = None

    if existing_blob is None and actions_files:
        needs_invalidation = True
        reason = "missing manifest with existing cached actions"
    elif existing_blob is not None and existing_blob != current_blob:
        needs_invalidation = True
        reason = "manifest mismatch"

    if needs_invalidation:
        removed = 0
        for fname in actions_files:
            try:
                os.remove(os.path.join(actions_cache_dir, fname))
                removed += 1
            except OSError:
                pass
        print(
            f"[run:{run_name}] Action cache invalidated ({reason}). "
            f"Removed {removed} cached action files."
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cache_manifest, f, indent=2)


def _load_external_series_candidates(
    market_cache_dir: str,
    candidates: list[str],
    start_date: str,
    end_date: str,
    interval: str,
    download_if_missing: bool,
    price_col: str = "Close",
):
    last_exc = None
    for ticker_candidate in candidates:
        try:
            return load_external_series(
                market_cache_dir=market_cache_dir,
                ticker=ticker_candidate,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                download_if_missing=download_if_missing,
                price_col=price_col,
            ), ticker_candidate
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise ValueError(
            f"Unable to load external series from candidates={candidates}. Last error: {last_exc}"
        ) from last_exc
    raise ValueError(f"Unable to load external series from candidates={candidates}.")


def _sigma_mode(config: dict[str, Any]) -> str:
    return str(config.get("sigma_mode", "train_average")).strip().lower()


def _historical_sigma_window_days(config: dict[str, Any]) -> int:
    raw = config.get("historical_sigma_window_days", config.get("trading_days_per_year", 252))
    return int(raw)


def _benchmark_delta_sigma_mode(config: dict[str, Any]) -> str:
    return str(config.get("benchmark_delta_sigma_mode", "none")).strip().lower()


def _resolve_float_override(config: dict[str, Any], key: str, fallback: float) -> float:
    raw = config.get(key, None)
    if raw is None:
        return float(fallback)
    return float(raw)


def _build_window_prehistory_from_series(
    close_df: pd.DataFrame,
    window_start_dates: list[pd.Timestamp],
    context_days: int,
    price_col: str = "Close",
) -> np.ndarray:
    """
    Build per-window pre-history prices (oldest -> newest), excluding S0.
    """
    context_days = int(context_days)
    if context_days <= 0:
        return np.zeros((len(window_start_dates), 0), dtype=np.float32)

    df = close_df.copy()
    if "Date" not in df.columns:
        raise ValueError("close_df must contain 'Date' column.")
    if price_col not in df.columns:
        raise ValueError(f"close_df must contain '{price_col}' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError("close_df is empty after cleaning.")

    dates = df["Date"].to_numpy()
    prices = df[price_col].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("close_df prices must be finite and > 0.")

    pre_hist = np.zeros((len(window_start_dates), context_days), dtype=np.float64)
    for i, dt in enumerate(window_start_dates):
        ts = pd.Timestamp(dt)
        idx = np.searchsorted(dates, np.datetime64(ts), side="left")
        if idx >= len(dates):
            idx = len(dates) - 1
        # Prefer exact start date. If not found, use previous available date.
        if idx < len(dates) and pd.Timestamp(dates[idx]) != ts:
            idx = max(0, idx - 1)
        start = max(0, idx - context_days)
        hist = prices[start:idx]
        if hist.size >= context_days:
            pre_hist[i, :] = hist[-context_days:]
        elif hist.size > 0:
            pad = np.full((context_days - hist.size,), hist[0], dtype=np.float64)
            pre_hist[i, :] = np.concatenate([pad, hist], axis=0)
        else:
            anchor = prices[idx]
            pre_hist[i, :] = np.full((context_days,), anchor, dtype=np.float64)

    return pre_hist.astype(np.float32)


def run_comparison(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    dirs = _run_dirs(run_name, config_path=config_path)
    copied_cfg = copy_config_snapshot(config_path, dirs["run_dir"])
    resolved_cfg_path = os.path.join(dirs["run_dir"], "resolved_config.json")
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[run:{run_name}] Config validated and copied to: {copied_cfg}")
    print(f"[run:{run_name}] Storage root: {RESULT1B_ROOT}")

    eval_seed = int(config["eval_seed"])
    set_global_determinism(eval_seed)

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

    n = int(config["n"])
    instrument = build_instrument_from_config(cfg_for_builders)

    claim = build_claim_from_config(cfg_for_builders)
    risk_measure = build_risk_measure_from_config(cfg_for_builders)

    benchmark_agent = build_agent_from_config(
        agent_name=str(config["benchmark_agent_name"]),
        instrument=instrument,
        claim=claim,
        config=cfg_for_builders,
    )

    trained_agents = []
    for item in config["trained_agents"]:
        agent = build_agent_from_config(
            agent_name=str(item["agent_name"]),
            instrument=instrument,
            claim=claim,
            config=cfg_for_builders,
        )
        model_path = _model_path(agent, str(item["model_name"]))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found for trained agent {item['agent_name']} at: {model_path}")
        agent.load_model(model_path)
        print(f"[run:{run_name}] Loaded model: {model_path}")
        trained_agents.append(agent)

    benchmark_compare_agents = []
    used_names = {str(getattr(benchmark_agent, "name", "benchmark"))}
    for i, item in enumerate(config.get("benchmark_agents_to_compare", [])):
        per_agent_cfg = dict(cfg_for_builders)
        per_agent_cfg.update(dict(item.get("config_overrides", {})))
        agent = build_agent_from_config(
            agent_name=str(item["agent_name"]),
            instrument=instrument,
            claim=claim,
            config=per_agent_cfg,
        )
        label = str(item.get("label") or _display_name(agent, str(config["language"]))).strip()
        slug = _slugify_label(label)
        base_name = f"bm_{i+1:02d}_{slug}"
        unique_name = base_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)
        agent.name = unique_name
        agent.agent_id = unique_name
        agent.plot_name = {"es": label, "en": label}
        benchmark_compare_agents.append(agent)
        print(f"[run:{run_name}] Added benchmark comparison agent: {label} ({unique_name})")

    compare_targets = benchmark_compare_agents + trained_agents
    all_agents = [benchmark_agent] + compare_targets
    env = build_environment(
        agent=benchmark_agent,
        instrument=instrument,
        claim=claim,
        proportional_cost=float(config["proportional_cost"]),
        risk_measure=risk_measure,
        n_epochs=1,
        batch_size=min(int(config["eval_paths"]), 10_000),
        learning_rate_schedule=1e-3,
        optimizer_cls=tf.keras.optimizers.Adam,
        resample_each_epoch=False,
        train_seed=eval_seed,
        use_price_history_context=bool(config.get("use_price_history_context", False)),
        context_length=int(config.get("context_length", 0)),
        context_feature_mode=str(config.get("context_feature_mode", "log_returns")),
        context_visible_to_agent=not bool(config.get("context_for_path_generation_only", False)),
    )

    eval_paths_tensor = None
    eval_pre_history_tensor = None
    per_path_r = None
    per_path_sigma = None
    windows_meta = None

    test_data_mode = str(config["test_data_mode"]).strip().lower()
    delta_sigma_mode = _benchmark_delta_sigma_mode(config)
    delta_sigma_context_days = int(config.get("benchmark_delta_sigma_context_days", 50))
    delta_sigma_min_obs = int(config.get("benchmark_delta_sigma_min_obs", 10))
    delta_sigma_alpha = _resolve_float_override(
        config, "benchmark_delta_sigma_garch_alpha", config.get("garch_alpha", 0.05)
    )
    delta_sigma_beta = _resolve_float_override(
        config, "benchmark_delta_sigma_garch_beta", config.get("garch_beta", 0.9)
    )
    delta_sigma_leverage = _resolve_float_override(
        config, "benchmark_delta_sigma_garch_leverage", config.get("garch_leverage", 0.0)
    )
    delta_sigma_omega = config.get("benchmark_delta_sigma_garch_omega", None)
    delta_sigma_floor = float(config.get("benchmark_delta_sigma_floor", 1e-6))
    delta_sigma_cap = config.get("benchmark_delta_sigma_cap", None)
    delta_sigma_default = _resolve_float_override(
        config, "benchmark_delta_sigma_default", float(calib.sigma_train)
    )

    if test_data_mode == "historical_windows":
        test_csv = os.path.join(
            dirs["market_cache_dir"],
            f"{config['ticker']}_{config['test_start_date']}_{config['test_end_date']}_{config['interval']}.csv",
        )
        if not os.path.isfile(test_csv):
            from DeepHedging.utils.market_data import download_ohlcv_to_csv

            download_ohlcv_to_csv(
                ticker=str(config["ticker"]),
                start_date=str(config["test_start_date"]),
                end_date=str(config["test_end_date"]),
                interval=str(config["interval"]),
                output_csv=test_csv,
            )

        windows_2d, windows_meta = build_historical_windows_from_csv(
            csv_path=test_csv,
            n_hedging_steps=n,
            start_date=str(config["test_start_date"]),
            end_date=str(config["test_end_date"]),
            price_col=str(config["price_col"]),
            stride=int(config["historical_stride"]),
            max_windows=None if config["max_test_windows"] is None else int(config["max_test_windows"]),
        )

        # Normalize each real window to configured S0 so test scale matches training world.
        raw_window_start = windows_2d[:, 0].astype(np.float64)
        if np.any(~np.isfinite(raw_window_start)) or np.any(raw_window_start <= 0.0):
            raise ValueError("Invalid historical window start prices for normalization.")
        target_s0 = float(config["s0"])
        windows_2d = (target_s0 * (windows_2d / raw_window_start[:, None])).astype(np.float32)
        windows_meta["raw_window_start_price"] = raw_window_start
        windows_meta["normalized_start_price"] = target_s0

        eval_paths_tensor = to_environment_paths(windows_2d)
        eval_paths_2d = np.asarray(eval_paths_tensor, dtype=np.float32)[:, :, 0]

        risk_free_mode = str(config["risk_free_mode"]).strip().lower()
        if str(config["risk_free_source"]).strip().lower() == "fixed":
            per_path_r = np.full((eval_paths_tensor.shape[0],), float(config["fixed_risk_free"]), dtype=np.float32)
        elif risk_free_mode == "train_average":
            per_path_r = np.full((eval_paths_tensor.shape[0],), float(calib.r_train), dtype=np.float32)
        else:
            try:
                irx_df, irx_ticker = _load_external_series_candidates(
                    market_cache_dir=dirs["market_cache_dir"],
                    candidates=[IRX_TICKER, "IRX", "^FVX", "^TNX"],
                    start_date=str(config["test_start_date"]),
                    end_date=str(config["test_end_date"]),
                    interval=str(config["interval"]),
                    download_if_missing=bool(config["download_if_missing"]),
                    price_col="Close",
                )
                irx_scale = 0.01 if irx_ticker in {IRX_TICKER, "IRX"} else 0.001
                per_path_r = map_series_to_window_start(
                    series_df=irx_df,
                    window_start_dates=windows_meta["start_date"].tolist(),
                    value_col="Close",
                    scale=float(irx_scale),
                    default_value=float(calib.r_train),
                ).astype(np.float32)
            except Exception as exc:
                print(
                    f"[run:{run_name}] [warning] Failed to load per-window risk-free series; "
                    f"falling back to calibrated train-average r. Detail: {exc}"
                )
                per_path_r = np.full((eval_paths_tensor.shape[0],), float(calib.r_train), dtype=np.float32)

        sigma_mode = _sigma_mode(config)
        sigma_source = str(config["sigma_source"]).strip().lower()
        if sigma_mode == "train_average":
            per_path_sigma = np.full((eval_paths_tensor.shape[0],), float(calib.sigma_train), dtype=np.float32)
        elif sigma_source == "implied" and sigma_mode == "per_window_start":
            implied_source = str(config["implied_vol_source"]).strip().lower()
            if implied_source == "vix":
                try:
                    vix_df, _ = _load_external_series_candidates(
                        market_cache_dir=dirs["market_cache_dir"],
                        candidates=[VIX_TICKER, "VIX"],
                        start_date=str(config["test_start_date"]),
                        end_date=str(config["test_end_date"]),
                        interval=str(config["interval"]),
                        download_if_missing=bool(config["download_if_missing"]),
                        price_col="Close",
                    )
                    per_path_sigma = map_series_to_window_start(
                        series_df=vix_df,
                        window_start_dates=windows_meta["start_date"].tolist(),
                        value_col="Close",
                        scale=0.01,
                        default_value=float(calib.sigma_train),
                    ).astype(np.float32)
                except Exception as exc:
                    print(
                        f"[run:{run_name}] [warning] Failed to load per-window implied sigma; "
                        f"falling back to train-average sigma. Detail: {exc}"
                    )
                    per_path_sigma = np.full((eval_paths_tensor.shape[0],), float(calib.sigma_train), dtype=np.float32)
            else:
                if implied_source == "option_market":
                    option_csv = os.path.abspath(os.path.expanduser(str(config["option_quotes_csv"])))
                    print(
                        f"[run:{run_name}] Mapping per-window implied sigma from option prices: {option_csv}"
                    )
                    sigma_vals, iv_details = map_option_market_implied_vol_to_window_start(
                        option_quotes_csv=option_csv,
                        option_quote_date_col=str(config["option_quote_date_col"]),
                        option_expiry_col=str(config["option_expiry_col"]),
                        option_strike_col=str(config["option_strike_col"]),
                        option_type_col=str(config["option_type_col"]),
                        option_bid_col=str(config["option_bid_col"]),
                        option_ask_col=str(config["option_ask_col"]),
                        option_last_col=str(config["option_last_col"]),
                        option_type=str(config["implied_option_type"]),
                        window_start_dates=windows_meta["start_date"].tolist(),
                        window_start_spots=windows_meta["raw_window_start_price"].to_numpy(dtype=np.float64),
                        per_path_r=per_path_r.astype(np.float64),
                        n_trading_days=n,
                        trading_days_per_year=int(config["trading_days_per_year"]),
                        calendar_days_per_year=float(config["calendar_days_per_year"]),
                        max_quote_lag_days=int(config["option_max_quote_lag_days"]),
                        max_expiry_diff_days=int(config["option_max_expiry_diff_days"]),
                        default_sigma=float(calib.sigma_train),
                    )
                    per_path_sigma = sigma_vals.astype(np.float32)
                    iv_details_csv = os.path.join(
                        dirs["tables_dir"], "option_market_implied_sigma_details.csv"
                    )
                    iv_details.to_csv(iv_details_csv, index=False)
                    matched = int(
                        (iv_details.get("sigma_source_status", pd.Series(dtype=str)) == "option_implied").sum()
                    )
                    print(
                        f"[run:{run_name}] Option-market implied sigma mapped: matched={matched}/{len(iv_details)}. "
                        f"Details: {iv_details_csv}"
                    )
                else:
                    # implied_vol_source='fixed' with per-window mode is effectively constant.
                    per_path_sigma = np.full((eval_paths_tensor.shape[0],), float(calib.sigma_train), dtype=np.float32)
        elif sigma_source == "historical" and sigma_mode == "rolling_pre_window":
            full_close_df = load_external_series(
                market_cache_dir=dirs["market_cache_dir"],
                ticker=str(config["ticker"]),
                start_date=str(config["train_start_date"]),
                end_date=str(config["test_end_date"]),
                interval=str(config["interval"]),
                download_if_missing=bool(config["download_if_missing"]),
                price_col=str(config["price_col"]),
            )
            sigma_window_days = _historical_sigma_window_days(config)
            per_path_sigma = map_historical_sigma_to_window_start(
                close_df=full_close_df,
                window_start_dates=windows_meta["start_date"].tolist(),
                price_col=str(config["price_col"]),
                window_days=int(sigma_window_days),
                trading_days_per_year=int(config["trading_days_per_year"]),
                default_value=float(calib.sigma_train),
            ).astype(np.float32)
        else:
            # Any not-explicitly-pathwise mode falls back to train-calibrated sigma.
            per_path_sigma = np.full((eval_paths_tensor.shape[0],), float(calib.sigma_train), dtype=np.float32)

        # Build normalized pre-history for context-aware agents and/or benchmark-delta sigma estimation.
        pre_hist_max_days = max(
            int(config.get("context_length", 0)) if bool(config.get("use_price_history_context", False)) else 0,
            int(delta_sigma_context_days) if delta_sigma_mode != "none" else 0,
        )
        pre_hist_for_sigma = None
        if pre_hist_max_days > 0:
            full_close_df_ctx = load_external_series(
                market_cache_dir=dirs["market_cache_dir"],
                ticker=str(config["ticker"]),
                start_date=str(config["train_start_date"]),
                end_date=str(config["test_end_date"]),
                interval=str(config["interval"]),
                download_if_missing=bool(config["download_if_missing"]),
                price_col=str(config["price_col"]),
            )
            pre_hist_raw = _build_window_prehistory_from_series(
                close_df=full_close_df_ctx,
                window_start_dates=windows_meta["start_date"].tolist(),
                context_days=pre_hist_max_days,
                price_col=str(config["price_col"]),
            )
            # Normalize on same scale as eval_paths_tensor.
            scale = (float(config["s0"]) / windows_meta["raw_window_start_price"].to_numpy(dtype=np.float64))
            pre_hist_norm = (pre_hist_raw * scale[:, None]).astype(np.float32)
            if bool(config.get("use_price_history_context", False)) and int(config.get("context_length", 0)) > 0:
                agent_ctx = int(config.get("context_length", 0))
                eval_pre_history_tensor = pre_hist_norm[:, -agent_ctx:]
            if delta_sigma_mode != "none" and int(delta_sigma_context_days) > 0:
                pre_hist_for_sigma = pre_hist_norm[:, -int(delta_sigma_context_days):]

        if delta_sigma_mode != "none":
            garch_mode = "stepwise" if delta_sigma_mode == "garch_context_stepwise" else "static"
            per_path_sigma = estimate_pathwise_garch_sigma_from_context(
                hedge_paths_2d=eval_paths_2d,
                pre_history_prices_2d=pre_hist_for_sigma,
                context_days=int(delta_sigma_context_days),
                mode=garch_mode,
                trading_days_per_year=int(config["trading_days_per_year"]),
                garch_alpha=float(delta_sigma_alpha),
                garch_beta=float(delta_sigma_beta),
                garch_leverage=float(delta_sigma_leverage),
                garch_omega=None if delta_sigma_omega is None else float(delta_sigma_omega),
                min_obs=int(delta_sigma_min_obs),
                default_sigma=float(delta_sigma_default),
                sigma_floor=float(delta_sigma_floor),
                sigma_cap=None if delta_sigma_cap is None else float(delta_sigma_cap),
            )
            if np.ndim(per_path_sigma) == 1:
                print(
                    f"[run:{run_name}] benchmark_delta_sigma_mode={delta_sigma_mode}: "
                    f"sigma mean={float(np.mean(per_path_sigma)):.6f}, std={float(np.std(per_path_sigma)):.6f}"
                )
            else:
                sigma_t0 = np.asarray(per_path_sigma)[:, 0]
                print(
                    f"[run:{run_name}] benchmark_delta_sigma_mode={delta_sigma_mode}: "
                    f"sigma_t0 mean={float(np.mean(sigma_t0)):.6f}, std={float(np.std(sigma_t0)):.6f}, "
                    f"steps={int(np.asarray(per_path_sigma).shape[1])}"
                )

        windows_meta["risk_free_window"] = per_path_r
        if np.ndim(per_path_sigma) == 1:
            windows_meta["sigma_window"] = per_path_sigma
        else:
            sigma_arr = np.asarray(per_path_sigma, dtype=np.float32)
            windows_meta["sigma_window_t0"] = sigma_arr[:, 0]
            sigma_steps_csv = os.path.join(dirs["tables_dir"], "sigma_window_stepwise.csv")
            pd.DataFrame(sigma_arr).to_csv(sigma_steps_csv, index=False)
            print(f"[run:{run_name}] Saved stepwise sigma matrix: {sigma_steps_csv}")
        window_csv = os.path.join(dirs["tables_dir"], "historical_window_metadata.csv")
        windows_meta.to_csv(window_csv, index=False)
        print(f"[run:{run_name}] Saved historical window metadata: {window_csv}")
        print(f"[run:{run_name}] Evaluating on {eval_paths_tensor.shape[0]} historical windows.")

    elif test_data_mode == "simulated" and delta_sigma_mode != "none":
        # Force deterministic simulated paths with explicit pre-history for pathwise GARCH sigma estimation.
        eval_n_paths = int(config["eval_paths"])
        context_for_agent = int(config.get("context_length", 0)) if bool(config.get("use_price_history_context", False)) else 0
        n_context_gen = max(int(delta_sigma_context_days), int(context_for_agent))
        if hasattr(instrument, "generate_paths_with_context") and n_context_gen > 0:
            full_paths = instrument.generate_paths_with_context(
                eval_n_paths,
                n_context_steps=int(n_context_gen),
                random_seed=eval_seed,
            )
            full_paths = tf.convert_to_tensor(full_paths, dtype=tf.float32)
            all_pre_history = full_paths[:, : int(n_context_gen)]
            if context_for_agent > 0:
                eval_pre_history_tensor = all_pre_history[:, -int(context_for_agent):]
            eval_paths_tensor = tf.expand_dims(full_paths[:, int(n_context_gen) :], axis=-1)
            pre_history_for_sigma = all_pre_history[:, -int(delta_sigma_context_days):]
        else:
            eval_paths_tensor = env.generate_data(eval_n_paths, random_seed=eval_seed)
            eval_pre_history_tensor = None
            pre_history_for_sigma = None

        eval_paths_2d = np.asarray(eval_paths_tensor, dtype=np.float32)[:, :, 0]
        if str(config["risk_free_source"]).strip().lower() == "fixed":
            per_path_r = np.full((eval_paths_tensor.shape[0],), float(config["fixed_risk_free"]), dtype=np.float32)
        else:
            per_path_r = np.full((eval_paths_tensor.shape[0],), float(calib.r_train), dtype=np.float32)

        garch_mode = "stepwise" if delta_sigma_mode == "garch_context_stepwise" else "static"
        per_path_sigma = estimate_pathwise_garch_sigma_from_context(
            hedge_paths_2d=eval_paths_2d,
            pre_history_prices_2d=None if pre_history_for_sigma is None else np.asarray(pre_history_for_sigma, dtype=np.float32),
            context_days=int(delta_sigma_context_days),
            mode=garch_mode,
            trading_days_per_year=int(config["trading_days_per_year"]),
            garch_alpha=float(delta_sigma_alpha),
            garch_beta=float(delta_sigma_beta),
            garch_leverage=float(delta_sigma_leverage),
            garch_omega=None if delta_sigma_omega is None else float(delta_sigma_omega),
            min_obs=int(delta_sigma_min_obs),
            default_sigma=float(delta_sigma_default),
            sigma_floor=float(delta_sigma_floor),
            sigma_cap=None if delta_sigma_cap is None else float(delta_sigma_cap),
        )
        if np.ndim(per_path_sigma) == 1:
            print(
                f"[run:{run_name}] benchmark_delta_sigma_mode={delta_sigma_mode} (simulated): "
                f"sigma mean={float(np.mean(per_path_sigma)):.6f}, std={float(np.std(per_path_sigma)):.6f}"
            )
        else:
            sigma_t0 = np.asarray(per_path_sigma)[:, 0]
            print(
                f"[run:{run_name}] benchmark_delta_sigma_mode={delta_sigma_mode} (simulated): "
                f"sigma_t0 mean={float(np.mean(sigma_t0)):.6f}, std={float(np.std(sigma_t0)):.6f}, "
                f"steps={int(np.asarray(per_path_sigma).shape[1])}"
            )

    actions_cache_dir = None
    if bool(config.get("reuse_actions_between_steps", True)):
        actions_cache_dir = os.path.join(dirs["run_dir"], "actions_cache")
        cache_manifest = {
            "schema_version": 5,
            "actions_cache_schema_version": int(ACTIONS_CACHE_SCHEMA_VERSION),
            "run_name": str(run_name),
            "benchmark_agent": str(config["benchmark_agent_name"]),
            "trained_agents": config["trained_agents"],
            "benchmark_agents_to_compare": config.get("benchmark_agents_to_compare", []),
            "pricing_method": str(config["pricing_method"]),
            "price_computation_mode": str(config.get("price_computation_mode", "pathwise_if_available")),
            "test_data_mode": str(config["test_data_mode"]),
            "risk_free_source": str(config["risk_free_source"]),
            "risk_free_mode": str(config["risk_free_mode"]),
            "per_path_r_signature": _vector_signature(per_path_r),
            "sigma_source": str(config["sigma_source"]),
            "sigma_mode": _sigma_mode(config),
            "benchmark_delta_sigma_mode": _benchmark_delta_sigma_mode(config),
            "benchmark_delta_sigma_context_days": int(config.get("benchmark_delta_sigma_context_days", 50)),
            "benchmark_delta_sigma_min_obs": int(config.get("benchmark_delta_sigma_min_obs", 10)),
            "benchmark_delta_sigma_garch_alpha": config.get("benchmark_delta_sigma_garch_alpha"),
            "benchmark_delta_sigma_garch_beta": config.get("benchmark_delta_sigma_garch_beta"),
            "benchmark_delta_sigma_garch_leverage": config.get("benchmark_delta_sigma_garch_leverage"),
            "benchmark_delta_sigma_garch_omega": config.get("benchmark_delta_sigma_garch_omega"),
            "instrument_model": str(config.get("instrument_model", "gbm")).strip().lower(),
            "gbm_sigma_per_path_mode": str(config.get("gbm_sigma_per_path_mode", "fixed")),
            "gbm_sigma_uniform_low": config.get("gbm_sigma_uniform_low"),
            "gbm_sigma_uniform_high": config.get("gbm_sigma_uniform_high"),
            "gbm_sigma_discrete_values": config.get("gbm_sigma_discrete_values"),
            "gbm_sigma_discrete_probs": config.get("gbm_sigma_discrete_probs"),
            "garch_alpha": config.get("garch_alpha"),
            "garch_beta": config.get("garch_beta"),
            "garch_omega": config.get("garch_omega"),
            "garch_leverage": config.get("garch_leverage"),
            "garch_use_student_t": config.get("garch_use_student_t"),
            "garch_student_t_df": config.get("garch_student_t_df"),
            "per_path_sigma_signature": _vector_signature(per_path_sigma),
            "use_price_history_context": bool(config.get("use_price_history_context", False)),
            "context_for_path_generation_only": bool(config.get("context_for_path_generation_only", False)),
            "context_length": int(config.get("context_length", 0)),
            "context_feature_mode": str(config.get("context_feature_mode", "log_returns")),
            "context_pre_ttm_mode": str(config.get("context_pre_ttm_mode", "calculated")),
            "history_conv1d_enabled": bool(config.get("history_conv1d_enabled", False)),
            "history_conv1d_layers": config.get("history_conv1d_layers"),
            "history_conv1d_pooling": str(config.get("history_conv1d_pooling", "global_max")),
            "paths_signature": None if eval_paths_tensor is None else _paths_signature(eval_paths_tensor),
            "pre_history_signature": _vector_signature(
                None if eval_pre_history_tensor is None else np.asarray(eval_pre_history_tensor, dtype=np.float32)
            ),
        }
        _ensure_actions_cache_consistency(
            actions_cache_dir=actions_cache_dir,
            cache_manifest=cache_manifest,
            run_name=run_name,
        )

    eval_agent_batch_size = config.get("eval_agent_batch_size")
    terminal_progress_every = config.get("terminal_progress_log_every_agent_batches")

    loss_fns = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase()]
    pairwise_rows = []
    for agent in compare_targets:
        pair_name = f"{benchmark_agent.name}_vs_{agent.name}"
        plot_path = os.path.join(dirs["plots_dir"], f"{pair_name}.jpg")
        stats_path = os.path.join(dirs["tables_dir"], f"{pair_name}.xlsx")
        pair_df = env.terminal_hedging_error_multiple_agents(
            agents=[benchmark_agent, agent],
            n_paths=int(config["eval_paths"]),
            random_seed=eval_seed,
            paths_to_test=eval_paths_tensor,
            pre_history_prices_to_test=eval_pre_history_tensor,
            per_path_r=per_path_r,
            per_path_sigma=per_path_sigma,
            plot_error=True,
            plot_title="Error de Cobertura Terminal",
            save_plot_path=plot_path,
            save_stats_path=stats_path,
            loss_functions=loss_fns,
            min_x=float(config["plot_min_x"]),
            max_x=float(config["plot_max_x"]),
            language="es",
            pricing_method=str(config["pricing_method"]),
            price_computation_mode=str(config.get("price_computation_mode", "pathwise_if_available")),
            agent_eval_batch_size=int(eval_agent_batch_size) if eval_agent_batch_size is not None else None,
            progress_log_every_agent_batches=int(terminal_progress_every) if terminal_progress_every is not None else 5,
            save_actions_path=actions_cache_dir,
        )
        pair_df.insert(0, "pair_name", pair_name)
        pairwise_rows.append(pair_df)
        print(f"[run:{run_name}] Saved pair plot: {plot_path}")
        print(f"[run:{run_name}] Saved pair stats: {stats_path}")

    if pairwise_rows:
        pairwise_all = pd.concat(pairwise_rows, ignore_index=True)
    else:
        pairwise_all = pd.DataFrame()
    pairwise_csv = os.path.join(dirs["tables_dir"], "pairwise_terminal_stats.csv")
    pairwise_all.to_csv(pairwise_csv, index=False)

    point_payload, errors_all = env.terminal_hedging_error_multiple_agents(
        agents=all_agents,
        n_paths=int(config["eval_paths"]),
        random_seed=eval_seed,
        paths_to_test=eval_paths_tensor,
        pre_history_prices_to_test=eval_pre_history_tensor,
        per_path_r=per_path_r,
        per_path_sigma=per_path_sigma,
        plot_error=False,
        loss_functions=loss_fns,
        min_x=float(config["plot_min_x"]),
        max_x=float(config["plot_max_x"]),
        language="es",
        pricing_method=str(config["pricing_method"]),
        price_computation_mode=str(config.get("price_computation_mode", "pathwise_if_available")),
        agent_eval_batch_size=int(eval_agent_batch_size) if eval_agent_batch_size is not None else None,
        progress_log_every_agent_batches=int(terminal_progress_every) if terminal_progress_every is not None else 5,
        save_actions_path=actions_cache_dir,
        return_errors=True,
    )
    mean_errors, std_errors, loss_results = point_payload

    point_rows = []
    empirical_rows = []
    benchmark_error = np.asarray(errors_all[0]).reshape(-1)
    for i, agent in enumerate(all_agents):
        row = {
            "Agent": _display_name(agent, str(config["language"])),
            "Mean": float(mean_errors[i]),
            "StdDev": float(std_errors[i]),
        }
        for key, values in (loss_results or {}).items():
            row[key] = float(values[i])
        point_rows.append(row)

        err = np.asarray(errors_all[i]).reshape(-1)
        risk_row = {"Agent": _display_name(agent, str(config["language"]))}
        risk_row.update(_compute_empirical_error_metrics(err))
        risk_row["exceedance_left_1pct"] = float(np.mean(err <= np.quantile(benchmark_error, 0.01)))
        risk_row["exceedance_left_0_1pct"] = float(np.mean(err <= np.quantile(benchmark_error, 0.001)))
        empirical_rows.append(risk_row)

    point_df = pd.DataFrame(point_rows)
    point_csv = os.path.join(dirs["tables_dir"], "point_metrics.csv")
    point_df.to_csv(point_csv, index=False)
    print(f"[run:{run_name}] Saved point metrics: {point_csv}")

    empirical_df = pd.DataFrame(empirical_rows)
    empirical_csv = os.path.join(dirs["tables_dir"], "empirical_risk_metrics.csv")
    empirical_df.to_csv(empirical_csv, index=False)
    print(f"[run:{run_name}] Saved empirical risk metrics: {empirical_csv}")

    mc_profile_rows = []
    for agent in all_agents:
        rows = getattr(agent, "_mc_profile_rows", None)
        if rows:
            for row in rows:
                enriched = dict(row)
                enriched["agent_name"] = str(getattr(agent, "name", "unknown_agent"))
                mc_profile_rows.append(enriched)
    if mc_profile_rows:
        mc_profile_df = pd.DataFrame(mc_profile_rows)
        mc_profile_csv = os.path.join(dirs["tables_dir"], "mc_profile.csv")
        mc_profile_df.to_csv(mc_profile_csv, index=False)
        print(f"[run:{run_name}] Saved MC profile: {mc_profile_csv}")

    if bool(config["bootstrap_enabled"]):
        boot_df = env.bootstrap_confidence_intervals(
            agents=all_agents,
            statistics=bootstrap_statistics_list(),
            n_paths=int(config["eval_paths"]),
            n_bootstraps=int(config["bootstrap_n_bootstraps"]),
            confidence_level=float(config["bootstrap_confidence_level"]),
            random_seed=eval_seed,
            paths_to_test=eval_paths_tensor,
            pre_history_prices_to_test=eval_pre_history_tensor,
            per_path_r=per_path_r,
            per_path_sigma=per_path_sigma,
            plot_histograms=False,
            language="es",
            pricing_method=str(config["pricing_method"]),
            price_computation_mode=str(config.get("price_computation_mode", "pathwise_if_available")),
            batch_size=int(config["bootstrap_batch_size"]),
            bootstrap_method=str(config["bootstrap_method"]),
            moving_block_size=None if config["moving_block_size"] is None else int(config["moving_block_size"]),
            save_actions_path=actions_cache_dir,
        )
        boot_csv = os.path.join(dirs["tables_dir"], "bootstrap_metrics_wide.csv")
        boot_df.to_csv(boot_csv, index=False)
        print(f"[run:{run_name}] Saved bootstrap wide table: {boot_csv}")

    calibration_manifest = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "ticker": str(config["ticker"]),
                "train_start_date": str(config["train_start_date"]),
                "train_end_date": str(config["train_end_date"]),
                "test_start_date": str(config["test_start_date"]),
                "test_end_date": str(config["test_end_date"]),
                "sigma_source": str(config["sigma_source"]),
                "sigma_mode": _sigma_mode(config),
                "benchmark_delta_sigma_mode": _benchmark_delta_sigma_mode(config),
                "benchmark_delta_sigma_context_days": int(config.get("benchmark_delta_sigma_context_days", 50)),
                "historical_sigma_window_days": _historical_sigma_window_days(config),
                "use_price_history_context": bool(config.get("use_price_history_context", False)),
                "context_for_path_generation_only": bool(config.get("context_for_path_generation_only", False)),
                "context_length": int(config.get("context_length", 0)),
                "context_feature_mode": str(config.get("context_feature_mode", "log_returns")),
                "context_pre_ttm_mode": str(config.get("context_pre_ttm_mode", "calculated")),
                "history_conv1d_enabled": bool(config.get("history_conv1d_enabled", False)),
                "history_conv1d_layers": json.dumps(config.get("history_conv1d_layers")),
                "history_conv1d_pooling": str(config.get("history_conv1d_pooling", "global_max")),
                "gbm_sigma_per_path_mode": str(config.get("gbm_sigma_per_path_mode", "fixed")),
                "gbm_sigma_uniform_low": config.get("gbm_sigma_uniform_low"),
                "gbm_sigma_uniform_high": config.get("gbm_sigma_uniform_high"),
                "gbm_sigma_discrete_values": json.dumps(config.get("gbm_sigma_discrete_values")),
                "gbm_sigma_discrete_probs": json.dumps(config.get("gbm_sigma_discrete_probs")),
                "risk_free_source": str(config["risk_free_source"]),
                "risk_free_mode": str(config["risk_free_mode"]),
                "sigma_train": float(calib.sigma_train),
                "sigma_eval_mean": float(np.mean(per_path_sigma)) if per_path_sigma is not None else float(calib.sigma_train),
                "sigma_eval_std": float(np.std(per_path_sigma)) if per_path_sigma is not None else 0.0,
                "r_train": float(calib.r_train),
                "mu_train": float(calib.mu_train),
                "train_rows": int(calib.train_rows),
                "close_csv_path": str(calib.close_csv_path),
                "implied_csv_path": "" if calib.implied_csv_path is None else str(calib.implied_csv_path),
                "risk_free_csv_path": "" if calib.risk_free_csv_path is None else str(calib.risk_free_csv_path),
            }
        ]
    )
    calibration_csv = os.path.join(dirs["tables_dir"], "calibration_manifest.csv")
    calibration_manifest.to_csv(calibration_csv, index=False)
    print(f"[run:{run_name}] Saved calibration manifest: {calibration_csv}")

    meta = {
        "run_name": run_name,
        "benchmark_agent": str(config["benchmark_agent_name"]),
        "trained_agents": config["trained_agents"],
        "benchmark_agents_to_compare": config.get("benchmark_agents_to_compare", []),
        "eval_paths_requested": int(config["eval_paths"]),
        "test_data_mode": str(config["test_data_mode"]),
        "price_computation_mode": str(config.get("price_computation_mode", "pathwise_if_available")),
        "sigma_mode": _sigma_mode(config),
        "benchmark_delta_sigma_mode": _benchmark_delta_sigma_mode(config),
        "benchmark_delta_sigma_context_days": int(config.get("benchmark_delta_sigma_context_days", 50)),
        "historical_sigma_window_days": _historical_sigma_window_days(config),
        "bootstrap_enabled": bool(config["bootstrap_enabled"]),
        "bootstrap_n_bootstraps": int(config["bootstrap_n_bootstraps"]),
        "bootstrap_confidence_level": float(config["bootstrap_confidence_level"]),
    }
    meta_path = os.path.join(dirs["run_dir"], "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[run:{run_name}] Saved run metadata: {meta_path}")
    print(f"[run:{run_name}] Completed.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare thesis Result 1b agents from JSON config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional config filename in ./thesis_result1b_configs/compare (extension optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1b_configs", "compare")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, cfg_path, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter COMPARE JSON config name from 'thesis_result1b_configs/compare': ",
    )
    print(f"[run:{run_name}] Loaded config: {cfg_path}")
    validate_config(cfg)
    run_comparison(run_name=run_name, config_path=cfg_path, config=cfg)


if __name__ == "__main__":
    main()
