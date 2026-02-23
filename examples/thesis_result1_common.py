"""
Common utilities for thesis "Result 1" experiment runners.

Runners using this module must provide strict JSON configs and pass only the
file name (without path), following the same UX pattern as timegan scripts.
"""

import inspect
import json
import os
import random
import shutil
from typing import Any

import numpy as np
import tensorflow as tf

# Ensure local imports work without installing package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in os.sys.path:
    os.sys.path.insert(0, SRC_DIR)

from DeepHedging.Agents import (
    ArithmeticAsianControlVariateAgent,
    DeltaHedgingAgent,
    GRUAgent,
    LSTMAgent,
    MonteCarloAgent,
    QuantlibAsianGeometricAgent,
    RecurrentAgent,
    SimpleAgent,
    WaveNetAgent,
    GeometricAsianDeltaHedgingAgent,
    GeometricAsianDeltaHedgingAgent2,
    GeometricAsianNumericalDeltaHedgingAgent,
    ArithmeticAsianMonteCarloAgent,
)
from DeepHedging.ContingentClaims import (
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
    EuropeanCall,
    EuropeanPut,
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.Environments import Environment
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.RiskMeasures import CVaR, MAE, Mean, StdDev, WorstCase
from DeepHedging.RiskMeasures import MSE

THESIS_MODELS_ROOT = os.path.normpath(r"G:\Mi unidad\Tesis2026\Models\Organized")

TRAINABLE_AGENT_NAMES = {"SimpleAgent", "RecurrentAgent", "LSTMAgent", "GRUAgent", "WaveNetAgent"}

AGENTS = {
    "SimpleAgent": SimpleAgent,
    "RecurrentAgent": RecurrentAgent,
    "LSTMAgent": LSTMAgent,
    "GRUAgent": GRUAgent,
    "WaveNetAgent": WaveNetAgent,
    "DeltaHedgingAgent": DeltaHedgingAgent,
    "GeometricAsianDeltaHedgingAgent": GeometricAsianDeltaHedgingAgent,
    "GeometricAsianDeltaHedgingAgent2": GeometricAsianDeltaHedgingAgent2,
    "GeometricAsianNumericalDeltaHedgingAgent": GeometricAsianNumericalDeltaHedgingAgent,
    "QuantlibAsianGeometricAgent": QuantlibAsianGeometricAgent,
    "ArithmeticAsianMonteCarloAgent": ArithmeticAsianMonteCarloAgent,
    "ArithmeticAsianControlVariateAgent": ArithmeticAsianControlVariateAgent,
    "MonteCarloAgent": MonteCarloAgent,
}


def _normalize_path_separators(path: str) -> str:
    return str(path).replace("\\", "/")


def get_config_relative_stem(config_path: str) -> str:
    """
    Convert an absolute/relative config path into an organized relative stem.

    Examples:
    - configs/runs/A1/euro_gbm/wavenet/train/seen.json
      -> A1/euro_gbm/wavenet/train/seen
    - thesis_result1b_configs/train/foo.json
      -> thesis_result1b/train/foo
    """
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

    # Normalize legacy roots so they stay grouped under a stable namespace.
    if rel_no_ext.startswith("train/") or rel_no_ext.startswith("compare/"):
        rel_no_ext = f"thesis_result1b/{rel_no_ext}"
    elif rel_no_ext.startswith("option_market_compare/"):
        rel_no_ext = f"thesis_result1b/{rel_no_ext}"

    return rel_no_ext

CLAIMS = {
    "EuropeanCall": EuropeanCall,
    "EuropeanPut": EuropeanPut,
    "AsianGeometricCall": AsianGeometricCall,
    "AsianGeometricPut": AsianGeometricPut,
    "AsianArithmeticCall": AsianArithmeticCall,
    "AsianArithmeticPut": AsianArithmeticPut,
}


def normalize_config_name(user_input: str) -> str:
    name = str(user_input).strip()
    if not name:
        raise ValueError("Config file name cannot be empty.")
    if os.path.basename(name) != name:
        raise ValueError("Provide only the filename (no directories).")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


def load_config_by_name(configs_dir: str, config_name: str | None, prompt_label: str) -> tuple[str, str, dict[str, Any]]:
    if config_name is None:
        raw = input(prompt_label).strip()
    else:
        raw = str(config_name).strip()
    filename = normalize_config_name(raw)
    path = os.path.join(configs_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object.")
    run_name = str(cfg.get("run_name") or os.path.splitext(filename)[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")
    return run_name, path, cfg


def strict_validate_keys(
    config: dict[str, Any],
    required_keys: set[str],
    optional_keys: set[str] | None = None,
) -> None:
    optional_keys = optional_keys or set()
    allowed_keys = set(required_keys) | set(optional_keys)
    config_keys = set(config.keys())
    missing = sorted(required_keys - config_keys)
    extra = sorted(config_keys - allowed_keys)
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if extra:
        raise ValueError(f"Unknown keys (not allowed): {extra}")


def validate_common_market_fields(config: dict[str, Any]) -> None:
    if int(config["schema_version"]) != 1:
        raise ValueError("schema_version must be 1.")
    if str(config["model_family"]).strip().lower() != "deep_hedging_result_1":
        raise ValueError("model_family must be 'deep_hedging_result_1'.")
    if int(config["n"]) <= 0:
        raise ValueError("n must be > 0.")
    if int(config["trading_days_per_year"]) <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    if float(config["s0"]) <= 0:
        raise ValueError("s0 must be > 0.")
    if float(config["sigma"]) <= 0:
        raise ValueError("sigma must be > 0.")
    if float(config["strike"]) <= 0:
        raise ValueError("strike must be > 0.")
    if not isinstance(config["claim_underlying_index"], int) or int(config["claim_underlying_index"]) < 0:
        raise ValueError("claim_underlying_index must be a non-negative integer.")
    if config["fixing_indices"] is not None:
        if not isinstance(config["fixing_indices"], list):
            raise ValueError("fixing_indices must be null or a list of integers.")
        if len(config["fixing_indices"]) == 0:
            raise ValueError("fixing_indices cannot be an empty list.")
        for idx in config["fixing_indices"]:
            if not isinstance(idx, int):
                raise ValueError("fixing_indices must contain integers only.")
            if idx < 0 or idx > int(config["n"]):
                raise ValueError(f"Each fixing index must be in [0, {int(config['n'])}].")
    if str(config["contingent_claim"]) not in CLAIMS:
        raise ValueError(f"Unknown contingent_claim '{config['contingent_claim']}'. Allowed: {list(CLAIMS.keys())}")
    if "global_random_seed" in config and not isinstance(config["global_random_seed"], int):
        raise ValueError("global_random_seed must be an integer.")
    if not isinstance(config["description"], str) or not config["description"].strip():
        raise ValueError("description must be a non-empty string.")


def validate_agent_name(agent_name: str) -> None:
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent_name '{agent_name}'. Allowed: {list(AGENTS.keys())}")


def set_global_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_instrument_from_config(config: dict[str, Any]) -> GBMStock:
    n = int(config["n"])
    trading_days = int(config["trading_days_per_year"])
    t = n / float(trading_days)
    sigma_mode = str(config.get("gbm_sigma_per_path_mode", "fixed")).strip().lower()
    kwargs: dict[str, Any] = {
        "sigma_per_path_mode": sigma_mode,
    }
    if sigma_mode == "uniform":
        kwargs["sigma_uniform_low"] = float(config["gbm_sigma_uniform_low"])
        kwargs["sigma_uniform_high"] = float(config["gbm_sigma_uniform_high"])
    elif sigma_mode == "discrete":
        kwargs["sigma_discrete_values"] = [float(v) for v in config["gbm_sigma_discrete_values"]]
        if config.get("gbm_sigma_discrete_probs") is not None:
            kwargs["sigma_discrete_probs"] = [float(p) for p in config["gbm_sigma_discrete_probs"]]
    return GBMStock(
        S0=float(config["s0"]),
        T=t,
        N=n,
        r=float(config["r"]),
        sigma=float(config["sigma"]),
        **kwargs,
    )


def build_claim_from_config(config: dict[str, Any]):
    claim_cls = CLAIMS[str(config["contingent_claim"])]
    return claim_cls(
        strike=float(config["strike"]),
        underlying_index=int(config["claim_underlying_index"]),
        fixing_indices=config["fixing_indices"],
    )


def build_agent_from_config(
    agent_name: str,
    instrument,
    claim,
    config: dict[str, Any],
):
    validate_agent_name(agent_name)
    agent_cls = AGENTS[agent_name]
    init_signature = inspect.signature(agent_cls.__init__)
    init_params = init_signature.parameters

    if getattr(agent_cls, "is_trainable", False):
        use_context = bool(config.get("use_price_history_context", False))
        context_for_paths_only = bool(config.get("context_for_path_generation_only", False))
        context_length = int(config.get("context_length", 0)) if use_context else 0
        history_conv1d_enabled = bool(config.get("history_conv1d_enabled", False))
        history_conv1d_layers = config.get("history_conv1d_layers")
        history_conv1d_pooling = str(config.get("history_conv1d_pooling", "global_max")).strip().lower()
        sequence_context_agents = {"LSTMAgent", "GRUAgent", "WaveNetAgent"}
        context_as_timesteps = bool(
            use_context
            and (not context_for_paths_only)
            and agent_name in sequence_context_agents
        )
        # Conv1D encoder consumes seen context features. For sequence agents this
        # requires feature mode (not temporal-prefix mode) to avoid bypassing
        # history_features in process_batch.
        if history_conv1d_enabled and context_as_timesteps:
            context_as_timesteps = False
        context_pre_ttm_mode = str(config.get("context_pre_ttm_mode", "calculated")).strip().lower()
        if context_pre_ttm_mode == "extended":
            context_pre_ttm_mode = "calculated"
        if context_pre_ttm_mode not in {"calculated", "zero"}:
            raise ValueError("context_pre_ttm_mode must be 'calculated' or 'zero'.")
        if context_for_paths_only:
            history_feature_dim = 0
        else:
            raw_history_dim = 0 if context_as_timesteps else context_length
            if history_conv1d_enabled:
                if not isinstance(history_conv1d_layers, list) or len(history_conv1d_layers) == 0:
                    raise ValueError(
                        "history_conv1d_layers must be a non-empty list when history_conv1d_enabled=true."
                    )
                last_layer = history_conv1d_layers[-1]
                if not isinstance(last_layer, dict) or "filters" not in last_layer:
                    raise ValueError(
                        "history_conv1d_layers last layer must include 'filters'."
                    )
                history_feature_dim = int(last_layer["filters"])
            else:
                history_feature_dim = int(raw_history_dim)

        path_transformation_type = str(config.get("path_transformation_type", "log_moneyness")).strip().lower()
        if path_transformation_type not in {"none", "log", "log_moneyness"}:
            raise ValueError(
                "path_transformation_type must be one of {'none','log','log_moneyness'}."
            )
        if path_transformation_type == "none":
            path_cfg = [{"transformation_type": None}]
        elif path_transformation_type == "log":
            path_cfg = [{"transformation_type": "log"}]
        else:
            path_cfg = [{"transformation_type": "log_moneyness", "K": float(config["strike"])}]

        include_log_strike_feature = bool(config.get("include_log_strike_feature", False))
        strike_feature_dim = 1 if include_log_strike_feature else 0

        kwargs = {
            "path_transformation_configs": path_cfg
        }
        if "n_hedging_timesteps" in init_params:
            kwargs["n_hedging_timesteps"] = int(config["n"])
        if "n_instruments" in init_params:
            kwargs["n_instruments"] = 1
        if "num_filters" in init_params and "wavenet_num_filters" in config:
            kwargs["num_filters"] = int(config["wavenet_num_filters"])
        if "num_residual_blocks" in init_params and "wavenet_num_residual_blocks" in config:
            kwargs["num_residual_blocks"] = int(config["wavenet_num_residual_blocks"])
        if "history_feature_dim" in init_params:
            kwargs["history_feature_dim"] = int(history_feature_dim + strike_feature_dim)
        if "context_as_timesteps" in init_params:
            kwargs["context_as_timesteps"] = bool(context_as_timesteps)
        if "context_pre_ttm_mode" in init_params:
            kwargs["context_pre_ttm_mode"] = str(context_pre_ttm_mode)
        agent = agent_cls(**kwargs)
        # Optional constant feature appended to every timestep/input row.
        agent.append_log_strike_feature = bool(include_log_strike_feature)
        agent.log_strike_value = float(np.log(max(float(config["strike"]), 1e-8)))
        if hasattr(agent, "configure_history_conv1d_encoder"):
            agent.configure_history_conv1d_encoder(
                enabled=history_conv1d_enabled,
                layers_config=history_conv1d_layers,
                pooling=history_conv1d_pooling,
            )
        return agent

    no_intervention_bound = config.get(
        "benchmark_no_intervention_bound",
        config.get("no_intervention_bound", config.get("benchmark_no_trade_band", 0.0)),
    )
    no_intervention_mode = str(
        config.get(
            "benchmark_no_intervention_mode",
            config.get("no_intervention_mode", "absolute"),
        )
    ).strip().lower()
    if no_intervention_mode not in {"absolute", "percentage"}:
        raise ValueError(
            "benchmark_no_intervention_mode/no_intervention_mode must be 'absolute' or 'percentage'."
        )

    candidate_kwargs = {
        "bump_size": float(config.get("benchmark_bump_size", 0.001)),
        "num_simulations": int(config.get("benchmark_num_simulations", 10_000)),
        "seed": int(config.get("benchmark_seed", 33)),
        "no_trade_band": float(no_intervention_bound),
        "mc_use_vectorized": bool(config.get("benchmark_mc_use_vectorized", True)),
        "mc_chunk_size": int(config.get("benchmark_mc_chunk_size", 64)),
        "mc_parallel_enabled": bool(config.get("benchmark_mc_parallel_enabled", False)),
        "mc_n_workers": int(config.get("benchmark_mc_n_workers", 1)),
        "mc_parallel_backend": str(config.get("benchmark_mc_parallel_backend", "thread")),
        "mc_state_chunk_size": int(config.get("benchmark_mc_state_chunk_size", 64)),
        "mc_seed_mode": str(config.get("benchmark_mc_seed_mode", "shared_crn")),
        "parallel_enabled": bool(config.get("benchmark_mc_parallel_enabled", False)),
        "n_workers": int(config.get("benchmark_mc_n_workers", 1)),
        "parallel_backend": str(config.get("benchmark_mc_parallel_backend", "thread")),
        "parallel_chunk_size": (
            None
            if config.get("benchmark_mc_parallel_chunk_size", None) is None
            else int(config.get("benchmark_mc_parallel_chunk_size"))
        ),
        "parallel_min_states": int(config.get("benchmark_mc_parallel_min_states", 128)),
    }
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in init_params}
    agent = agent_cls(instrument, claim, **kwargs)
    if hasattr(agent, "set_no_trade_band_mode"):
        agent.set_no_trade_band_mode(no_intervention_mode)
    return agent


def build_risk_measure_from_config(config: dict[str, Any]):
    raw_name = str(config["risk_measure_name"]).strip()
    name = raw_name.lower().replace("_", "")

    if name == "mse":
        return MSE()
    if name == "mae":
        return MAE()

    if name.startswith("cvar"):
        suffix = name[4:]
        if suffix:
            # Examples accepted: "CVaR95", "cvar99"
            if not suffix.isdigit():
                raise ValueError(
                    "For CVaR shorthand use digits only, e.g. 'CVaR95'."
                )
            alpha = float(suffix) / 100.0
        else:
            alpha = float(config["cvar_alpha"])
        if not (0.0 < alpha < 1.0):
            raise ValueError("CVaR alpha must satisfy 0 < alpha < 1.")
        return CVaR(alpha=alpha)

    raise ValueError(
        "Unsupported risk_measure_name. Allowed: 'CVaR', 'CVaR95', 'MSE', 'MAE'."
    )


def build_environment(
    agent,
    instrument,
    claim,
    proportional_cost: float,
    risk_measure,
    n_epochs: int,
    batch_size: int,
    learning_rate_schedule,
    optimizer_cls,
    resample_each_epoch: bool,
    train_seed: int,
    use_price_history_context: bool = False,
    context_length: int = 0,
    context_feature_mode: str = "log_returns",
    context_visible_to_agent: bool = True,
):
    return Environment(
        agent=agent,
        T=float(instrument.T),
        N=int(instrument.N),
        r=float(instrument.r),
        instrument_list=[instrument],
        n_instruments=1,
        contingent_claim=claim,
        cost_function=ProportionalCost(float(proportional_cost)),
        risk_measure=risk_measure,
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        learning_rate=learning_rate_schedule,
        optimizer=optimizer_cls,
        resample_each_epoch=bool(resample_each_epoch),
        train_random_seed=int(train_seed),
        history_context_length=int(context_length) if bool(use_price_history_context) else 0,
        history_feature_mode=str(context_feature_mode),
        history_context_visible_to_agent=bool(context_visible_to_agent),
    )


def model_and_optimizer_paths(agent, model_name: str, models_dir: str, optimizers_dir: str) -> tuple[str, str]:
    # Keep args for backward compatibility with existing callers/configs,
    # but persist all artifacts under thesis root.
    _ = models_dir
    _ = optimizers_dir
    model_path = os.path.join(THESIS_MODELS_ROOT, "models", agent.name, f"{model_name}.keras")
    optimizer_path = os.path.join(THESIS_MODELS_ROOT, "optimizers", agent.name, model_name)
    return model_path, optimizer_path


def ensure_run_dirs(output_root: str, run_type: str, run_name: str) -> dict[str, str]:
    # Keep output_root arg for compatibility, but all run artifacts go to thesis root.
    _ = output_root
    run_dir = os.path.join(THESIS_MODELS_ROOT, "results", run_type, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    plots_dir = os.path.join(run_dir, "plots")
    tables_dir = os.path.join(run_dir, "tables")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return {
        "storage_root": THESIS_MODELS_ROOT,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "tables_dir": tables_dir,
    }


def copy_config_snapshot(config_path: str, dst_dir: str) -> str:
    dst_path = os.path.join(dst_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst_path)
    return dst_path


def bootstrap_statistics_list():
    return [Mean(), StdDev(), CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase()]
