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

THESIS_MODELS_ROOT = os.path.normpath(r"G:\Mi unidad\Tesis2026\Models")

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
    return GBMStock(
        S0=float(config["s0"]),
        T=t,
        N=n,
        r=float(config["r"]),
        sigma=float(config["sigma"]),
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
        kwargs = {
            "path_transformation_configs": [{"transformation_type": "log_moneyness", "K": float(config["strike"])}]
        }
        if "n_hedging_timesteps" in init_params:
            kwargs["n_hedging_timesteps"] = int(config["n"])
        if "n_instruments" in init_params:
            kwargs["n_instruments"] = 1
        if "num_filters" in init_params and "wavenet_num_filters" in config:
            kwargs["num_filters"] = int(config["wavenet_num_filters"])
        if "num_residual_blocks" in init_params and "wavenet_num_residual_blocks" in config:
            kwargs["num_residual_blocks"] = int(config["wavenet_num_residual_blocks"])
        return agent_cls(**kwargs)

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
    }
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in init_params}
    agent = agent_cls(instrument, claim, **kwargs)
    if hasattr(agent, "set_no_trade_band_mode"):
        agent.set_no_trade_band_mode(no_intervention_mode)
    return agent


def build_risk_measure_from_config(config: dict[str, Any]):
    name = str(config["risk_measure_name"]).strip().lower()
    if name != "cvar":
        raise ValueError("Only risk_measure_name='CVaR' is supported in thesis result 1 runner.")
    alpha = float(config["cvar_alpha"])
    if not (0.0 < alpha < 1.0):
        raise ValueError("cvar_alpha must satisfy 0 < cvar_alpha < 1.")
    return CVaR(alpha=alpha)


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
