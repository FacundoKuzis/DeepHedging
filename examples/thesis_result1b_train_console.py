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
    build_risk_measure_from_config,
    copy_config_snapshot,
    load_config_by_name,
    set_global_determinism,
    strict_validate_keys,
    validate_agent_name,
)
from DeepHedging.HedgingInstruments import GBMStock  # noqa: E402
from DeepHedging.utils.gbm_calibration import calibrate_gbm_from_market_data  # noqa: E402


RESULT1B_ROOT = os.path.join(THESIS_MODELS_ROOT, "thesis_result1b")


def _run_dirs(run_name: str) -> dict[str, str]:
    run_dir = os.path.join(RESULT1B_ROOT, "train", run_name)
    logs_dir = os.path.join(run_dir, "logs")
    plots_dir = os.path.join(run_dir, "plots")
    tables_dir = os.path.join(run_dir, "tables")
    models_root = os.path.join(RESULT1B_ROOT, "models")
    optimizers_root = os.path.join(RESULT1B_ROOT, "optimizers")
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
    return set()



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

    if str(config["learning_rate_strategy"]).strip().lower() not in {"exponential_decay", "constant"}:
        raise ValueError("learning_rate_strategy must be 'exponential_decay' or 'constant'.")

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
    dirs = _run_dirs(run_name)
    cfg_snapshot = copy_config_snapshot(config_path, dirs["run_dir"])
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

    n = int(config["n"])
    trading_days = int(config["trading_days_per_year"])
    instrument = GBMStock(
        S0=float(config["s0"]),
        T=float(n / float(trading_days)),
        N=n,
        r=float(calib.r_train),
        sigma=float(calib.sigma_train),
    )

    cfg_for_builders = dict(config)
    cfg_for_builders["r"] = float(calib.r_train)
    cfg_for_builders["sigma"] = float(calib.sigma_train)

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
        f"with calibrated r={calib.r_train:.6f}, sigma={calib.sigma_train:.6f}"
    )
    t0 = time.perf_counter()
    out = env.train(
        train_paths=int(config["train_paths"]),
        val_paths=int(config["val_paths"]),
        random_seed=seed,
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
