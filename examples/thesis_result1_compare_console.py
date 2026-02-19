"""
Console runner for thesis Result 1 comparisons.

Loads benchmark + trained agents from JSON config, creates pairwise
benchmark-vs-agent plots, and computes bootstrap confidence intervals for
thesis-style metric tables.
"""

import argparse
import json
import os
import sys
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
    bootstrap_statistics_list,
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
    validate_agent_name,
    validate_common_market_fields,
)
from DeepHedging.RiskMeasures import CVaR, MAE, WorstCase


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
        "r",
        "sigma",
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
    }


def optional_keys() -> set[str]:
    return {
        "benchmark_no_trade_band",
        "no_intervention_bound",
        "benchmark_no_intervention_bound",
        "no_intervention_mode",
        "benchmark_no_intervention_mode",
        "eval_agent_batch_size",
        "terminal_progress_log_every_agent_batches",
    }


def validate_config(config: dict[str, Any]) -> None:
    strict_validate_keys(config, required_keys(), optional_keys())
    validate_common_market_fields(config)

    validate_agent_name(str(config["benchmark_agent_name"]))
    if str(config["benchmark_agent_name"]) in TRAINABLE_AGENT_NAMES:
        raise ValueError("benchmark_agent_name must be a non-trainable benchmark agent.")

    if float(config["proportional_cost"]) < 0.0:
        raise ValueError("proportional_cost must be >= 0.")
    if "benchmark_no_trade_band" in config:
        val = config["benchmark_no_trade_band"]
        if val is not None and (not isinstance(val, (int, float)) or float(val) < 0.0):
            raise ValueError("benchmark_no_trade_band must be null or a number >= 0.")
    if "no_intervention_bound" in config:
        val = config["no_intervention_bound"]
        if val is not None and (not isinstance(val, (int, float)) or float(val) < 0.0):
            raise ValueError("no_intervention_bound must be null or a number >= 0.")
    if "benchmark_no_intervention_bound" in config:
        val = config["benchmark_no_intervention_bound"]
        if val is not None and (not isinstance(val, (int, float)) or float(val) < 0.0):
            raise ValueError("benchmark_no_intervention_bound must be null or a number >= 0.")
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
    effective_bound = config.get(
        "benchmark_no_intervention_bound",
        config.get("no_intervention_bound", config.get("benchmark_no_trade_band", 0.0)),
    )
    if effective_mode == "percentage" and effective_bound is not None:
        bound_float = float(effective_bound)
        if not (0.0 <= bound_float <= 1.0):
            raise ValueError(
                "For percentage mode, no-intervention bound must be in [0, 1]. "
                "Example: 0.05 = 5%."
            )
    if int(config["eval_paths"]) <= 0:
        raise ValueError("eval_paths must be > 0.")
    if "eval_agent_batch_size" in config and config["eval_agent_batch_size"] is not None:
        if int(config["eval_agent_batch_size"]) <= 0:
            raise ValueError("eval_agent_batch_size must be null or > 0.")
    if "terminal_progress_log_every_agent_batches" in config and config["terminal_progress_log_every_agent_batches"] is not None:
        if int(config["terminal_progress_log_every_agent_batches"]) <= 0:
            raise ValueError("terminal_progress_log_every_agent_batches must be null or > 0.")
    if not isinstance(config["eval_seed"], int):
        raise ValueError("eval_seed must be int.")
    if str(config["pricing_method"]) not in {"fixed", "individual"}:
        raise ValueError("pricing_method must be 'fixed' or 'individual'.")
    if not isinstance(config["language"], str) or str(config["language"]) not in {"en", "es"}:
        raise ValueError("language must be 'en' or 'es'.")
    if float(config["plot_min_x"]) >= float(config["plot_max_x"]):
        raise ValueError("plot_min_x must be < plot_max_x.")

    if not isinstance(config["bootstrap_enabled"], bool):
        raise ValueError("bootstrap_enabled must be bool.")
    if int(config["bootstrap_n_bootstraps"]) <= 0:
        raise ValueError("bootstrap_n_bootstraps must be > 0.")
    if int(config["bootstrap_batch_size"]) <= 0:
        raise ValueError("bootstrap_batch_size must be > 0.")
    if not (0.0 < float(config["bootstrap_confidence_level"]) < 1.0):
        raise ValueError("bootstrap_confidence_level must satisfy 0 < c < 1.")

    if not isinstance(config["trained_agents"], list) or len(config["trained_agents"]) == 0:
        raise ValueError("trained_agents must be a non-empty list.")
    for i, item in enumerate(config["trained_agents"]):
        if not isinstance(item, dict):
            raise ValueError(f"trained_agents[{i}] must be an object.")
        if set(item.keys()) != {"agent_name", "model_name"}:
            raise ValueError(
                f"trained_agents[{i}] must contain exactly keys ['agent_name','model_name']."
            )
        name = str(item["agent_name"])
        validate_agent_name(name)
        if name not in TRAINABLE_AGENT_NAMES:
            raise ValueError(f"trained_agents[{i}].agent_name must be trainable. Got {name}.")
        if not isinstance(item["model_name"], str) or not item["model_name"].strip():
            raise ValueError(f"trained_agents[{i}].model_name must be a non-empty string.")

    for key in ["output_root", "run_name", "models_dir", "optimizers_dir"]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")


def _display_name(agent, language: str) -> str:
    plot_name = getattr(agent, "plot_name", None)
    if isinstance(plot_name, dict):
        return str(plot_name.get(language, agent.name))
    if isinstance(plot_name, str):
        return plot_name
    return str(getattr(agent, "name", "unknown_agent"))


def _bootstrap_to_thesis_table(df_bootstrap: pd.DataFrame, language: str) -> pd.DataFrame:
    metric_order = ["Mean", "StdDev", "CVaR_50", "CVaR_95", "CVaR_99", "WorstCase", "MAE"]
    rows = []
    for _, row in df_bootstrap.iterrows():
        agent = str(row["Agent"])
        for metric in metric_order:
            point = float(row[f"{metric}_point_estimate"])
            low = float(row[f"{metric}_ci_lower"])
            high = float(row[f"{metric}_ci_upper"])
            rows.append(
                {
                    "Agent": agent,
                    "Metric": metric,
                    "PointEstimate": point,
                    "CI_Lower": low,
                    "CI_Upper": high,
                    "Formatted": f"{point:.4f} [{low:.4f}; {high:.4f}]",
                }
            )
    out = pd.DataFrame(rows)
    out["Metric"] = pd.Categorical(out["Metric"], categories=metric_order, ordered=True)
    out = out.sort_values(["Metric", "Agent"]).reset_index(drop=True)
    return out


def run_comparison(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    run_dirs = ensure_run_dirs(
        output_root=str(config["output_root"]),
        run_type="compare",
        run_name=run_name,
    )
    copied_cfg = copy_config_snapshot(config_path, run_dirs["run_dir"])
    print(f"[run:{run_name}] Config validated and copied to: {copied_cfg}")
    print(f"[run:{run_name}] Storage root: {THESIS_MODELS_ROOT}")

    set_global_determinism(int(config["eval_seed"]))
    plot_language = "es"
    eval_agent_batch_size = config.get("eval_agent_batch_size")
    terminal_progress_every = config.get("terminal_progress_log_every_agent_batches")

    instrument = build_instrument_from_config(config)
    claim = build_claim_from_config(config)
    risk_measure = build_risk_measure_from_config(config)

    benchmark_agent = build_agent_from_config(
        agent_name=str(config["benchmark_agent_name"]),
        instrument=instrument,
        claim=claim,
        config=config,
    )

    trained_agents = []
    for item in config["trained_agents"]:
        agent = build_agent_from_config(
            agent_name=str(item["agent_name"]),
            instrument=instrument,
            claim=claim,
            config=config,
        )
        model_path, _ = model_and_optimizer_paths(
            agent=agent,
            model_name=str(item["model_name"]),
            models_dir=str(config["models_dir"]),
            optimizers_dir=str(config["optimizers_dir"]),
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found for trained agent {item['agent_name']} at: {model_path}"
            )
        agent.load_model(model_path)
        print(f"[run:{run_name}] Loaded model: {model_path}")
        trained_agents.append(agent)

    all_agents = [benchmark_agent] + trained_agents
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
        train_seed=int(config["eval_seed"]),
    )

    # Pairwise plots as in thesis (benchmark vs each trained agent).
    loss_fns = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase()]
    pairwise_rows = []
    for agent in trained_agents:
        pair_name = f"{benchmark_agent.name}_vs_{agent.name}"
        plot_path = os.path.join(run_dirs["plots_dir"], f"{pair_name}.jpg")
        stats_path = os.path.join(run_dirs["tables_dir"], f"{pair_name}.xlsx")
        pair_df = env.terminal_hedging_error_multiple_agents(
            agents=[benchmark_agent, agent],
            n_paths=int(config["eval_paths"]),
            random_seed=int(config["eval_seed"]),
            plot_error=True,
            plot_title="Error de Cobertura Terminal",
            save_plot_path=plot_path,
            save_stats_path=stats_path,
            loss_functions=loss_fns,
            min_x=float(config["plot_min_x"]),
            max_x=float(config["plot_max_x"]),
            language=plot_language,
            pricing_method=str(config["pricing_method"]),
            agent_eval_batch_size=int(eval_agent_batch_size) if eval_agent_batch_size is not None else None,
            progress_log_every_agent_batches=int(terminal_progress_every) if terminal_progress_every is not None else 5,
        )
        pair_df.insert(0, "pair_name", pair_name)
        pairwise_rows.append(pair_df)
        print(f"[run:{run_name}] Saved pair plot: {plot_path}")
        print(f"[run:{run_name}] Saved pair stats: {stats_path}")

    if pairwise_rows:
        pairwise_all = pd.concat(pairwise_rows, ignore_index=True)
    else:
        pairwise_all = pd.DataFrame()
    pairwise_csv = os.path.join(run_dirs["tables_dir"], "pairwise_terminal_stats.csv")
    pairwise_all.to_csv(pairwise_csv, index=False)
    print(f"[run:{run_name}] Saved pairwise table: {pairwise_csv}")

    # Point estimates over all agents in one shot.
    mean_errors, std_errors, loss_results = env.terminal_hedging_error_multiple_agents(
        agents=all_agents,
        n_paths=int(config["eval_paths"]),
        random_seed=int(config["eval_seed"]),
        plot_error=False,
        loss_functions=loss_fns,
        min_x=float(config["plot_min_x"]),
        max_x=float(config["plot_max_x"]),
        language=plot_language,
        pricing_method=str(config["pricing_method"]),
        agent_eval_batch_size=int(eval_agent_batch_size) if eval_agent_batch_size is not None else None,
        progress_log_every_agent_batches=int(terminal_progress_every) if terminal_progress_every is not None else 5,
    )
    point_rows = []
    for i, agent in enumerate(all_agents):
        row = {
            "Agent": _display_name(agent, str(config["language"])),
            "Mean": float(mean_errors[i]),
            "StdDev": float(std_errors[i]),
        }
        for key, values in (loss_results or {}).items():
            row[key] = float(values[i])
        point_rows.append(row)
    point_df = pd.DataFrame(point_rows)
    point_csv = os.path.join(run_dirs["tables_dir"], "point_metrics.csv")
    point_df.to_csv(point_csv, index=False)
    print(f"[run:{run_name}] Saved point metrics: {point_csv}")

    if bool(config["bootstrap_enabled"]):
        boot_df = env.bootstrap_confidence_intervals(
            agents=all_agents,
            statistics=bootstrap_statistics_list(),
            n_paths=int(config["eval_paths"]),
            n_bootstraps=int(config["bootstrap_n_bootstraps"]),
            confidence_level=float(config["bootstrap_confidence_level"]),
            random_seed=int(config["eval_seed"]),
            plot_histograms=False,
            language=plot_language,
            pricing_method=str(config["pricing_method"]),
            batch_size=int(config["bootstrap_batch_size"]),
        )
        boot_csv = os.path.join(run_dirs["tables_dir"], "bootstrap_metrics_wide.csv")
        boot_df.to_csv(boot_csv, index=False)
        print(f"[run:{run_name}] Saved bootstrap wide table: {boot_csv}")

        thesis_df = _bootstrap_to_thesis_table(boot_df, language=str(config["language"]))
        thesis_csv = os.path.join(run_dirs["tables_dir"], "bootstrap_metrics_thesis_table.csv")
        thesis_df.to_csv(thesis_csv, index=False)
        print(f"[run:{run_name}] Saved bootstrap thesis table: {thesis_csv}")

    meta = {
        "run_name": run_name,
        "benchmark_agent": str(config["benchmark_agent_name"]),
        "trained_agents": config["trained_agents"],
        "eval_paths": int(config["eval_paths"]),
        "bootstrap_enabled": bool(config["bootstrap_enabled"]),
        "bootstrap_n_bootstraps": int(config["bootstrap_n_bootstraps"]),
        "bootstrap_confidence_level": float(config["bootstrap_confidence_level"]),
    }
    meta_path = os.path.join(run_dirs["run_dir"], "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[run:{run_name}] Saved run metadata: {meta_path}")
    print(f"[run:{run_name}] Completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare thesis Result 1 agents from JSON config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional config filename in ./thesis_result1_configs/compare (extension optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1_configs", "compare")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, cfg_path, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter COMPARE JSON config name from 'thesis_result1_configs/compare' (example: european_no_cost_compare.json): ",
    )
    print(f"[run:{run_name}] Loaded config: {cfg_path}")
    validate_config(cfg)
    run_comparison(run_name=run_name, config_path=cfg_path, config=cfg)


if __name__ == "__main__":
    main()
