"""
Consolidated report builder for thesis Result 1b compare runs.
"""

import argparse
import json
import os

import pandas as pd


THESIS_MODELS_ROOT = os.path.normpath(r"G:\Mi unidad\Tesis2026\Models")
RESULT1B_ROOT = os.path.join(THESIS_MODELS_ROOT, "thesis_result1b")



def _read_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None



def _normalize_config_name(user_input: str) -> str:
    name = str(user_input).strip()
    if not name:
        raise ValueError("Config file name cannot be empty.")
    if os.path.basename(name) != name:
        raise ValueError("Provide only the filename (no directories).")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name



def _load_run_name_from_config(configs_dir: str, config_name: str) -> tuple[str, str]:
    filename = _normalize_config_name(config_name)
    cfg_path = os.path.join(configs_dir, filename)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run_name = str(cfg.get("run_name") or os.path.splitext(filename)[0]).strip()
    if not run_name:
        raise ValueError("run_name cannot be empty.")
    return run_name, filename



def _resolve_run_names(configs_dir: str, requested: list[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if requested:
        for item in requested:
            run_name, filename = _load_run_name_from_config(configs_dir=configs_dir, config_name=item)
            pairs.append((run_name, filename))
        return pairs

    for filename in sorted(os.listdir(configs_dir)):
        if not filename.endswith(".json"):
            continue
        run_name, _ = _load_run_name_from_config(configs_dir=configs_dir, config_name=filename)
        pairs.append((run_name, filename))
    return pairs



def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated report for thesis_result1b compare runs.")
    parser.add_argument(
        "config_names",
        nargs="*",
        help="Optional compare config filenames (extension optional). If omitted, all compare configs are used.",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1b_configs", "compare")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    runs = _resolve_run_names(configs_dir=configs_dir, requested=args.config_names)
    if not runs:
        raise ValueError("No compare runs found to report.")

    report_dir = os.path.join(RESULT1B_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    point_all = []
    empirical_all = []
    bootstrap_all = []

    for run_name, _ in runs:
        run_tables = os.path.join(RESULT1B_ROOT, "compare", run_name, "tables")
        point_df = _read_if_exists(os.path.join(run_tables, "point_metrics.csv"))
        empirical_df = _read_if_exists(os.path.join(run_tables, "empirical_risk_metrics.csv"))
        boot_df = _read_if_exists(os.path.join(run_tables, "bootstrap_metrics_wide.csv"))

        if point_df is not None:
            point_df.insert(0, "run_name", run_name)
            point_all.append(point_df)
        if empirical_df is not None:
            empirical_df.insert(0, "run_name", run_name)
            empirical_all.append(empirical_df)
        if boot_df is not None:
            boot_df.insert(0, "run_name", run_name)
            bootstrap_all.append(boot_df)

    if point_all:
        point_concat = pd.concat(point_all, ignore_index=True)
        point_path = os.path.join(report_dir, "consolidated_point_metrics.csv")
        point_concat.to_csv(point_path, index=False)
        print(f"Saved: {point_path}")

    if empirical_all:
        empirical_concat = pd.concat(empirical_all, ignore_index=True)
        empirical_path = os.path.join(report_dir, "consolidated_empirical_risk_metrics.csv")
        empirical_concat.to_csv(empirical_path, index=False)
        print(f"Saved: {empirical_path}")

        rank_df = empirical_concat[["run_name", "Agent", "es_99", "q_0_1pct", "std_error"]].copy()
        rank_df["rank_score"] = rank_df[["es_99", "q_0_1pct", "std_error"]].abs().sum(axis=1)
        rank_df = rank_df.sort_values(["rank_score", "run_name", "Agent"]).reset_index(drop=True)
        rank_path = os.path.join(report_dir, "ranking_by_tail_risk.csv")
        rank_df.to_csv(rank_path, index=False)
        print(f"Saved: {rank_path}")

    if bootstrap_all:
        boot_concat = pd.concat(bootstrap_all, ignore_index=True)
        boot_path = os.path.join(report_dir, "consolidated_bootstrap_metrics.csv")
        boot_concat.to_csv(boot_path, index=False)
        print(f"Saved: {boot_path}")


if __name__ == "__main__":
    main()
