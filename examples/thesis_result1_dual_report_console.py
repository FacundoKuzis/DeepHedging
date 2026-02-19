"""
Builds a side-by-side report from two comparison JSON configs.

Usage examples:
    python examples/thesis_result1_dual_report_console.py european_no_cost_compare european_cost_1pct_compare
    python examples/thesis_result1_dual_report_console.py
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.thesis_result1_common import THESIS_MODELS_ROOT

def _normalize_config_name(user_input: str) -> str:
    name = str(user_input).strip()
    if not name:
        raise ValueError("Config file name cannot be empty.")
    if os.path.basename(name) != name:
        raise ValueError("Provide only the filename (no directories).")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


def _load_compare_config(config_name: str | None, prompt_label: str) -> tuple[str, str, dict]:
    configs_dir = os.path.join(os.getcwd(), "thesis_result1_configs", "compare")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")
    if config_name is None:
        raw = input(prompt_label).strip()
    else:
        raw = str(config_name).strip()
    filename = _normalize_config_name(raw)
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


def _load_metrics_table(cfg: dict) -> pd.DataFrame:
    run_name = str(cfg["run_name"])
    table_path = os.path.join(
        THESIS_MODELS_ROOT,
        "results",
        "compare",
        run_name,
        "tables",
        "bootstrap_metrics_thesis_table.csv",
    )
    if not os.path.isfile(table_path):
        raise FileNotFoundError(
            f"Missing metrics table for run '{run_name}'. Expected: {table_path}\n"
            "Run thesis_result1_compare_console.py first."
        )
    df = pd.read_csv(table_path)
    df.insert(0, "run_name", run_name)
    return df


def _plot_metric_panels(df: pd.DataFrame, out_path: str) -> None:
    metrics = ["Mean", "StdDev", "CVaR_50", "CVaR_95", "MAE", "WorstCase"]
    run_names = list(df["run_name"].dropna().unique())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        d = df[df["Metric"] == metric].copy()
        if d.empty:
            ax.set_visible(False)
            continue
        pivot = d.pivot_table(
            index="Agent",
            columns="run_name",
            values="PointEstimate",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=run_names)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Point Estimate")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title="Scenario")

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle("Thesis Result 1 Comparison (Two Scenarios)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dual-scenario report from two thesis comparison JSON configs."
    )
    parser.add_argument("config_a", nargs="?", help="First compare config filename.")
    parser.add_argument("config_b", nargs="?", help="Second compare config filename.")
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional output report name. Default: <runA>__vs__<runB>.",
    )
    args = parser.parse_args()

    run_a, path_a, cfg_a = _load_compare_config(
        config_name=args.config_a,
        prompt_label="Enter FIRST compare JSON name from 'thesis_result1_configs/compare': ",
    )
    run_b, path_b, cfg_b = _load_compare_config(
        config_name=args.config_b,
        prompt_label="Enter SECOND compare JSON name from 'thesis_result1_configs/compare': ",
    )

    print(f"[report] Loaded A: {path_a}")
    print(f"[report] Loaded B: {path_b}")

    df_a = _load_metrics_table(cfg_a)
    df_b = _load_metrics_table(cfg_b)
    combined = pd.concat([df_a, df_b], ignore_index=True)

    report_name = args.report_name or f"{run_a}__vs__{run_b}"
    report_dir = os.path.join(THESIS_MODELS_ROOT, "results", "reports", report_name)
    os.makedirs(report_dir, exist_ok=True)

    combined_csv = os.path.join(report_dir, "combined_bootstrap_metrics.csv")
    combined.to_csv(combined_csv, index=False)

    panel_path = os.path.join(report_dir, "metric_panels_two_scenarios.jpg")
    _plot_metric_panels(combined, panel_path)

    pivot_fmt = combined.pivot_table(
        index=["Metric", "Agent"],
        columns="run_name",
        values="Formatted",
        aggfunc="first",
    ).reset_index()
    pivot_csv = os.path.join(report_dir, "thesis_style_table_two_scenarios.csv")
    pivot_fmt.to_csv(pivot_csv, index=False)

    print(f"[report] Saved combined metrics: {combined_csv}")
    print(f"[report] Saved thesis-style table: {pivot_csv}")
    print(f"[report] Saved panel plot: {panel_path}")
    print("[report] Completed.")


if __name__ == "__main__":
    main()
