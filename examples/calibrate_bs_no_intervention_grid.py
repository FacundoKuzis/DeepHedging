"""
Grid calibration runner for BS no-intervention band.

What it does:
1) Runs BS-banded compares across the full NI grid.
2) Uses multiple simulations (different eval seeds) per band.
3) Disables bootstrap during calibration runs.
4) Builds summary tables with metrics by band.
5) Updates benchmark_no_intervention_bound in the target compare config.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.compare_console import _apply_relaxed_defaults_compare  # noqa: E402
from examples.thesis_result1_common import THESIS_MODELS_ROOT, get_config_relative_stem  # noqa: E402
from examples.thesis_result1b_compare_console import run_comparison  # noqa: E402
from examples.unified_console_common import (  # noqa: E402
    detect_pipeline,
    load_merged_config,
    resolve_config_path,
    strip_meta_keys,
)


def _metric_key(raw: str) -> str:
    key = str(raw).strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def _run_dir_for_config(config_path: str) -> str:
    rel_stem = get_config_relative_stem(config_path)
    return os.path.normpath(os.path.join(THESIS_MODELS_ROOT, rel_stem))


def _resolve_existing_run_dir(run_ref: str) -> str:
    ref = str(run_ref).strip()
    if not ref:
        raise ValueError("Empty run_ref for benchmark_actions_reuse_from_run.")
    if os.path.isabs(ref) and os.path.isdir(ref):
        return os.path.normpath(ref)
    normalized = os.path.normpath(ref.replace("/", os.sep).replace("\\", os.sep))
    candidate = os.path.normpath(os.path.join(THESIS_MODELS_ROOT, normalized))
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cannot resolve benchmark_actions_reuse_from_run='{run_ref}' under '{THESIS_MODELS_ROOT}'."
    )


def _load_source_run_eval_params(run_dir: str) -> tuple[int | None, int | None]:
    cfg_path = os.path.join(run_dir, "resolved_config.json")
    if not os.path.isfile(cfg_path):
        return None, None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        return None, None
    seed = cfg.get("eval_seed", None)
    n_paths = cfg.get("eval_paths", None)
    seed_out = int(seed) if seed is not None else None
    n_paths_out = int(n_paths) if n_paths is not None else None
    return seed_out, n_paths_out


def _has_core_metrics_outputs(run_dir: str) -> bool:
    point_csv = os.path.join(run_dir, "tables", "point_metrics.csv")
    empirical_csv = os.path.join(run_dir, "tables", "empirical_risk_metrics.csv")
    return os.path.isfile(point_csv) and os.path.isfile(empirical_csv)


def _load_grid_from_calibration_dir(calibration_dir: str) -> list[float]:
    if not os.path.isdir(calibration_dir):
        return []
    pattern = re.compile(r"^band_(\d{4})\.json$", flags=re.IGNORECASE)
    bounds: list[float] = []
    for name in sorted(os.listdir(calibration_dir)):
        match = pattern.match(name)
        if match is None:
            continue
        path = os.path.join(calibration_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict) and "benchmark_no_intervention_bound" in cfg:
            bounds.append(float(cfg["benchmark_no_intervention_bound"]))
        else:
            slug = int(match.group(1))
            bounds.append(float(slug) / 1000.0)
    unique_sorted = sorted({round(b, 10) for b in bounds})
    return unique_sorted


def _make_grid_from_range(min_pct: float, max_pct: float, step_pct: float) -> list[float]:
    if step_pct <= 0:
        raise ValueError("grid_step_pct must be > 0.")
    if min_pct < 0:
        raise ValueError("grid_min_pct must be >= 0.")
    if max_pct < 0:
        raise ValueError("grid_max_pct must be >= 0.")
    if max_pct < min_pct:
        raise ValueError("grid_max_pct must be >= grid_min_pct.")
    values: list[float] = []
    current = float(min_pct)
    while current <= float(max_pct) + 1e-12:
        values.append(round(current / 100.0, 10))
        current += float(step_pct)
    return values


def _bound_slug(bound: float) -> str:
    return f"{int(round(float(bound) * 1000.0)):04d}"


def _extract_agent_metrics(point_df: pd.DataFrame, empirical_df: pd.DataFrame, row_idx: int, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    point_row = point_df.iloc[row_idx].to_dict()
    empirical_row = empirical_df.iloc[row_idx].to_dict()

    if "Agent" in point_row:
        out[f"{prefix}_point_agent"] = str(point_row.pop("Agent"))
    if "Agent" in empirical_row:
        out[f"{prefix}_empirical_agent"] = str(empirical_row.pop("Agent"))

    for key, value in point_row.items():
        col = f"{prefix}_point_{_metric_key(key)}"
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        out[col] = float(numeric) if pd.notna(numeric) else value

    for key, value in empirical_row.items():
        col = f"{prefix}_empirical_{_metric_key(key)}"
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        out[col] = float(numeric) if pd.notna(numeric) else value

    return out


def _save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_config_bound(config_path: str, bound: float) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    cfg["benchmark_no_intervention_bound"] = float(bound)
    cfg.setdefault("benchmark_no_intervention_mode", "percentage")
    cfg["benchmark_actions_reuse_apply_no_intervention"] = True
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BS no-intervention band over a full grid.")
    parser.add_argument(
        "config_ref",
        help="Compare config path/name (normally a *vs_bs_banded*.json).",
    )
    parser.add_argument("--n-sims", type=int, default=5, help="Number of simulations per band.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help=(
            "Base eval seed (sim i uses seed_start + i). "
            "If omitted and benchmark actions are reused, it auto-aligns to source run eval_seed."
        ),
    )
    parser.add_argument("--grid-source", choices=["calibration_configs", "range"], default="calibration_configs")
    parser.add_argument("--grid-min-pct", type=float, default=0.5, help="Range mode min percent.")
    parser.add_argument("--grid-max-pct", type=float, default=10.0, help="Range mode max percent.")
    parser.add_argument("--grid-step-pct", type=float, default=0.5, help="Range mode step percent.")
    parser.add_argument(
        "--objective-col",
        default="benchmark_empirical_cvar_50",
        help="Metric column (from per-simulation table) used to pick best band.",
    )
    parser.add_argument("--objective-direction", choices=["min", "max"], default="min")
    parser.add_argument("--eval-paths", type=int, default=None, help="Optional override for eval_paths during calibration.")
    parser.add_argument("--force-run", action="store_true", help="Recompute runs even if per-run point/empirical tables exist.")
    parser.add_argument("--no-update-config", action="store_true", help="Do not write best bound back to target JSON config.")
    parser.add_argument(
        "--proportional-cost",
        type=float,
        default=None,
        help="Optional override for proportional_cost during calibration runs (e.g., 0.001 for 0.1%%).",
    )
    parser.add_argument(
        "--include-deep",
        action="store_true",
        help="Also evaluate trained deep agents during calibration (disabled by default).",
    )
    args = parser.parse_args()

    if int(args.n_sims) <= 0:
        raise ValueError("n_sims must be > 0.")

    config_path = resolve_config_path(
        config_name=args.config_ref,
        task="compare",
        prompt_label="Enter COMPARE config name/path from 'configs': ",
    )
    source_path, merged = load_merged_config(config_path)
    pipeline = detect_pipeline(merged)
    if pipeline != "result1b":
        raise ValueError(f"This calibration script supports only pipeline='result1b'. Got: {pipeline}")

    cfg = strip_meta_keys(merged)
    cfg = _apply_relaxed_defaults_compare(cfg, pipeline=pipeline)
    if str(cfg.get("benchmark_agent_name", "")).strip() != "DeltaHedgingAgent":
        raise ValueError(
            "Target config must use benchmark_agent_name='DeltaHedgingAgent' for BS calibration."
        )
    if (
        args.eval_paths is not None
        and cfg.get("benchmark_actions_reuse_from_run", None) is not None
        and str(cfg.get("benchmark_actions_reuse_from_run", "")).strip()
    ):
        raise ValueError(
            "Cannot use --eval-paths when benchmark actions are reused from another run. "
            "Reused action cache length must match eval_paths exactly."
        )

    reuse_actions_from = cfg.get("benchmark_actions_reuse_from_run", None)
    source_eval_seed = None
    source_eval_paths = None
    if reuse_actions_from is not None and str(reuse_actions_from).strip():
        source_run_dir = _resolve_existing_run_dir(str(reuse_actions_from))
        source_eval_seed, source_eval_paths = _load_source_run_eval_params(source_run_dir)
        if int(args.n_sims) > 1:
            raise ValueError(
                "n_sims > 1 is not valid when reusing fixed benchmark actions from another run: "
                "actions are path-indexed and only consistent for a single aligned path generation. "
                "Use --n-sims 1 (recommended) or disable action reuse."
            )

    compare_dir = os.path.dirname(source_path)
    calibration_dir = os.path.normpath(os.path.join(compare_dir, "..", "calibration"))
    if args.grid_source == "calibration_configs":
        bounds = _load_grid_from_calibration_dir(calibration_dir)
        if not bounds:
            bounds = _make_grid_from_range(
                min_pct=float(args.grid_min_pct),
                max_pct=float(args.grid_max_pct),
                step_pct=float(args.grid_step_pct),
            )
    else:
        bounds = _make_grid_from_range(
            min_pct=float(args.grid_min_pct),
            max_pct=float(args.grid_max_pct),
            step_pct=float(args.grid_step_pct),
        )

    if not bounds:
        raise ValueError("No NI bounds were resolved for calibration.")

    tmp_dir = os.path.join(compare_dir, "_tmp_bs_band_calibration")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[calibration] target config: {source_path}")
    print(f"[calibration] bounds: {len(bounds)} values ({bounds[0]:.4f} .. {bounds[-1]:.4f})")
    print(f"[calibration] simulations per bound: {int(args.n_sims)}")
    print("[calibration] bootstrap disabled for all calibration runs.")
    print(
        "[calibration] evaluation mode: "
        + ("benchmark + deep targets" if bool(args.include_deep) else "benchmark-only")
    )

    simulation_rows: list[dict[str, Any]] = []
    base_run_name = str(cfg.get("run_name", "bs_band_calibration")).strip()
    default_cfg_seed = int(cfg.get("eval_seed", 34))
    base_seed = int(args.seed_start) if args.seed_start is not None else (
        int(source_eval_seed) if source_eval_seed is not None else int(default_cfg_seed)
    )
    if args.seed_start is None and source_eval_seed is not None:
        print(
            f"[calibration] seed-start not provided; aligned to source run eval_seed={int(source_eval_seed)} "
            f"from action-reuse run."
        )
    if (
        args.seed_start is None
        and source_eval_seed is None
        and reuse_actions_from is not None
        and str(reuse_actions_from).strip()
    ):
        print(
            "[calibration] source run eval_seed not available (missing resolved_config.json). "
            f"Falling back to target config eval_seed={int(default_cfg_seed)}."
        )
    if source_eval_paths is not None:
        print(f"[calibration] source action-reuse run eval_paths={int(source_eval_paths)}.")

    for bound in bounds:
        slug = _bound_slug(bound)
        for sim_index in range(int(args.n_sims)):
            eval_seed = int(base_seed) + int(sim_index)

            sim_cfg = dict(cfg)
            sim_cfg["run_name"] = f"{base_run_name}__calib_bs_band_{slug}__sim_{sim_index + 1:02d}"
            sim_cfg["bootstrap_enabled"] = False
            sim_cfg["benchmark_no_intervention_mode"] = "percentage"
            sim_cfg["benchmark_no_intervention_bound"] = float(bound)
            sim_cfg["benchmark_actions_reuse_apply_no_intervention"] = True
            sim_cfg["eval_seed"] = int(eval_seed)
            if args.proportional_cost is not None:
                sim_cfg["proportional_cost"] = float(args.proportional_cost)
            if not bool(args.include_deep):
                sim_cfg["trained_agents"] = []
                sim_cfg["benchmark_agents_to_compare"] = []
            if args.eval_paths is not None:
                sim_cfg["eval_paths"] = int(args.eval_paths)

            tmp_cfg_path = os.path.join(tmp_dir, f"band_{slug}__sim_{sim_index + 1:02d}.json")
            _save_json(tmp_cfg_path, sim_cfg)
            run_dir = _run_dir_for_config(tmp_cfg_path)

            print(
                f"[calibration] band={bound:.4f} sim={sim_index + 1}/{int(args.n_sims)} "
                f"seed={eval_seed} run_dir={run_dir}"
            )
            if not bool(args.force_run) and _has_core_metrics_outputs(run_dir):
                print("[calibration] run outputs already exist, skipping execution.")
            else:
                run_comparison(
                    run_name=str(sim_cfg["run_name"]),
                    config_path=tmp_cfg_path,
                    config=sim_cfg,
                )

            point_csv = os.path.join(run_dir, "tables", "point_metrics.csv")
            empirical_csv = os.path.join(run_dir, "tables", "empirical_risk_metrics.csv")
            if not (os.path.isfile(point_csv) and os.path.isfile(empirical_csv)):
                raise FileNotFoundError(
                    f"Missing metrics after calibration run. point='{point_csv}', empirical='{empirical_csv}'"
                )

            point_df = pd.read_csv(point_csv)
            empirical_df = pd.read_csv(empirical_csv)
            if point_df.empty or empirical_df.empty:
                raise ValueError(f"Empty metrics tables for run_dir='{run_dir}'.")

            row: dict[str, Any] = {
                "band_bound": float(bound),
                "band_pct": float(bound) * 100.0,
                "band_slug": slug,
                "simulation_index": int(sim_index + 1),
                "eval_seed": int(eval_seed),
                "run_dir": run_dir,
            }
            row.update(_extract_agent_metrics(point_df, empirical_df, row_idx=0, prefix="benchmark"))
            simulation_rows.append(row)

    sim_df = pd.DataFrame(simulation_rows).sort_values(
        by=["band_bound", "simulation_index"], kind="stable"
    )
    if sim_df.empty:
        raise ValueError("Calibration produced no simulation rows.")

    numeric_metric_cols = [
        c
        for c in sim_df.columns
        if c not in {"band_bound", "band_pct", "band_slug", "simulation_index", "eval_seed", "run_dir"}
        and pd.api.types.is_numeric_dtype(sim_df[c])
    ]

    grouped = sim_df.groupby("band_bound", sort=True)
    summary_rows: list[dict[str, Any]] = []
    for bound, group_df in grouped:
        row = {
            "band_bound": float(bound),
            "band_pct": float(bound) * 100.0,
            "band_slug": _bound_slug(float(bound)),
            "n_simulations": int(len(group_df.index)),
        }
        for col in numeric_metric_cols:
            vals = pd.to_numeric(group_df[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["band_bound"], kind="stable")

    objective_col = str(args.objective_col).strip()
    objective_mean_col = f"{objective_col}_mean"
    if objective_mean_col not in summary_df.columns:
        available = sorted([c for c in summary_df.columns if c.endswith("_mean")])
        raise ValueError(
            f"Objective '{objective_mean_col}' not found in summary table. "
            f"Available mean columns: {available}"
        )

    if str(args.objective_direction).lower() == "min":
        best_idx = int(summary_df[objective_mean_col].idxmin())
    else:
        best_idx = int(summary_df[objective_mean_col].idxmax())
    best_row = summary_df.loc[best_idx]
    best_bound = float(best_row["band_bound"])

    target_run_dir = _run_dir_for_config(source_path)
    target_tables_dir = os.path.join(target_run_dir, "tables")
    os.makedirs(target_tables_dir, exist_ok=True)

    sim_csv_target = os.path.join(target_tables_dir, "ni_band_calibration_simulations.csv")
    summary_csv_target = os.path.join(target_tables_dir, "ni_band_calibration_summary.csv")
    best_json_target = os.path.join(target_tables_dir, "ni_band_calibration_best.json")

    sim_df.to_csv(sim_csv_target, index=False)
    summary_df.to_csv(summary_csv_target, index=False)
    _save_json(
        best_json_target,
        {
            "source_config": source_path,
            "objective_col": objective_col,
            "objective_direction": str(args.objective_direction),
            "best_band_bound": float(best_bound),
            "best_band_pct": float(best_bound) * 100.0,
            "best_row": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best_row.to_dict().items()},
        },
    )

    if os.path.isdir(calibration_dir):
        sim_csv_repo = os.path.join(calibration_dir, "bs_band_calibration_simulations.csv")
        summary_csv_repo = os.path.join(calibration_dir, "bs_band_calibration_summary.csv")
        best_json_repo = os.path.join(calibration_dir, "bs_band_calibration_best.json")
        sim_df.to_csv(sim_csv_repo, index=False)
        summary_df.to_csv(summary_csv_repo, index=False)
        _save_json(
            best_json_repo,
            {
                "source_config": source_path,
                "objective_col": objective_col,
                "objective_direction": str(args.objective_direction),
                "best_band_bound": float(best_bound),
                "best_band_pct": float(best_bound) * 100.0,
            },
        )

    if not bool(args.no_update_config):
        _write_config_bound(source_path, best_bound)
        print(
            f"[calibration] Updated target config bound: {source_path} -> "
            f"benchmark_no_intervention_bound={best_bound:.6f}"
        )
    else:
        print("[calibration] Config update skipped (--no-update-config).")

    print(f"[calibration] Best NI band: {best_bound:.6f} ({best_bound * 100.0:.2f}%)")
    print(f"[calibration] Sim table: {sim_csv_target}")
    print(f"[calibration] Summary table: {summary_csv_target}")
    print(f"[calibration] Best selection: {best_json_target}")


if __name__ == "__main__":
    main()
