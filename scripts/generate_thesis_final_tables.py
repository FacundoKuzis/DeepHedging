"""
Generate LaTeX tables (with bootstrap CIs) for THESIS_FINAL runs.

This script reads `bootstrap_metrics_wide.csv` files produced by compare runs and
exports LaTeX `table` environments under:

  assets/thesis_final/tables/

It is intentionally narrow and thesis-oriented (Spanish formatting, decimal comma,
booktabs + makecell).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    csv_key: str
    label: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    csv_path: Path
    agents: list[AgentSpec]
    caption: str
    label: str


METRICS: list[tuple[str, str]] = [
    ("Media", "Mean"),
    ("Desvío estándar", "StdDev"),
    ("CVaR 50\\%", "CVaR_50"),
    ("CVaR 95\\%", "CVaR_95"),
    ("CVaR 99\\%", "CVaR_99"),
    ("MAE", "MAE"),
    ("Peor caso", "WorstCase"),
]


def _fmt_number(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    s = f"{float(value):.{decimals}f}"
    if s == "-0.0000":
        s = "0.0000"
    return s.replace(".", ",")


def _metric_cell(point: float, low: float, high: float, decimals: int = 4) -> str:
    p = _fmt_number(point, decimals=decimals)
    lo = _fmt_number(low, decimals=decimals)
    hi = _fmt_number(high, decimals=decimals)
    return rf"\makecell{{{p}\\{{\scriptsize [{lo}; {hi}]}}}}"


def _read_bootstrap_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if "Agent" not in rows[0]:
        raise ValueError(f"Missing 'Agent' column in CSV: {path}")
    return rows


def _row_by_agent(rows: list[dict[str, str]], agent_key: str) -> dict[str, str]:
    for r in rows:
        if str(r.get("Agent", "")).strip() == agent_key:
            return r
    available = sorted({str(r.get("Agent", "")).strip() for r in rows if str(r.get("Agent", "")).strip()})
    raise KeyError(f"Agent '{agent_key}' not found. Available: {available}")


def _float_or_nan(raw: str) -> float:
    try:
        return float(str(raw).strip())
    except Exception:
        return float("nan")


def _best_agent_index(point_estimates: list[float], metric_key: str) -> int:
    if not point_estimates:
        return 0
    if metric_key == "Mean":
        scored = [abs(v) for v in point_estimates]
        return int(min(range(len(scored)), key=lambda i: scored[i]))
    return int(min(range(len(point_estimates)), key=lambda i: point_estimates[i]))


def _to_latex_table(spec: TableSpec) -> str:
    rows = _read_bootstrap_csv(spec.csv_path)
    agent_rows = [_row_by_agent(rows, a.csv_key) for a in spec.agents]

    col_spec = "l" + ("c" * len(spec.agents))
    header = " & ".join([r"\textbf{Métrica}"] + [rf"\textbf{{{a.label}}}" for a in spec.agents]) + r" \\"

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{spec.caption}}}")
    lines.append(rf"\label{{{spec.label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\thickhline")

    for display_name, metric_key in METRICS:
        pts = [_float_or_nan(r.get(f"{metric_key}_point_estimate", "")) for r in agent_rows]
        best_idx = _best_agent_index(pts, metric_key=metric_key)

        cells: list[str] = []
        for i, r in enumerate(agent_rows):
            point = _float_or_nan(r.get(f"{metric_key}_point_estimate", ""))
            low = _float_or_nan(r.get(f"{metric_key}_ci_lower", ""))
            high = _float_or_nan(r.get(f"{metric_key}_ci_upper", ""))
            cell = _metric_cell(point, low, high, decimals=4)
            if i == best_idx:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)

        lines.append(" & ".join([display_name] + cells) + r" \\")
        lines.append(r"\midrule")

    # Remove last midrule for cleaner output.
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "assets" / "thesis_final" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_models_root = Path(r"G:\Mi unidad\Tesis2026\Models\Organized")

    specs: list[TableSpec] = [
        TableSpec(
            name="w1_euro_tc0_vs_bs",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_bs/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("Agente delta de opción europea", "BS Delta"),
                AgentSpec("Agente Deep Hedging", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call europea (W1, TC=0\\%).",
            label="tab:w1_euro_tc0_vs_bs",
        ),
        TableSpec(
            name="w1_euro_tc0_vs_lrm",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_lrm_mc_2batches/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("Agente Local Risk Minimization", "LRM (MC)"),
                AgentSpec("Agente Deep Hedging", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call europea (W1, TC=0\\%): benchmark Local Risk Minimization.",
            label="tab:w1_euro_tc0_vs_lrm",
        ),
        TableSpec(
            name="w1_euro_tc1_vs_bs_banded",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/compare/vs_bs_banded/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("bs_delta_hedging", "BS Delta (banda)"),
                AgentSpec("recurrent", "Deep Hedging (Recurrent)"),
                AgentSpec("lstm", "Deep Hedging (LSTM)"),
            ],
            caption="Métricas del error de cobertura para una opción call europea (W1, TC=1\\%): benchmark con banda de no-intervención.",
            label="tab:w1_euro_tc1_vs_bs_banded",
        ),
        TableSpec(
            name="w1_asian_geo_tc0_vs_delta",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1b_1_asian_geo_tc0_cvar50/compare/vs_geo_bs__recurrent_only/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("asian_delta_hedging", "Delta Asia geom."),
                AgentSpec("recurrent", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call asiática geométrica (W1, TC=0\\%).",
            label="tab:w1_asian_geo_tc0_vs_delta",
        ),
        TableSpec(
            name="w1_asian_geo_tc1_vs_delta_banded",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/compare/vs_geo_bs_banded/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("asian_delta_hedging", "Delta Asia geom. (banda)"),
                AgentSpec("recurrent", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call asiática geométrica (W1, TC=1\\%): benchmark con banda de no-intervención.",
            label="tab:w1_asian_geo_tc1_vs_delta_banded",
        ),
        TableSpec(
            name="w1_asian_arith_tc0_vs_lrm",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("local_risk_minimization", "LRM (MC)"),
                AgentSpec("recurrent", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call asiática aritmética (W1, TC=0\\%).",
            label="tab:w1_asian_arith_tc0_vs_lrm",
        ),
        TableSpec(
            name="w1_asian_arith_tc1_vs_lrm_banded",
            csv_path=thesis_models_root
            / "THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded/tables/bootstrap_metrics_wide.csv",
            agents=[
                AgentSpec("local_risk_minimization", "LRM (banda)"),
                AgentSpec("recurrent", "Agente Deep Hedging"),
            ],
            caption="Métricas del error de cobertura para una opción call asiática aritmética (W1, TC=1\\%): benchmark con banda de no-intervención.",
            label="tab:w1_asian_arith_tc1_vs_lrm_banded",
        ),
    ]

    for spec in specs:
        if not spec.csv_path.is_file():
            raise FileNotFoundError(f"Missing bootstrap CSV for '{spec.name}': {spec.csv_path}")
        latex = _to_latex_table(spec)
        out_path = out_dir / f"{spec.name}.tex"
        out_path.write_text(latex, encoding="utf-8")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

