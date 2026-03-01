# THESIS_FINAL: Escenarios Definitivos y Runbook

## 1. Objetivo
Este documento describe en detalle los entrenamientos y comparaciones de `configs/runs/THESIS_FINAL`, exactamente con los escenarios definidos para tesis.

## 2. Carpeta de Configs
- Bases reutilizables: `configs/bases/THESIS_FINAL/train/` y `configs/bases/THESIS_FINAL/compare/`
- Runs concretos: `configs/runs/THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/...`
- En escenarios `.2` hay `calibration/band_template.json` para calibrar no-intervention band.

## 3. Mundos
- `W1_gbm_fixed`: GBM con sigma fija 20% y r fija 3%, sin contexto.
- `W2_gbm_random_ctx50`: GBM con sigma y r aleatorias uniformes por path, contexto seen de 50 pasos con Conv1D.
- `W3_garch_t_random_ctx50`: GARCH sin leverage + innovaciones t-Student, parámetros aleatorios por path, contexto 50.
- `W4_garch_t_tail_ctx50`: Igual W3 + tail shocks negativos aleatorios (magnitud 2%-10%, gaps 10-30).
- `W5_hmm3_garch_t_ctx100`: Igual W3 + HMM de 3 estados que condiciona volatilidad de largo plazo y r, contexto 100.

## 4. Subescenarios Implementados
Estos son los únicos subescenarios incluidos:
- W1: `1a_1`, `1a_1p`, `1a_2`, `1b_1`, `1b_2`, `1c_1`, `1c_2`
- W2-W5: `1a_1`, `1a_2`, `1c_1`, `1c_2`

- `1a_1_euro_tc0_cvar50`: Europea, TC=0, entrena Recurrent+LSTM minimizando CVaR50. Compare contra BS Delta + LRM MC.
- `1a_1p_euro_tc0_lstm_cvar_sweep`: Solo W1: LSTM entrenado con CVaR50/90/99, compare solo entre esos 3 LSTM (trained_only).
- `1a_2_euro_tc1_band`: Europea, TC=1%. Se reusan acciones benchmark de 1a.1 y se aplica banda no-intervention (calibrable).
- `1b_1_asian_geo_tc0_cvar50`: Solo W1: Asiática geométrica, TC=0, Recurrent+LSTM, benchmark Delta geométrico + LRM MC asiático.
- `1b_2_asian_geo_tc1_band`: Solo W1: Asiática geométrica, TC=1%, reuso de acciones benchmark de 1b.1 + banda NI.
- `1c_1_asian_arith_tc0_cvar50`: Asiática aritmética, TC=0, Recurrent+LSTM, benchmark único LRM MC asiático.
- `1c_2_asian_arith_tc1_band`: Asiática aritmética, TC=1%, benchmark LRM MC con reuso de acciones de 1c.1 + banda NI.

## 5. Reglas de Comparación y Fairness
- En los compares se usa `pricing_method: fixed` para que el precio por path sea el del primer agente y lo compartan todos los agentes comparados.
- En W2-W5 se activa estimación causal stepwise para benchmark: `benchmark_delta_sigma_mode` y `benchmark_delta_r_mode` desde contexto, sin look-ahead.
- En escenarios de TC=1% (`.2`) se reusan acciones benchmark desde `.1` (`benchmark_actions_reuse_from_run`) y se aplica banda NI en esas acciones.

## 6. Calibración de No-Intervention Band (escenarios `.2`)
Cada carpeta `.2` trae:
- `calibration/band_template.json`: plantilla para barrido de banda.
- `compare/*_banded.json`: compare final con banda ya seteada (default 0.03).

Grilla obligatoria de banda NI: desde `0.5%` hasta `10%`, en pasos de `0.5%` (i.e. `0.005, 0.010, ..., 0.100`).
En cada carpeta `calibration/` ya se generaron configs listos: `band_0005.json` ... `band_0100.json`.
La selección final debe hacerse minimizando CVaR50 del benchmark en tablas guardadas.

## 7. Comandos de Ejecución
Entrenamiento:
```bash
python examples/train_console.py THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/train/<config_sin_json>
```
Comportamiento default:
- Si el target ya está completo, no re-entrena (skip).
- Para forzar re-ejecución completa: `--force-run`.

Notas de entrenamiento THESIS_FINAL:
- `early_stopping_patience=20`.
- Extensión dinámica de tope de épocas habilitada (`max_epochs_extension_enabled=true`):
  al llegar al tope de épocas (por default 100), si el mejor epoch está dentro de los últimos
  `max_epochs_extension_window_epochs` (default 10), se extiende `max_epochs_extension_by`
  (default +10) y se repite ese criterio.

Comparación:
```bash
python examples/compare_console.py THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/compare/<config_sin_json>
```
Comportamiento default:
- Si el target de compare ya está completo, no recomputa (skip).
- Si solo falta `bootstrap_metrics_wide.csv` y existe payload raw, corre solo bootstrap.
- Para forzar re-ejecución completa: `--force-run`.

Calibración (escenarios `.2`):
```bash
python examples/compare_console.py THESIS_FINAL/<MUNDO>/<SUBESCENARIO>/calibration/band_template
```

Ejecución por carpeta (train -> compare, secuencial):
```bash
python examples/run_scenarios_console.py THESIS_FINAL/<MUNDO> --checkpoint-every-epochs 5
```
Opcional:
- `--force-run`: ignora artefactos existentes y re-ejecuta todo.

## 8. Ejemplos Directos
- `THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/recurrent`
- `THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/lstm`
- `THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_bs`
- `THESIS_FINAL/W1_gbm_fixed/1a_1p_euro_tc0_lstm_cvar_sweep/compare/lstm_cvar_sweep`
- `THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs`
- `THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded`

## 9. Inventario Completo de Configs
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1p_euro_tc0_lstm_cvar_sweep/compare/lstm_cvar_sweep.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1p_euro_tc0_lstm_cvar_sweep/train/lstm_cvar90.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_1p_euro_tc0_lstm_cvar_sweep/train/lstm_cvar99.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/compare/vs_bs_banded.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1a_2_euro_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_1_asian_geo_tc0_cvar50/compare/vs_geo_bs.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_1_asian_geo_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_1_asian_geo_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/compare/vs_geo_bs.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/compare/vs_geo_bs_banded.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1b_2_asian_geo_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_1_asian_arith_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_1_asian_arith_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W1_gbm_fixed/1c_2_asian_arith_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_2_euro_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_2_euro_tc1_band/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_2_euro_tc1_band/compare/vs_bs_banded.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_2_euro_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1a_2_euro_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_1_asian_arith_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_1_asian_arith_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_2_asian_arith_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_2_asian_arith_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W2_gbm_random_ctx50/1c_2_asian_arith_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_2_euro_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_2_euro_tc1_band/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_2_euro_tc1_band/compare/vs_bs_banded.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_2_euro_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1a_2_euro_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_1_asian_arith_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_1_asian_arith_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_2_asian_arith_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_2_asian_arith_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W3_garch_t_random_ctx50/1c_2_asian_arith_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_2_euro_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_2_euro_tc1_band/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_2_euro_tc1_band/compare/vs_bs_banded.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_2_euro_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1a_2_euro_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_1_asian_arith_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_1_asian_arith_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_2_asian_arith_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_2_asian_arith_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W4_garch_t_tail_ctx50/1c_2_asian_arith_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_1_euro_tc0_cvar50/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_1_euro_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_1_euro_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_2_euro_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_2_euro_tc1_band/compare/vs_bs.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_2_euro_tc1_band/compare/vs_bs_banded.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_2_euro_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1a_2_euro_tc1_band/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_1_asian_arith_tc0_cvar50/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_1_asian_arith_tc0_cvar50/train/lstm.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_1_asian_arith_tc0_cvar50/train/recurrent.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/calibration/band_template.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/compare/vs_lrm_mc_banded.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/train/lstm.json`
- `configs/runs/THESIS_FINAL/W5_hmm3_garch_t_ctx100/1c_2_asian_arith_tc1_band/train/recurrent.json`

## 10. Artefactos para análisis sin rerun
Cada comparación guarda tablas y caches (errores por path, métricas, acciones cacheadas, snapshots de config), suficientes para recalcular indicadores sin volver a computar acciones.

